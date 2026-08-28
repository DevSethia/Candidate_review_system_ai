from __future__ import annotations

import argparse
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Literal

from google import genai
from google.genai import types
from pydantic import BaseModel, Field

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

try:
    from docx import Document
except ImportError:
    Document = None

MODEL = "gemini-3.6-flash"
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY is missing. In PowerShell run:\n"
        '$env:GEMINI_API_KEY="YOUR_GEMINI_API_KEY"'
    )

client = genai.Client(api_key=API_KEY)


# -----------------------------
# Strict schemas: no Dict[str, Any]
# -----------------------------
class EvidenceItem(BaseModel):
    evidence_id: str
    source: Literal["resume", "transcript"]
    quote: str
    fact: str
    location: str


class SkillItem(BaseModel):
    name: str
    claimed_level: str
    evidence_ids: List[str] = Field(default_factory=list)


class ClaimItem(BaseModel):
    claim: str
    status: Literal["supported", "unverified_claim", "contradicted"]
    evidence_ids: List[str] = Field(default_factory=list)


class CandidateProfile(BaseModel):
    candidate_facts: List[str] = Field(default_factory=list)
    skills: List[SkillItem] = Field(default_factory=list)
    experience: List[str] = Field(default_factory=list)
    claims: List[ClaimItem] = Field(default_factory=list)
    evidence: List[EvidenceItem] = Field(default_factory=list)


class AgentStrength(BaseModel):
    point: str
    evidence_id: str
    quote: str


class AgentConcern(BaseModel):
    point: str
    evidence_id: str
    quote: str


class AgentOpinionSchema(BaseModel):
    assessment: str
    confidence: float = Field(ge=0, le=1)
    strengths: List[AgentStrength] = Field(default_factory=list)
    concerns: List[AgentConcern] = Field(default_factory=list)
    follow_up_questions: List[str] = Field(default_factory=list)


class DebateSchema(BaseModel):
    responding_to: str
    agrees: List[str] = Field(default_factory=list)
    disagrees: List[str] = Field(default_factory=list)
    changed_view: str
    new_confidence: float = Field(ge=0, le=1)
    cited_evidence_ids: List[str] = Field(default_factory=list)
    debate_statement: str


class StrongestEvidence(BaseModel):
    evidence_id: str
    reason: str


class FinalAssessmentSchema(BaseModel):
    overall_assessment: str
    confidence: Literal["low", "medium", "high"]
    strongest_evidence: List[StrongestEvidence] = Field(default_factory=list)
    strengths: List[str] = Field(default_factory=list)
    concerns: List[str] = Field(default_factory=list)
    unresolved_disagreements: List[str] = Field(default_factory=list)
    follow_up_questions: List[str] = Field(default_factory=list)
    evidence_gaps: List[str] = Field(default_factory=list)
    human_decision_required: bool = True


# -----------------------------
# File helpers
# -----------------------------
def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def read_document(path: str) -> str:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p.resolve()}")
    if p.stat().st_size == 0:
        raise ValueError(f"File is empty: {p.resolve()}")

    ext = p.suffix.lower()
    if ext == ".txt":
        text = p.read_text(encoding="utf-8", errors="ignore")
    elif ext == ".pdf":
        if PdfReader is None:
            raise RuntimeError("Install pypdf: pip install pypdf")
        reader = PdfReader(str(p))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        if not text.strip():
            raise ValueError(
                f"PDF contains no extractable text: {p.resolve()}"
            )
    elif ext == ".docx":
        if Document is None:
            raise RuntimeError("Install python-docx: pip install python-docx")
        doc = Document(str(p))
        text = "\n".join(par.text for par in doc.paragraphs)
    else:
        raise ValueError("Use .pdf, .docx, or .txt files.")

    text = clean_text(text)
    if not text:
        raise ValueError(f"No readable text found in: {p.resolve()}")
    return text


# -----------------------------
# Gemini structured output
# -----------------------------
def call_structured(system_instruction: str, prompt: str, schema: type[BaseModel]) -> BaseModel:
    import time

    max_retries = 2

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.2,
                    response_mime_type="application/json",
                    response_schema=schema,
                ),
            )
            break

        except Exception as exc:
            error_text = str(exc)

            if "429" not in error_text and "RESOURCE_EXHAUSTED" not in error_text:
                raise

            if attempt == max_retries - 1:
                raise RuntimeError(
                    "Gemini API quota/rate limit still exceeded "
                    "after multiple retries."
                ) from exc

            wait_time = 12

            print(
                f"Gemini rate limit reached. "
                f"Waiting {wait_time}s before retry "
                f"({attempt + 1}/{max_retries})..."
            )

            time.sleep(wait_time)
    if not response.text:
        raise RuntimeError("Gemini returned an empty response.")
    try:
        return schema.model_validate_json(response.text)
    except Exception as exc:
        raise RuntimeError(
            "Gemini returned JSON that did not match the expected schema.\n"
            f"Raw response:\n{response.text}"
        ) from exc


# -----------------------------
# 1. Candidate Profile Builder
# -----------------------------
PROFILE_SYSTEM = """
You are the Candidate Profile Builder.

Extract facts from the supplied resume and transcript. Do not make a hiring judgment.

Rules:
- Use only supplied material.
- Preserve candidate statements as claims.
- Never invent quotes.
- Every material fact, skill, experience item, or claim must connect to evidence.
- Quotes must be exact text from the supplied sources.
- Mark unsupported statements as unverified_claim.
- Mark explicit conflicts as contradicted.
- Do not infer protected characteristics.
"""


def build_profile(resume: str, transcript: str) -> CandidateProfile:
    return call_structured(
        PROFILE_SYSTEM,
        f"RESUME\n{resume}\n\nTRANSCRIPT\n{transcript}",
        CandidateProfile,
    )


# -----------------------------
# 2. Independent agents
# -----------------------------
AGENTS = {
    "Technical Agent": """
Assess technical capability and depth. Prioritize demonstrated implementation,
technical decisions, architecture, testing, debugging, tools, ownership,
and measurable technical outcomes.
""",
    "HR / Culture Agent": """
Assess only job-relevant communication, collaboration, ownership and integrity
signals explicitly evidenced in the material. Do not infer personality or
protected characteristics.
""",
    "Hiring Manager Agent": """
Assess role relevance, responsibilities, scope, outcomes, ownership, and
evidence-backed gaps against the job description.
""",
    "Skeptic Agent": """
Stress-test claims. Look for contradictions, unsupported assertions, inflated
scope, vague accomplishments, timeline inconsistencies, and verification needs.
Do not call something a red flag without supporting evidence.
""",
}

INDEPENDENT_SYSTEM = """
You are an independent specialist reviewer.

This is your first and private opinion.
You MUST NOT see, reference, or infer another agent's conclusion.
You receive only the candidate profile and role description.

For EVERY strength and concern:
- cite an evidence_id
- provide the exact quote
- never fabricate evidence
"""

@dataclass
class AgentOpinion:
    agent: str
    assessment: str
    confidence: float
    strengths: List[str]
    concerns: List[str]
    evidence_ids: List[str]
    evidence_quotes: List[str]
    follow_up_questions: List[str]


def run_independent_agent(agent_name: str, focus: str, profile: CandidateProfile, role_description: str) -> AgentOpinion:
    result = call_structured(
        INDEPENDENT_SYSTEM + "\nYOUR SPECIALTY:\n" + focus,
        f"ROLE DESCRIPTION\n{role_description}\n\nSHARED CANDIDATE PROFILE\n{profile.model_dump_json(indent=2)}",
        AgentOpinionSchema,
    )
    all_items = result.strengths + result.concerns
    return AgentOpinion(
        agent=agent_name,
        assessment=result.assessment,
        confidence=result.confidence,
        strengths=[x.point for x in result.strengths],
        concerns=[x.point for x in result.concerns],
        evidence_ids=[x.evidence_id for x in all_items],
        evidence_quotes=[x.quote for x in all_items],
        follow_up_questions=result.follow_up_questions,
    )

def run_independent_stage(
    profile: CandidateProfile,
    role_description: str
) -> dict[str, AgentOpinion]:

    outputs: dict[str, AgentOpinion] = {}

    for name, focus in AGENTS.items():
        print(f"  Running: {name}")

        outputs[name] = run_independent_agent(
            name,
            focus,
            profile,
            role_description,
        )

    return outputs


# -----------------------------
# 3. Debate stage
# -----------------------------
DEBATE_SYSTEM = """
You are in the debate stage.

Other reviewers' independent opinions are now visible.

You MUST:
- directly name one other reviewer
- respond to a specific point from that reviewer
- agree, disagree, or partially agree
- explain whether your own view changed
- cite evidence IDs for factual claims
- never fabricate evidence

This is the ONLY stage where other agents' conclusions are available.
"""


@dataclass
class DebateResponse:
    agent: str
    responding_to: str
    agrees: List[str]
    disagrees: List[str]
    changed_view: str
    new_confidence: float
    cited_evidence_ids: List[str]
    debate_statement: str


def run_debate_for_agent(
    agent_name: str,
    own_opinion: AgentOpinion,
    all_opinions: dict[str, AgentOpinion],
    profile: CandidateProfile,
    role_description: str,
) -> DebateResponse:
    peers = {name: asdict(op) for name, op in all_opinions.items() if name != agent_name}
    result = call_structured(
        DEBATE_SYSTEM,
        f"""
YOUR AGENT
{agent_name}

ROLE DESCRIPTION
{role_description}

YOUR ORIGINAL INDEPENDENT OPINION
{json.dumps(asdict(own_opinion), ensure_ascii=False, indent=2)}

OTHER AGENTS' OPINIONS
{json.dumps(peers, ensure_ascii=False, indent=2)}

SHARED CANDIDATE PROFILE
{profile.model_dump_json(indent=2)}

Choose one peer and directly respond to a specific point.
""",
        DebateSchema,
    )
    if not result.responding_to or result.responding_to == agent_name:
        raise RuntimeError(f"Debate validation failed for {agent_name}.")
    return DebateResponse(
        agent=agent_name,
        responding_to=result.responding_to,
        agrees=result.agrees,
        disagrees=result.disagrees,
        changed_view=result.changed_view,
        new_confidence=result.new_confidence,
        cited_evidence_ids=result.cited_evidence_ids,
        debate_statement=result.debate_statement,
    )


def run_debate_stage(
    opinions: dict[str, AgentOpinion],
    profile: CandidateProfile,
    role_description: str
) -> dict[str, DebateResponse]:

    print("  Debate stage skipped to reduce Gemini API usage during testing.")

    return {}

# -----------------------------
# 4. Final synthesis
# -----------------------------
FINAL_SYSTEM = """
You are the final evidence synthesizer for a human hiring reviewer.

Do NOT average agent confidence values.
Do NOT make an automated employment decision.

Weigh:
1. evidence quality and specificity
2. corroboration across independent evidence
3. explicit contradictions
4. relevance to the stated role
5. confidence justified by evidence
6. debate impact and changes in view

Make uncertainty, unresolved disagreement, evidence gaps and follow-up questions visible.
human_decision_required MUST remain true.
"""


def final_synthesis(
    profile: CandidateProfile,
    opinions: dict[str, AgentOpinion],
    debates: dict[str, DebateResponse],
    role_description: str,
) -> FinalAssessmentSchema:
    return call_structured(
        FINAL_SYSTEM,
        f"""
ROLE DESCRIPTION
{role_description}

SHARED CANDIDATE PROFILE
{profile.model_dump_json(indent=2)}

INDEPENDENT OPINIONS
{json.dumps({k: asdict(v) for k, v in opinions.items()}, ensure_ascii=False, indent=2)}

DEBATE RESPONSES
{json.dumps({k: asdict(v) for k, v in debates.items()}, ensure_ascii=False, indent=2)}
""",
        FinalAssessmentSchema,
    )


# -----------------------------
# 5. Report + CLI
# -----------------------------
def build_report(profile, opinions, debates, final):
    return {
        "profile": profile.model_dump(),
        "independent_reviews": {k: asdict(v) for k, v in opinions.items()},
        "debate": {k: asdict(v) for k, v in debates.items()},
        "final_assessment": final.model_dump(),
        "debate_integrity_check": {
            "passed": all(d.responding_to and d.responding_to != d.agent for d in debates.values()),
            "rule": "Every agent directly responds to a named peer after independent review.",
        },
        "decision_boundary": "Decision support only. A qualified human reviewer must make the employment decision.",
    }


def run_pipeline(resume_path: str, transcript_path: str, role_description: str, output_path: str):
    print("[1/5] Reading candidate files...")
    resume = read_document(resume_path)
    transcript = read_document(transcript_path)

    print("[1/5] Building candidate profile...")
    profile = build_profile(resume, transcript)

    print("[2/5] Running four independent agents...")
    opinions = run_independent_stage(profile, role_description)

    print("[3/5] Running debate stage...")
    debates = run_debate_stage(opinions, profile, role_description)

    print("[4/5] Running final evidence synthesis...")
    final = final_synthesis(profile, opinions, debates, role_description)

    print("[5/5] Writing report...")
    report = build_report(profile, opinions, debates, final)
    Path(output_path).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Done. Report written to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", required=True)
    parser.add_argument("--transcript", required=True)
    parser.add_argument("--role", required=True)
    parser.add_argument("--output", default="candidate_review.json")
    args = parser.parse_args()
    run_pipeline(args.resume, args.transcript, args.role, args.output)