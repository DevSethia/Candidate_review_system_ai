from flask import Flask, render_template, request
from pathlib import Path
import json
import uuid
import traceback

print("Starting Candidate Review web application...")

# Import your existing pipeline
try:
    from candidate_review_gemini import run_pipeline
    print("✓ Gemini pipeline imported successfully")
except Exception as exc:
    print("✗ Failed to import candidate_review_gemini.py")
    print(f"Error: {exc}")
    traceback.print_exc()
    raise


app = Flask(__name__)

UPLOAD_FOLDER = Path("uploads")
UPLOAD_FOLDER.mkdir(exist_ok=True)

app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():

    resume = request.files.get("resume")
    transcript_text = request.form.get("transcript", "").strip()
    role = request.form.get("role", "").strip()

    if not resume:
        return render_template(
            "index.html",
            error="Please upload a resume."
        )

    if not transcript_text:
        return render_template(
            "index.html",
            error="Please enter the interview transcript."
        )

    if not role:
        return render_template(
            "index.html",
            error="Please enter the job role."
        )

    resume_name = resume.filename or ""

    if not resume_name.lower().endswith(".pdf"):
        return render_template(
            "index.html",
            error="Resume must be a PDF file."
        )

    job_id = uuid.uuid4().hex

    resume_path = UPLOAD_FOLDER / f"{job_id}_resume.pdf"
    transcript_path = UPLOAD_FOLDER / f"{job_id}_transcript.txt"
    output_path = UPLOAD_FOLDER / f"{job_id}_result.json"

    try:

        print("\n--- New candidate analysis ---")

        print("Saving uploaded files...")

        resume.save(resume_path)
        transcript_path.write_text(transcript_text, encoding="utf-8")

        print(f"Resume: {resume_path}")
        print(f"Transcript: {transcript_path}")

        print("\nStarting Gemini candidate review...")
        print("This may take a few minutes because the agents run sequentially.\n")

        run_pipeline(
            str(resume_path),
            str(transcript_path),
            role,
            str(output_path)
        )

        print("\nGemini analysis completed.")

        if not output_path.exists():
            raise RuntimeError(
                "The Gemini pipeline finished but did not create the result JSON."
            )

        print("Reading result...")

        with open(output_path, "r", encoding="utf-8") as f:
            result = json.load(f)

        print("Rendering results page...")

        return render_template(
            "result.html",
            result=result,
            role=role
        )

    except Exception as exc:

        print("\n✗ ANALYSIS FAILED")
        print(str(exc))
        traceback.print_exc()

        return render_template(
            "index.html",
            error=f"Analysis failed: {str(exc)}"
        )

    finally:

        try:
            resume_path.unlink(missing_ok=True)
            transcript_path.unlink(missing_ok=True)
            output_path.unlink(missing_ok=True)
        except Exception:
            pass


if __name__ == "__main__":

    print("\n======================================")
    print(" Candidate Review AI")
    print("======================================")
    print("Starting Flask server...")
    print("Open this address in your browser:")
    print("http://127.0.0.1:5000")
    print("======================================\n")

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )