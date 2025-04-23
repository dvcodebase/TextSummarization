from flask import Flask, render_template, request
from summarizer import summarize_text, extract_text_from_pdf
import os

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    summary = ""
    if request.method == "POST":
        file = request.files["file"]
        if file.filename.endswith(".pdf"):
            path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(path)
            text = extract_text_from_pdf(path)
        elif file.filename.endswith(".txt"):
            text = file.read().decode("utf-8")
        else:
            return "Unsupported file type"
        
        summary = summarize_text(text)

    return render_template("index.html", summary=summary)

if __name__ == "__main__":
    app.run(debug=True)
