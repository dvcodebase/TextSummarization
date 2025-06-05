# from flask import Flask, render_template, request, flash, redirect, url_for
# from werkzeug.utils import secure_filename
# from summarizer import summarize_large_text, extract_text_from_pdf
# import os

# app = Flask(__name__)
# app.secret_key = "your_secret_key"  # needed for flash messages

# UPLOAD_FOLDER = "uploads"
# ALLOWED_EXTENSIONS = {"pdf", "txt"}

# os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# def allowed_file(filename):
#     return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# @app.route("/", methods=["GET", "POST"])
# def index():
#     summary = ""
#     if request.method == "POST":
#         if "file" not in request.files:
#             flash("No file part in the request")
#             return redirect(request.url)
        
#         file = request.files["file"]
        
#         if file.filename == "":
#             flash("No file selected")
#             return redirect(request.url)
        
#         if file and allowed_file(file.filename):
#             filename = secure_filename(file.filename)
#             path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
#             file.save(path)
            
#             if filename.endswith(".pdf"):
#                 text = extract_text_from_pdf(path)
#             elif filename.endswith(".txt"):
#                 with open(path, "r", encoding="utf-8") as f:
#                     text = f.read()
#             else:
#                 flash("Unsupported file type")
#                 return redirect(request.url)
            
#             summary = summarize_large_text(text)
#         else:
#             flash("Allowed file types are: pdf, txt")
#             return redirect(request.url)

#     return render_template("index.html", summary=summary)

# if __name__ == "__main__":
#     app.run(debug=True)


from flask import Flask, render_template, request
from summarizer import extract_text_from_pdf, summarize_large_text
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    print("Summarize route called!")
    if 'file' not in request.files:
        return "No file uploaded", 400
    file = request.files['file']
    if file.filename == '':
        return "No file selected", 400
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)
    
    try:
        text = extract_text_from_pdf(file_path)
        summary = summarize_large_text(text)
    finally:
        if os.path.exists(file_path):
            os.remove(file_path) # # Clean up only if file still exists                 
    return render_template('index.html', summary=summary)

if __name__ == '__main__':
    app.run(debug=True)