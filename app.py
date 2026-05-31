from flask import Flask, render_template, request, send_file
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from PyPDF2 import PdfReader
import re
import sqlite3
import io
from reportlab.pdfgen import canvas
import webbrowser
import threading

app = Flask(__name__)

# ---------------- DATABASE ----------------
def init_db():
    conn = sqlite3.connect('resumes.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            role TEXT,
            skills TEXT,
            missing TEXT
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# ---------------- ROUTES ----------------
@app.route('/')
def dashboard():
    return render_template('index.html')

@app.route('/roles')
def roles():
    return render_template('roles.html')

@app.route('/settings')
def settings():
    return render_template('settings.html')

@app.route('/history')
def history():
    conn = sqlite3.connect('resumes.db')
    c = conn.cursor()
    c.execute("SELECT * FROM history ORDER BY id DESC")
    data = c.fetchall()
    conn.close()
    return render_template("history.html", data=data)

# ---------------- SKILLS ----------------
skills_list = [
    "python", "java", "html", "css", "javascript", "react",
    "sql", "nodejs", "machine learning", "deep learning",
    "pandas", "excel", "power bi", "kotlin", "swift",
    "docker", "kubernetes", "aws", "linux", "c", "c++",
    "tensorflow", "nlp", "figma", "selenium"
]

# ---------------- ROLE SKILLS ----------------
role_skills = {
    "Web Developer": ["html", "css", "javascript"],
    "Frontend Developer": ["html", "css", "javascript", "react"],
    "Backend Developer": ["java", "python", "sql", "nodejs"],
    "Full Stack Developer": ["html", "css", "javascript", "react", "nodejs", "sql"],
    "ML Engineer": ["python", "machine learning", "deep learning", "pandas"],
    "Data Scientist": ["python", "machine learning", "statistics"],
    "Data Analyst": ["python", "sql", "excel", "power bi"],
    "DevOps Engineer": ["linux", "docker", "kubernetes", "aws"],
    "AI Engineer": ["python", "deep learning", "nlp", "tensorflow"]
}

# ---------------- TRAINING DATA ----------------
training_data = [
    ("python machine learning deep learning ai", "ML Engineer"),
    ("python pandas sql excel data analysis", "Data Analyst"),
    ("html css javascript react frontend", "Frontend Developer"),
    ("html css javascript nodejs mongodb backend", "Full Stack Developer"),
    ("java spring backend api sql", "Backend Developer"),
    ("aws docker kubernetes devops linux", "DevOps Engineer"),
    ("python tensorflow nlp ai ml", "AI Engineer")
]

training_data += [
    ("python pytorch nlp deep learning ai research", "ML Engineer"),
    ("python sql spark etl big data pipeline", "Data Analyst"),
    ("excel power bi sql reporting business analysis", "Data Analyst"),
    ("linux networking system admin troubleshooting", "DevOps Engineer"),
    ("html css javascript react ui frontend", "Frontend Developer")
]

texts = [x[0] for x in training_data]
labels = [x[1] for x in training_data]

vectorizer = CountVectorizer(ngram_range=(1, 2))
X = vectorizer.fit_transform(texts)

model = MultinomialNB()
model.fit(X, labels)

# ---------------- PDF EXTRACTION ----------------
def extract_text_from_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text.lower()

# ---------------- ANALYZE TEXT ----------------
@app.route('/analyze', methods=['POST'])
def analyze():
    resume = request.form['resume'].lower()
    return process_resume(resume)

# ---------------- ANALYZE PDF ----------------
@app.route('/analyze_pdf', methods=['POST'])
def analyze_pdf():
    file = request.files['resume_pdf']
    resume_text = extract_text_from_pdf(file)
    return process_resume(resume_text)

# ---------------- CORE FUNCTION ----------------
def process_resume(resume):

    tokens = set(re.findall(r"[a-zA-Z+#]+", resume))

    found_skills = [skill for skill in skills_list if skill in tokens]

    if len(found_skills) == 0:
        return render_template(
            'result.html',
            skills=[],
            role="Unknown",
            missing=[],
            match=0
        )

    input_text = " ".join(found_skills)
    X_test = vectorizer.transform([input_text])
    role = model.predict(X_test)[0]

    required = set(role_skills.get(role, []))
    found = set(found_skills)

    missing = list(required - found)

    match_percent = int((len(found & required) / len(required)) * 100) if required else 0

    # SAVE TO DB
    conn = sqlite3.connect('resumes.db')
    c = conn.cursor()
    c.execute(
        "INSERT INTO history (role, skills, missing) VALUES (?, ?, ?)",
        (role, ", ".join(found_skills), ", ".join(missing))
    )
    conn.commit()
    conn.close()

    return render_template(
        'result.html',
        skills=found_skills,
        role=role,
        missing=missing,
        match=match_percent
    )

# ---------------- PDF DOWNLOAD ----------------
@app.route('/download')
def download():
    conn = sqlite3.connect('resumes.db')
    c = conn.cursor()
    c.execute("SELECT * FROM history ORDER BY id DESC LIMIT 1")
    data = c.fetchone()
    conn.close()

    buffer = io.BytesIO()
    p = canvas.Canvas(buffer)

    p.drawString(100, 800, "AI Resume Analysis Report")
    p.drawString(100, 770, f"Role: {data[1]}")
    p.drawString(100, 740, f"Skills: {data[2]}")
    p.drawString(100, 710, f"Missing: {data[3]}")

    p.save()
    buffer.seek(0)

    return send_file(buffer, as_attachment=True, download_name="report.pdf")

# ---------------- RUN (OPEN IN BROWSER) ----------------
def open_browser():
    webbrowser.open("http://127.0.0.1:5000/")

if __name__ == '__main__':
    threading.Timer(1.2, open_browser).start()
    app.run(debug=True)