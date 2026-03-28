from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import PyPDF2
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

app = FastAPI(title="SkillSync AI - ATS Analyzer")

# --------------------------
# CORS Middleware
# --------------------------
origins = [
    "http://127.0.0.1:8000",
    "http://localhost:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------
# Mount static folder
# --------------------------
app.mount("/static", StaticFiles(directory="static"), name="static")

# --------------------------
# Serve Home Page
# --------------------------
@app.get("/", response_class=FileResponse)
async def read_home():
    home_path = os.path.join("static", "home.html")
    if os.path.exists(home_path):
        return FileResponse(home_path)
    return FileResponse("static/notfound.html")  # Optional 404 page

# --------------------------
# Extract text from PDF
# --------------------------
def extract_text_from_pdf(file):
    text = ""
    reader = PyPDF2.PdfReader(file)
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text

# --------------------------
# Clean text
# --------------------------
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# --------------------------
# Analyze Resume API
# --------------------------
@app.post("/analyze")
async def analyze_resume(
    resume: UploadFile = File(...),
    job_description: str = Form(...)
):
    # Extract text from PDF
    resume_text = extract_text_from_pdf(resume.file)

    # Clean texts
    clean_resume = clean_text(resume_text)
    clean_jd = clean_text(job_description)

    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform([clean_resume, clean_jd])

    # Cosine similarity
    similarity_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    ats_score = similarity_score[0][0] * 100

    # Skill extraction (top TF-IDF words)
    feature_names = vectorizer.get_feature_names_out()
    resume_vector = tfidf_matrix[0].toarray()[0]
    jd_vector = tfidf_matrix[1].toarray()[0]

    top_n = 10
    resume_indices = resume_vector.argsort()[-top_n:][::-1]
    jd_indices = jd_vector.argsort()[-top_n:][::-1]

    resume_skills = [feature_names[i] for i in resume_indices if feature_names[i].strip()]
    jd_skills = [feature_names[i] for i in jd_indices if feature_names[i].strip()]

    # Skill comparison
    resume_set = set(resume_skills)
    jd_set = set(jd_skills)

    matched_skills = list(resume_set.intersection(jd_set))
    missing_skills = list(jd_set - resume_set)

    # Skill score
    skill_match_score = (len(matched_skills) / len(jd_skills) * 100) if jd_skills else 0

    # Final weighted score
    final_score = (0.6 * ats_score) + (0.4 * skill_match_score)

    return {
        "ats_score": round(ats_score, 2),
        "skill_match_score": round(skill_match_score, 2),
        "final_score": round(final_score, 2),
        "matched_skills": matched_skills,
        "missing_skills": missing_skills
    }
