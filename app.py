from fastapi import FastAPI, UploadFile, File, Form
import PyPDF2
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()


# 🔹 Extract text from PDF
def extract_text_from_pdf(file):
    text = ""
    reader = PyPDF2.PdfReader(file)
    for page in reader.pages:
        text += page.extract_text()
    return text


# 🔹 Clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# 🔹 Main API
@app.post("/analyze")
async def analyze_resume(
    resume: UploadFile = File(...),
    job_description: str = Form(...)
):
    # Read PDF
    resume_text = extract_text_from_pdf(resume.file)

    # Clean text
    clean_resume = clean_text(resume_text)
    clean_jd = clean_text(job_description)

    # TF-IDF
    vectorizer = TfidfVectorizer(stop_words='english')
    texts = [clean_resume, clean_jd]
    tfidf_matrix = vectorizer.fit_transform(texts)

    # Cosine Similarity
    similarity_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    ats_score = similarity_score[0][0] * 100

    # 🔹 Skill Extraction (Simple TF-IDF)
    feature_names = vectorizer.get_feature_names_out()

    resume_vector = tfidf_matrix[0].toarray()[0]
    jd_vector = tfidf_matrix[1].toarray()[0]

    top_n = 10

    resume_indices = resume_vector.argsort()[-top_n:][::-1]
    jd_indices = jd_vector.argsort()[-top_n:][::-1]

    resume_skills = [feature_names[i] for i in resume_indices]
    jd_skills = [feature_names[i] for i in jd_indices]

    # 🔹 Skill Comparison
    resume_set = set(resume_skills)
    jd_set = set(jd_skills)

    matched_skills = list(resume_set.intersection(jd_set))
    missing_skills = list(jd_set - resume_set)

    # 🔹 Skill Score
    if len(jd_skills) > 0:
        skill_match_score = (len(matched_skills) / len(jd_skills)) * 100
    else:
        skill_match_score = 0

    # 🔹 Final Score
    final_score = (0.6 * ats_score) + (0.4 * skill_match_score)

    return {
        "ats_score": round(ats_score, 2),
        "skill_match_score": round(skill_match_score, 2),
        "final_score": round(final_score, 2),
        "matched_skills": matched_skills,
        "missing_skills": missing_skills
    }
