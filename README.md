📄 AI Resume Analyzer (ATS Score Checker)
🚀 Live Demo

👉 Try the app here:
🔗 SkillSync AI Resume Analyzer
 

(Your deployed version: https://skillsync-ai-34cg.onrender.com/
)

🚀 Overview

The AI Resume Analyzer is a Machine Learning + NLP-based application that evaluates a candidate's resume against a given job description.

It simulates an Applicant Tracking System (ATS) by calculating a compatibility score, identifying matched and missing skills, and generating actionable suggestions to improve the resume.

This project helps job seekers optimize resumes and increase their chances of getting shortlisted.

🎯 Features
📂 Resume Upload (PDF → Text Extraction)
📝 Job Description Input
📊 ATS Score Calculation (Cosine Similarity)
🧠 Skill Extraction using NLP
✅ Matched vs ❌ Missing Skills Detection
⚖️ Weighted Score Breakdown
💡 Resume Improvement Suggestions
📈 Final Result Dashboard
🛠️ Tech Stack
Language: Python
Libraries:
scikit-learn (TF-IDF, Cosine Similarity)
nltk / spacy (Text Processing)
PyPDF2 / pdfplumber (PDF Parsing)
KeyBERT (Keyword Extraction - optional)
Frontend: Streamlit / Next.js
Backend: FastAPI
⚙️ How It Works
Upload your resume (PDF)
Enter the job description
Text preprocessing (cleaning, tokenization)
TF-IDF vectorization
Cosine similarity calculation → ATS Score
Extract skills from both resume & JD
Compare → matched & missing skills
Generate weighted score + suggestions
Display final results
📊 Output
✅ ATS Score (0–100%)
🧠 Extracted Skills
📉 Missing Skills
💡 Personalized Suggestions
💡 Future Improvements
AI-based resume rewriting
Job recommendations
Skill roadmap generation
Integration with LinkedIn / job APIs
🤝 Contribution

Feel free to fork this repo and improve it. Contributions are welcome!

📌 Author

Anurag Tiwari
💻 AI/ML Enthusiast | Full Stack Developer
