# -*- coding: utf-8 -*-
"""
Created on Sat May 17 13:06:59 2025

@author: USER
"""

import streamlit as st
import pandas as pd
import joblib
import re
import spacy
import docx2txt
from sentence_transformers import SentenceTransformer
import numpy as np

# Load spaCy model
nlp = spacy.load("en_core_web_lg")

# Load models and vectorizers
category_model = joblib.load("resume_category_classifier.pkl")
category_vectorizer = joblib.load("resume_tfidf_vectorizer.pkl")
recommendation_model = joblib.load("xgboost_resume_match_model.pkl")
recommendation_vectorizer = joblib.load("xgboost_tfidf_vectorizer.pkl")
bert_model = SentenceTransformer("all-MiniLM-L6-v2")

# Load job descriptions
df_jobs = pd.read_csv("job_descriptions_with_full_extraction.csv")
df_jobs.fillna("", inplace=True)

# Skill dictionaries
CATEGORY_KEYWORDS = {
    "Information Technology (IT)": ["developer", "engineer", "software", "python", "cloud", "backend", "frontend"],
    "Human Resources (HR)": ["recruitment", "talent", "hr", "employee relations", "payroll"],
    "Designer (Graphic/UI/UX)": ["designer", "ui", "ux", "figma", "adobe", "prototype", "wireframe"],
    "Teacher": ["teaching", "curriculum", "lesson", "students", "classroom", "education"],
    "Advocate (Lawyer)": ["legal", "lawyer", "contract", "litigation", "case", "court"],
    "Business Development": ["sales", "business", "growth", "crm", "leads", "partnership"],
    "Healthcare Professional": ["clinic", "nurse", "hospital", "patient", "treatment", "diagnosis"],
    "Fitness Trainer": ["fitness", "trainer", "exercise", "workout", "hiit", "nutrition"],
    "Agriculture Professional": ["farming", "agriculture", "crop", "soil", "irrigation"],
    "Sales": ["sales", "quota", "lead", "pipeline", "crm", "closing"],
    "Consultant": ["consultant", "strategy", "analysis", "advisory", "presentation"],
    "Chef": ["chef", "cooking", "recipe", "menu", "kitchen", "culinary"]
}

# Define the PROFESSIONAL_SKILLS dictionary 
PROFESSIONAL_SKILLS = {
    "Information Technology (IT)": {
        "Technical Skills": [
            "Programming in Python, Java, JavaScript, C++, SQL, Go, Rust",
            "Building applications using frameworks like Django, React, Spring Boot, and Node.js",
            "Deploying and managing services on cloud platforms (AWS, Azure, GCP)",
            "Database management, optimization, and querying using MySQL, PostgreSQL, MongoDB, Redis",
            "Version control, containerization, orchestration, and CI/CD pipeline integration",
            "System security including encryption, penetration testing, and firewall configuration",
            "Linux system administration and load testing strategies"
        ],
        "Software Tools": [
            "Python", "Java", "JavaScript", "C++", "SQL", "Go", "Rust",
            "Django", "React", "Spring Boot", "Node.js",
            "AWS", "Azure", "Google Cloud Platform",
            "MySQL", "PostgreSQL", "MongoDB", "Redis",
            "Git", "GitHub", "Docker", "Kubernetes", "Jenkins"
        ]
    },

    "Human Resources (HR)": {
        "Technical Skills": [
            "Managing end-to-end recruitment and onboarding",
            "Building employee development plans and training modules",
            "Conducting performance reviews and workforce planning",
            "Handling compensation, benefits, and payroll structures",
            "Ensuring legal compliance (OSHA, EEOC, GDPR)",
            "Analyzing workforce metrics for strategic planning"
        ],
        "Software Tools": [
            "Workday", "BambooHR", "ADP", "SAP SuccessFactors",
            "LinkedIn Recruiter", "Greenhouse", "Lever",
            "Excel", "Power BI", "Google Sheets",
            "Gusto", "QuickBooks Payroll", "Zenefits"
        ]
    },

    "Designer (Graphic/UI/UX)": {
        "Technical Skills": [
            "Creating design systems and brand identities",
            "User research, journey mapping, and persona development",
            "Wireframing and prototyping interactive interfaces",
            "Applying accessibility, usability, and heuristic principles",
            "Animating and motion design for digital experiences",
            "Basic front-end coding (HTML5, CSS3, JavaScript)"
        ],
        "Software Tools": [
            "Adobe Illustrator", "Photoshop", "Figma", "Sketch", "Affinity Designer",
            "InVision", "Marvel", "Axure RP",
            "After Effects", "Lottie"
        ]
    },

    "Teacher": {
        "Technical Skills": [
            "Lesson planning aligned with educational standards",
            "Adapting teaching styles for various learning needs",
            "Conducting interactive online and offline learning sessions",
            "Designing and grading formative and summative assessments",
            "Utilizing LMS platforms and tracking student performance",
            "Creating engaging digital educational content"
        ],
        "Software Tools": [
            "Google Classroom", "Microsoft Teams", "Zoom", "Kahoot",
            "Quizlet", "Socrative", "ExamSoft",
            "Canva for Education", "PowerPoint", "Genially",
            "Moodle", "Blackboard", "Canvas",
            "Excel", "Google Sheets"
        ]
    },

    "Advocate (Lawyer)": {
        "Technical Skills": [
            "Legal drafting and contract analysis",
            "Case law research and statutory interpretation",
            "Client consultations and legal advising",
            "Litigation, court procedures, and courtroom strategy",
            "Mediation, arbitration, and dispute resolution",
            "Document review and legal compliance advisory"
        ],
        "Software Tools": [
            "LexisNexis", "Westlaw", "CaseMine",
            "Clio", "MyCase", "PracticePanther",
            "e-Discovery tools", "Notarization software"
        ]
    },

    "Business Development": {
        "Technical Skills": [
            "Identifying and qualifying new business opportunities",
            "Building and maintaining partnerships and client relationships",
            "Creating and presenting persuasive proposals and decks",
            "Conducting market research and strategic planning",
            "Optimizing outreach campaigns and marketing funnels",
            "Tracking KPIs and measuring conversion success"
        ],
        "Software Tools": [
            "Salesforce", "HubSpot", "Zoho",
            "LinkedIn Sales Navigator", "Apollo", "Mailchimp",
            "Google Analytics", "Tableau", "SEMrush",
            "MS Office Suite", "Canva for Business"
        ]
    },

    "Healthcare Professional": {
        "Technical Skills": [
            "Patient assessment, clinical diagnosis, and documentation",
            "Administering medical treatments (IV, injections, wound care)",
            "Reading and interpreting lab results and diagnostic imaging",
            "Following HIPAA regulations and infection control standards",
            "Managing chronic care and creating personalized treatment plans",
            "Handling emergency and trauma care protocols"
        ],
        "Software Tools": [
            "Epic", "Cerner", "Allscripts",
            "Stethoscopes", "EKG machines", "X-ray/MRI equipment"
        ]
    },

    "Fitness Trainer": {
        "Technical Skills": [
            "Developing customized fitness programs and regimens",
            "Conducting fitness assessments and mobility screenings",
            "Monitoring client progress and adjusting programs accordingly",
            "Educating clients on proper exercise techniques and recovery",
            "Offering basic nutritional advice and supplementation",
            "Handling emergency protocols (CPR, AED)"
        ],
        "Software Tools": [
            "Trainerize", "MyFitnessPal", "TrueCoach",
            "Fitbit", "WHOOP", "VO2 max analyzers",
            "NASM", "ACE", "Precision Nutrition", "CrossFit certifications"
        ]
    },

    "Agriculture Professional": {
        "Technical Skills": [
            "Managing soil quality, crop cycles, and yield optimization",
            "Controlling pests and applying fertilizers effectively",
            "Designing and maintaining irrigation systems",
            "Utilizing precision farming and agri-tech tools",
            "Applying sustainable farming and climate-smart techniques",
            "Post-harvest processing and storage planning"
        ],
        "Software Tools": [
            "FarmLogs", "AgriWebb", "Climate FieldView",
            "GPS-guided tractors", "Soil sensors", "Agricultural drones"
        ]
    },

    "Sales": {
        "Technical Skills": [
            "Prospecting and cold outreach",
            "Conducting discovery calls and qualifying leads",
            "Delivering sales pitches and handling objections",
            "Managing CRM pipelines and nurturing relationships",
            "Closing sales and post-sales customer support",
            "Tracking KPIs and optimizing sales workflows"
        ],
        "Software Tools": [
            "Salesforce", "Pipedrive", "Zoho CRM",
            "Apollo.io", "LinkedIn Ads", "Outreach.io",
            "Reply.io", "SalesLoft", "Mailshake",
            "Square", "Shopify POS", "Toast"
        ]
    },

    "Consultant": {
        "Technical Skills": [
            "Business analysis and problem structuring",
            "Data-driven strategy development and benchmarking",
            "Developing financial models and simulations",
            "Creating proposals, client reports, and workshops",
            "Conducting change management and stakeholder mapping",
            "Delivering client presentations and facilitating meetings"
        ],
        "Software Tools": [
            "Excel (advanced)", "Power BI", "Tableau",
            "PowerPoint", "Miro", "Notion", "Slack"
        ]
    },

    "Chef": {
        "Technical Skills": [
            "Executing precise cooking techniques (sautéing, baking, etc.)",
            "Creating innovative dishes and developing recipes",
            "Ensuring kitchen safety and hygiene compliance",
            "Managing food inventory and minimizing waste",
            "Plating and presentation for guest experience",
            "Addressing dietary restrictions and allergen safety"
        ],
        "Software Tools": [
            "Sous-vide equipment", "Induction cooktops", "Thermomix",
            "POS-linked inventory systems", "Kitchen Display Systems (KDS)",
            "HACCP checklists", "ServSafe management tools"
        ]
    }
}

# ------------------------------
# Utility Functions
# ------------------------------
def extract_text_from_pdf(uploaded_file):
    from PyPDF2 import PdfReader
    reader = PdfReader(uploaded_file)
    text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    return text

def extract_text_from_docx(uploaded_file):
    return docx2txt.process(uploaded_file)

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()

def predict_category(text):
    text_cleaned = clean_text(text)
    vec = category_vectorizer.transform([text_cleaned])
    return category_model.predict(vec)[0]

def extract_features(text, predicted_category):
    text = clean_text(text)
    doc = nlp(text)
    email_pattern = re.compile(r"[\w.-]+@[\w.-]+\.[a-zA-Z]{2,6}")
    phone_pattern = re.compile(r"\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}")

    name = ""
    for line in text.splitlines():
        if line.strip() and re.match(r'^[A-Z][a-z]+(\s[A-Z][a-z]+)+$', line.strip(), re.IGNORECASE):
            name = line.strip()
            break

    phone = phone_pattern.findall(text)
    email = email_pattern.findall(text)

    edu_pattern = re.compile(r"(?i)(bachelor|master|phd|associate).*?(?:in\s+)?([A-Z][a-zA-Z\s]+),?\s+([A-Z][a-zA-Z\s]+)")
    education_matches = edu_pattern.findall(text)
    education = [f"{match[0].title()} in {match[1].strip()}, {match[2].strip()}" for match in education_matches]

    exp_pattern = re.compile(r"([A-Z][a-zA-Z\s]+),\s*([A-Z][a-zA-Z\s]+)\s*[—-]\s*(\d{4})\s*to\s*(Present|\d{4})")
    experience_matches = exp_pattern.findall(text)
    experience = [" - ".join(match) for match in experience_matches]

    SOFT_SKILLS = {"communication", "teamwork", "leadership", "problem solving", "time management",
                   "adaptability", "creativity", "work ethic", "attention to detail", "interpersonal skills",
                   "collaboration", "empathy", "active listening", "critical thinking", "decision making",
                   "conflict resolution", "flexibility", "self-motivation", "organization", "negotiation"}

    found_soft_skills = set()
    for token in doc:
        for skill in SOFT_SKILLS:
            skill_vector = nlp(skill).vector
            token_vector = token.vector
            similarity = token_vector.dot(skill_vector) / (np.linalg.norm(token_vector) * np.linalg.norm(skill_vector))
            if similarity > 0.7:
                found_soft_skills.add(skill)

    # Dynamic category-based technical skills
    found_tech = set()
    found_tools = set()
    if predicted_category in PROFESSIONAL_SKILLS:
        for token in doc:
            for skill in PROFESSIONAL_SKILLS[predicted_category]["Technical Skills"]:
                if skill.lower() in text:
                    found_tech.add(skill)
            for tool in PROFESSIONAL_SKILLS[predicted_category]["Software Tools"]:
                if tool.lower() in text:
                    found_tools.add(tool)

    return {
        "Name": name,
        "Phone": phone[0] if phone else "",
        "Email": email[0] if email else "",
        "Education": ", ".join(education),
        "Soft Skills": ", ".join(found_soft_skills),
        "Technical Skills": ", ".join(found_tech),
        "Software Tools": ", ".join(found_tools),
        "Experience": " | ".join(experience)
    }

def get_top_5_recommendations(resume_text, resume_category):
    resume_clean = clean_text(resume_text)
    resume_vec = recommendation_vectorizer.transform([resume_clean])
    resume_embed = bert_model.encode(resume_clean)

    job_scores = []
    for _, j in df_jobs[df_jobs["Category"] == resume_category].iterrows():
        job_clean = clean_text(f"{j['Extracted Education']} {j['Extracted Skills']} {j['Extracted Duties']}")
        job_vec = recommendation_vectorizer.transform([job_clean])
        job_embed = bert_model.encode(job_clean)

        combined_vec = np.hstack((resume_vec.toarray()[0], job_vec.toarray()[0]))
        match_prob = recommendation_model.predict_proba([combined_vec])[0][1]

        similarity = np.dot(resume_embed, job_embed) / (np.linalg.norm(resume_embed) * np.linalg.norm(job_embed))

        job_scores.append({
            "Job Title": j["Job Title"],
            "Company": j.get("Company Name", ""),
            "Category": j["Category"],
            "Match Probability": round(match_prob, 3),
            "BERT Similarity": round(similarity, 3)
        })

    return sorted(job_scores, key=lambda x: x["Match Probability"], reverse=True)[:5]

# ------------------------------
# Streamlit App
# ------------------------------
st.set_page_config(page_title="Resume Matcher", layout="wide")
st.title("🔍 Resume Analysis & Job Matching")

uploaded_file = st.file_uploader("Upload your Resume (PDF or DOCX)", type=["pdf", "docx"])

if uploaded_file:
    if uploaded_file.name.endswith(".pdf"):
        resume_text = extract_text_from_pdf(uploaded_file)
    else:
        resume_text = extract_text_from_docx(uploaded_file)

    st.subheader("📄 Uploaded Resume")
    st.text(resume_text[:1000] + "..." if len(resume_text) > 1000 else resume_text)

    st.subheader("📌 Predicted Resume Category")
    predicted_category = predict_category(resume_text)
    st.success(f"Predicted Category: {predicted_category}")

    st.subheader("🔍 Extracted Resume Features")
    extracted = extract_features(resume_text, predicted_category)
    st.dataframe(pd.DataFrame([extracted]))

    st.subheader("📈 Top 5 Job Recommendations")
    recommendations = get_top_5_recommendations(resume_text, predicted_category)
    st.table(pd.DataFrame(recommendations))
