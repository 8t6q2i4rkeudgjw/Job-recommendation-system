import streamlit as st
import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import fitz  
import spacy
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from string import punctuation
from nltk.stem import WordNetLemmatizer
import base64
import io

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize NLP models 
nlp = spacy.load("en_core_web_sm")
stopwords_set = set(stopwords.words('english') + list(punctuation))
lemmatizer = WordNetLemmatizer()


JOOBLE_API_KEY = "f82c6d77-f532-4b4f-a423-a54b7e760974"
LINKEDIN_API_KEY = "linkedin_api_key"
GOOGLE_JOBS_API_KEY = "google_jobs_api_key"

if "uploaded_cvs" not in st.session_state:
    st.session_state.uploaded_cvs = []
if "job_description" not in st.session_state:
    st.session_state.job_description = ""
if "recommendations" not in st.session_state:
    st.session_state.recommendations = []
if "cv_scores" not in st.session_state:
    st.session_state.cv_scores = {}

def extract_text_from_pdf(pdf_file):
    pdf_text = ""
    with fitz.open(stream=pdf_file.read(), filetype="pdf") as doc:
        for page in doc:
            pdf_text += page.get_text()
    return pdf_text

def extract_skills(text):
    doc = nlp(text)
    skills = [token.text for token in doc if token.pos_ in ['NOUN', 'PROPN']]
    return skills

def fetch_realtime_jobs_jooble(keywords):
    url = f"https://jooble.org/api/{JOOBLE_API_KEY}"
    payload = {"keywords": keywords, "location": ""}
    headers = {"Content-Type": "application/json"}
    response = requests.post(url, json=payload, headers=headers)
    if response.status_code == 200:
        return response.json().get('jobs', [])
    return []

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    return [lemmatizer.lemmatize(token) for token in tokens if token not in stopwords_set]

def TFIDF(cv, jd):
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_jobid = tfidf_vectorizer.fit_transform(cv)
    user_tfidf = tfidf_vectorizer.transform(jd)
    cos_similarity_tfidf = map(lambda x: cosine_similarity(user_tfidf, x), tfidf_jobid)
    return list(cos_similarity_tfidf)


def create_download_link(file_data, file_name):
    b64 = base64.b64encode(file_data).decode()  
    return f'<a href="data:application/octet-stream;base64,{b64}" download="{file_name}">Download {file_name}</a>'

def create_text_download_link(text, file_name):
    b64 = base64.b64encode(text.encode()).decode()
    return f'<a href="data:text/plain;base64,{b64}" download="{file_name}">Download {file_name}</a>'


def job_seeker_section():
    st.title("Job Seeker")
    uploaded_files = st.file_uploader("Upload your CV(s)", type="pdf", accept_multiple_files=True)

    if uploaded_files:
        for file in uploaded_files:
            text = extract_text_from_pdf(file)
            skills = ' '.join(extract_skills(text))
            cv_data = {
                "file_name": file.name,
                "skills": skills,
                "text": text,
                "file_data": file.getvalue()
            }
            if not any(cv['file_name'] == cv_data['file_name'] for cv in st.session_state.uploaded_cvs):
                st.session_state.uploaded_cvs.append(cv_data)
        
    for idx, cv in enumerate(st.session_state.uploaded_cvs):
        st.subheader(f"CV {idx + 1}: {cv['file_name']}")
        st.write("Download CV PDF:")
        st.markdown(create_download_link(cv["file_data"], cv["file_name"]), unsafe_allow_html=True)
        
        
        pdf_display = f'<iframe src="data:application/pdf;base64,{base64.b64encode(cv["file_data"]).decode()}" width="700" height="500" type="application/pdf"></iframe>'
        st.markdown(pdf_display, unsafe_allow_html=True)
        st.write("---")

        
        job_data = pd.read_csv('_job_listing.csv')
        vectorizer = TfidfVectorizer()
        job_vectors = vectorizer.fit_transform(job_data['Job Description'])
        user_vector = vectorizer.transform([cv['skills']])
        similarities = cosine_similarity(user_vector, job_vectors)
        top_indices = similarities.argsort()[0][-5:][::-1]
        recommended_jobs = job_data.iloc[top_indices]

        st.write("Recommended Jobs from Dataset:")
        for i, row in recommended_jobs.iterrows():
            st.write(f"Job Title: {row['Job Title']}")
            st.write(f"Company: {row['Company Name']}")
            st.write(f"Salary Estimate: {row['Salary Estimate']}")
            st.write("---")

        # Real-Time Job Recommendations
        st.write("Real-Time Jobs from Jooble API:")
        jooble_jobs = fetch_realtime_jobs_jooble(cv['skills'])
        for job in jooble_jobs[:5]:
            st.write(f"Job Title: {job.get('title', 'N/A')}")
            st.write(f"Company: {job.get('company', 'N/A')}")
            st.write(f"Location: {job.get('location', 'N/A')}")
            st.write(f"[Apply Here]({job.get('link', '#')})")
            st.write("---")

# Recruiter Section with Candidate Scoring
def recruiter_section():
    st.title("Recruiter")
    st.session_state.job_description = st.text_area("Paste your job description here", value=st.session_state.job_description)
    num_recommendations = st.slider("Number of CV Recommendations:", min_value=1, max_value=10, step=1)

    generate_recommendations = st.button("Generate Recommendations")

    if generate_recommendations and st.session_state.job_description:
        job_text = st.session_state.job_description
        processed_jd = ' '.join(preprocess_text(job_text))
        
        cv_data = pd.DataFrame(st.session_state.uploaded_cvs)
        if not cv_data.empty:
            vectorizer = TfidfVectorizer()
            job_vector = vectorizer.fit_transform([processed_jd])
            cv_vectors = vectorizer.transform(cv_data['skills'])
            similarities = cosine_similarity(job_vector, cv_vectors).flatten()
            
            top_indices = similarities.argsort()[-num_recommendations:][::-1]
            recommended_cvs = cv_data.iloc[top_indices].drop_duplicates(subset="file_name")

            st.write("Recommended CVs:")
            score_text = "Scores:\n"
            for index, row in recommended_cvs.iterrows():
                st.write(f"CV: {row['file_name']}, Similarity Score: {similarities[index]:.2f}")
                

                pdf_display = f'<iframe src="data:application/pdf;base64,{base64.b64encode(row["file_data"]).decode()}" width="700" height="500" type="application/pdf"></iframe>'
                st.markdown(pdf_display, unsafe_allow_html=True)
                
                st.write("Download CV PDF:")
                st.markdown(create_download_link(row["file_data"], row["file_name"]), unsafe_allow_html=True)
                
                
                score_text += f"CV: {row['file_name']}, Similarity Score: {similarities[index]:.2f}, Recruiter Score: \n---\n"
            
           
            st.write("Download Scores:")
            st.markdown(create_text_download_link(score_text, "cv_scores.txt"), unsafe_allow_html=True)
        
        else:
            st.write("No CVs uploaded for recommendation.")
    elif not st.session_state.job_description:
        st.write("Please provide a job description.")

# Main App
st.sidebar.title("Job Recommendation System")
role = st.sidebar.radio("Select Role", ("Job Seeker", "Recruiter"))

if role == "Job Seeker":
    job_seeker_section()
elif role == "Recruiter":
    recruiter_section()
