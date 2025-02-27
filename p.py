import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import time

def extract_text_from_pdf(file):
    reader = PdfReader(file)
    text = " ".join([page.extract_text() for page in reader.pages if page.extract_text()])
    return text

def rank_resumes(job_desc, resumes):
    documents = [job_desc] + resumes
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)
    scores = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1:]).flatten()
    return scores

st.set_page_config(page_title="Resume Screening & Ranking", layout="wide")
st.title("📄 Resume Screening & Ranking System")

# Sidebar for inputs
with st.sidebar:
    st.header("Upload Details")
    job_description = st.text_area("Enter Job Description", height=150)
    uploaded_files = st.file_uploader("Upload Resumes (PDFs Only)", type=["pdf"], accept_multiple_files=True)
    process_button = st.button("Rank Resumes")

if process_button and job_description and uploaded_files:
    st.info("Processing Resumes... Please wait!")
    time.sleep(1)
    resumes_text = [extract_text_from_pdf(file) for file in uploaded_files]
    scores = rank_resumes(job_description, resumes_text)
    
    # Sorting results
    results = pd.DataFrame({"Resume": [file.name for file in uploaded_files], "Score": scores})
    results = results.sort_values(by="Score", ascending=False)
    
    # Display results
    st.subheader("📊 Ranked Resumes")
    st.dataframe(results.style.background_gradient(cmap="Blues"))
    
    # Download option
    csv = results.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download Results", data=csv, file_name="ranked_resumes.csv", mime="text/csv")
    
    st.success("Ranking Complete! 🎉")
else:
    st.warning("Please enter a job description and upload resumes to proceed.")
