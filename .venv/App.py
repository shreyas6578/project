# Standard Library Imports
import os
import io
import re
import time
import json
import base64
import random
import string
import datetime
from html import escape

# Data Processing & Analysis
import numpy as np
import pandas as pd
from rapidfuzz import fuzz

# Natural Language Processing - MUST COME FIRST
import nltk

# Configure NLTK data path
nltk_dir = os.path.join(os.path.expanduser("~"), "nltk_data")
os.makedirs(nltk_dir, exist_ok=True)
nltk.data.path.append(nltk_dir)

# Download required NLTK resources with comprehensive fallbacks
def setup_nltk():
    required_resources = [
        ('punkt', 'tokenizers/punkt'),
        ('averaged_perceptron_tagger', 'taggers/averaged_perceptron_tagger'),
        ('averaged_perceptron_tagger_eng', 'taggers/averaged_perceptron_tagger_eng'),  # Explicitly added
        ('stopwords', 'corpora/stopwords'),
        ('wordnet', 'corpora/wordnet')
    ]
    
    for resource, path in required_resources:
        try:
            # First try normal download
            nltk.download(resource, quiet=True)
            print(f"Successfully downloaded {resource}")
        except Exception as e:
            print(f"Error downloading {resource}: {e}")
            try:
                # Fallback to direct download to specified directory
                nltk.download(resource, download_dir=nltk_dir, quiet=True)
                print(f"Downloaded {resource} to {nltk_dir}")
            except Exception as e2:
                print(f"Critical error downloading {resource}: {e2}")
                if resource == 'averaged_perceptron_tagger_eng':
                    # Special handling for this problematic resource
                    try:
                        import urllib.request
                        url = "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/taggers/averaged_perceptron_tagger.zip"
                        dest = os.path.join(nltk_dir, "taggers", "averaged_perceptron_tagger_eng.zip")
                        os.makedirs(os.path.dirname(dest), exist_ok=True)
                        urllib.request.urlretrieve(url, dest)
                        print("Manually downloaded averaged_perceptron_tagger_eng")
                    except Exception as e3:
                        print(f"Failed manual download: {e3}")
                        raise RuntimeError(f"Could not obtain required NLTK resource: {resource}")

setup_nltk()

# Now import NLTK components after ensuring resources are available
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.corpus import stopwords

# Rest of your imports...
import language_tool_python
import textstat
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pyresparser import ResumeParser
from docx import Document
import requests
import pymysql
import pafy
import matplotlib.pyplot as plt
import plotly.express as px
import streamlit as st
from streamlit_tags import st_tags
from PIL import Image
import google.generativeai as genai

# Local Imports
from Courses import (
    ds_kw, ml_kw, web_kw, fullstack_kw, android_kw, ios_kw, uiux_kw,
    devops_kw, qa_kw, dataeng_kw, cloud_kw, biz_analytics_kw, cybersec_kw,
    blockchain_kw, game_dev_kw, embedded_kw, network_kw, pm_kw,
    ds_course, ml_course, web_course, fullstack_course, android_course,
    ios_course, uiux_course, devops_course, qa_course, dataeng_course,
    cloud_course, biz_analytics_course, cybersec_course, blockchain_course,
    game_dev_course, embedded_course, network_course, pm_course,
    resume_videos, interview_videos
)

# Initialize grammar tool
tool = language_tool_python.LanguageTool("en-US")

# Final verification
try:
    # Test if POS tagger works
    from nltk.tag import PerceptronTagger
    tagger = PerceptronTagger()
    print("POS tagger initialized successfully")
except Exception as e:
    print(f"POS tagger initialization failed: {e}")
    # Fallback to universal tagger
    from nltk.tag import UnigramTagger
    tagger = UnigramTagger(model='universal')
    print("Using universal tagger as fallback")

def skill_match_section(resume_skills):
    st.subheader("🔗 Skill–Job Matching")
    
    # Text area for job description
    jd = st.text_area("Paste Job Description for skill matching", height=150)
    
    # Add matching button
    if st.button("✨ Analyze Skill Match"):
        if not jd:
            st.info("Please paste a job description first")
            return
        
        with st.spinner("Analyzing skill match..."):
            jd_skills = extract_skills_from_jd(jd)
            score, matched, missing = compute_skill_match(resume_skills, jd_skills)
            
            # Display results
            st.metric("Match Score", f"{score}%")
            st.success(f"✅ Matched: {', '.join(matched) or 'None'}")
            st.warning(f"❌ Missing: {', '.join(missing) or 'None'}")
            
            if jd_skills:
                fig, ax = plt.subplots()
                ax.bar(["Matched", "Missing"], [len(matched), len(missing)])
                st.pyplot(fig)
            else:
                st.warning("No skills detected in the job description")

def extract_skills_from_jd(text):
    try:
        # Preprocess text
        text = re.sub(r'[^\w\s]', '', text.lower())
        
        # Tokenize and tag
        tokens = word_tokenize(text)
        tagged = pos_tag(tokens)
        
        # Extract potential skills using POS patterns
        skills = []
        grammar_pattern = r"""
            NP: {<NN.*>+}  # Noun phrases
            SKILL: {<NN.*>+<VBG>?}  # Nouns + optional gerund
        """
        chunker = nltk.RegexpParser(grammar_pattern)
        tree = chunker.parse(tagged)
        
        # Extract chunks
        for subtree in tree.subtrees():
            if subtree.label() in ['NP', 'SKILL']:
                skill = " ".join(word for word, tag in subtree.leaves())
                skills.append(skill)
        
        return list(set(skills))
    
    except Exception as e:
        st.error(f"Skill extraction error: {str(e)}")
        return []

def compute_skill_match(resume_skills, job_skills):
    try:
        matched = []
        missing = []
        
        # Normalize skills
        resume_skills = [str(s).lower().strip() for s in resume_skills]
        job_skills = [str(s).lower().strip() for s in job_skills]
        
        # Create similarity matrix with type safety
        similarity_matrix = np.zeros((len(job_skills), len(resume_skills)), dtype=np.float16)
        
        # Calculate similarity scores
        for j, j_skill in enumerate(job_skills):
            for r, r_skill in enumerate(resume_skills):
                similarity_matrix[j][r] = fuzz.token_set_ratio(j_skill, r_skill)
        
        # Find matches using threshold
        for j, j_skill in enumerate(job_skills):
            max_sim = np.max(similarity_matrix[j])
            if max_sim > 75:
                matched.append(j_skill)
            else:
                missing.append(j_skill)
        
        # Calculate score safely
        score = int((len(matched) / len(job_skills) * 100)) if job_skills else 0
        
        # Update session state with successful results
        st.session_state.last_successful_match = {
            'score': score,
            'matched': matched,
            'missing': missing
        }
        
        return score, matched, missing
    
    except Exception as e:
        st.error(f"Match analysis error: {str(e)}")
        
        # Return last successful results if available
        if 'last_successful_match' in st.session_state:
            st.warning("Showing last successful analysis results")
            return (
                st.session_state.last_successful_match['score'],
                st.session_state.last_successful_match['matched'],
                st.session_state.last_successful_match['missing']
            )
        
        return 0, [], []
    
    except Exception as e:
        st.error(f"Matching error: {str(e)}")
        return 0, [], []  # Return defaults without closing outputs
    
# ----------------- Helper Functions -----------------
def generate_docx(text):
    """Convert text to DOCX format"""
    doc = Document()
    doc.add_paragraph(text)
    bio = io.BytesIO()
    doc.save(bio)
    return bio.getvalue()

def generate_docx(text):
    """Convert text to DOCX format"""
    doc = Document()
    doc.add_paragraph(text)
    bio = io.BytesIO()
    doc.save(bio)
    return bio.getvalue()

def cover_letter_generator(resume_data):
    st.header("📝 AI Cover Letter Generator (Powered by Gemini 2.0 Flash)")
    
    # Configuration - Consider moving to environment variables
    GEMINI_API_KEY = "AIzaSyBU9Ko2Hvu2LsCbaUuhdWmPK9SiITEIljE"
    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"
    
    with st.expander("⚙️ Customization Options"):
        col1, col2 = st.columns(2)
        with col1:
            tone = st.selectbox("Tone Style:",
                                ["Professional", "Enthusiastic", "Creative", "Concise"])
            length = st.select_slider("Length (words):", [150, 200, 250, 300, 400], value=250)
        with col2:
            highlight = st.multiselect("Highlight Skills:",
                                       resume_data.get('skills', []),
                                       default=resume_data.get('skills', [])[:3])
            avoid_buzzwords = st.checkbox("Avoid Corporate Buzzwords", True)
    
    job_desc = st.text_area("Paste Job Description:",
                            height=200,
                            placeholder="Paste the job posting here...")
    
    if st.button("✨ Generate Cover Letter"):
        if not job_desc.strip():
            st.error("Please paste a job description")
            return
        
        with st.spinner("Crafting your perfect cover letter..."):
            try:
                # Build the prompt
                prompt = f"""Create a {tone.lower()} cover letter with these requirements:
                - Target job: {job_desc[:1000]}
                - Highlight skills: {', '.join(highlight)}
                - Length: {length} words
                - {"Avoid buzzwords" if avoid_buzzwords else ""}
                - Candidate info: {str(resume_data)[:500]}"""
                
                # Prepare the request
                headers = {'Content-Type': 'application/json'}
                payload = {
                    "contents": [{
                        "parts": [{"text": prompt}]
                    }],
                    "generationConfig": {
                        "temperature": 0.7,
                        "maxOutputTokens": 2000
                    }
                }
                
                # Make the API call
                response = requests.post(api_url, headers=headers, data=json.dumps(payload))
                response.raise_for_status()
                
                # Process response
                result = response.json()
                if 'candidates' not in result or not result['candidates']:
                    raise ValueError("Invalid API response format")
                
                letter = result['candidates'][0]['content']['parts'][0]['text']
                
                # Post-processing
                letter = letter.replace("[Your Name]", resume_data.get('name', "Your Name"))
                letter = letter.replace("[Your Email]", resume_data.get('email', "your.email@example.com"))
                
                # Display editor
                st.subheader("Generated Cover Letter")
                editable_letter = st.text_area("Edit your letter:", value=letter, height=400)
                
                # Download options
                st.download_button("📥 Download as DOCX",
                                   generate_docx(editable_letter),
                                   file_name="cover_letter.docx")
                
                st.download_button("📥 Download as TXT",
                                   editable_letter,
                                   file_name="cover_letter.txt")
            
            except requests.exceptions.HTTPError as e:
                error_data = e.response.json()
                error_msg = error_data.get('error', {}).get('message', str(e))
                st.error(f"API Error: {error_msg}")
            except Exception as e:
                st.error(f"Error generating cover letter: {str(e)}")
def get_table_download_link(df, filename, text):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'


def pdf_reader(path):
    resource_manager = PDFResourceManager()
    fake_handle = io.StringIO()
    converter = TextConverter(resource_manager, fake_handle, laparams=LAParams())
    interpreter = PDFPageInterpreter(resource_manager, converter)
    with open(path, 'rb') as fh:
        for page in PDFPage.get_pages(fh, caching=True, check_extractable=True):
            interpreter.process_page(page)
    text = fake_handle.getvalue()
    converter.close()
    fake_handle.close()
    return text


def show_pdf(path):
    with open(path, 'rb') as f:
        b64 = base64.b64encode(f.read()).decode('utf-8')
    iframe = f'<iframe src="data:application/pdf;base64,{b64}" width="700" height="1000"></iframe>'
    st.markdown(iframe, unsafe_allow_html=True)


def course_recommender(course_list):
    st.subheader("**Courses & Certificates Recommendations 🎓**")
    no_of_reco = st.slider('Number of recommendations:', 1, 10, 5)
    random.shuffle(course_list)
    rec = []
    for idx, (name, link) in enumerate(course_list, start=1):
        st.markdown(f"({idx}) [{name}]({link})")
        rec.append(name)
        if idx >= no_of_reco:
            break
    return rec


def grammar_and_spelling_suggestions(text, max_issues=20):
    matches = tool.check(text)
    suggestions = []
    for m in matches[:max_issues]:
        suggestions.append({
            'offset': m.offset,
            'length': m.errorLength,
            'message': m.message,
            'corrections': m.replacements,
            'context': m.context,
        })
    return suggestions


def compute_readability(text):
    return {
        'Flesch Reading Ease': textstat.flesch_reading_ease(text),
        'Flesch–Kincaid Grade': textstat.flesch_kincaid_grade(text),
        'Gunning Fog Index': textstat.gunning_fog(text),
        'SMOG Index': textstat.smog_index(text),
        'Coleman–Liau Index': textstat.coleman_liau_index(text),
        'ARI': textstat.automated_readability_index(text),
    }


def highlight_issues(text, suggestions):
    html = escape(text)
    for s in sorted(suggestions, key=lambda x: -x['offset']):
        start, end = s['offset'], s['offset'] + s['length']
        before, issue, after = html[:start], html[start:end], html[end:]
        html = f"{before}<span style='background-color:#ffcccc'>{issue}</span>{after}"
    return html

# ----------------- Database Setup -----------------

db_pass = os.getenv('DB_PASS', 'admin')
connection = pymysql.connect(host='localhost', user='root', password=db_pass, db='cv')
cursor = connection.cursor()

# Ensure table exists
table_sql = (
    "CREATE TABLE IF NOT EXISTS user_data ("
    "ID INT AUTO_INCREMENT PRIMARY KEY,"
    "Name VARCHAR(255), Email_ID VARCHAR(255), resume_score VARCHAR(8),"
    "Timestamp VARCHAR(50), Page_no VARCHAR(5), Predicted_Field TEXT,"
    "User_level TEXT, Actual_skills TEXT, Recommended_skills TEXT, Recommended_courses TEXT"
    ");"
)
cursor.execute("CREATE DATABASE IF NOT EXISTS CV;")
cursor.execute("USE CV;")
cursor.execute(table_sql)

# ----------------- Main App -----------------

st.set_page_config(page_title="AI Resume Analyzer", page_icon='./Logo/logo2.png')


def run():
    # Header
    img = Image.open('./Logo/logo2.png')
    st.image(img)
    st.title("AI Resume Analyzer")
    
    # Sidebar choice
    mode = st.sidebar.selectbox("Mode", ["User", "Admin"])
    
    
    if mode == 'User':
        st.markdown("<h5 style='color:#053fff;'>Upload your resume for analysis 👔</h5>", unsafe_allow_html=True)
        pdf_file = st.file_uploader("Upload PDF Resume", type=["pdf"] )
        
        if pdf_file:
            with st.spinner("Processing..."):
                time.sleep(1)
            save_path = os.path.join('Uploaded_Resumes', pdf_file.name)
            with open(save_path, 'wb') as f:
                f.write(pdf_file.getbuffer())
            
            show_pdf(save_path)
            resume_data = ResumeParser(save_path).get_extracted_data()
            
            if resume_data:
                # 1) Text extraction
                raw = pdf_reader(save_path)
                if not raw:
                    st.error("❗ Could not extract text from PDF.")
                    st.stop()
                resume_text = raw.lower()
                
                # 2) Grammar & Readability
                st.subheader("📘 Grammar & Readability Suggestions")
                st.json(compute_readability(raw))
                st.markdown("**🔍 Grammar Suggestions:**")
                suggestions = grammar_and_spelling_suggestions(raw)
                if suggestions:
                    st.markdown(highlight_issues(raw, suggestions), unsafe_allow_html=True)
                else:
                    st.success("✅ No grammar issues found!")
                
                # 3) Basic Info & Level
                st.header("**Resume Analysis**")
                st.success(f"Hello {resume_data.get('name','Candidate')}!")
                st.text(f"Email: {resume_data.get('email','N/A')}")
                st.text(f"Contact: {resume_data.get('mobile_number','N/A')}")
                pages = resume_data.get('no_of_pages', 1)
                if pages == 1:
                    level, color = 'Fresher', '#d73b5c'
                elif pages == 2:
                    level, color = 'Intermediate', '#1ed760'
                else:
                    level, color = 'Experienced', '#fba171'
                st.markdown(f"<h4 style='color:{color};'>You are at {level} level</h4>", unsafe_allow_html=True)
                
                # 4) Skill Tags & Recommendations
                skills = resume_data.get('skills', [])
                
                st_tags(label='### Your Skills', text='Edit if needed', value=skills, key='skills')
                skill_match_section(st.session_state['skills'])
                # determine field
                ds_kw             = ['tensorflow','keras','pytorch','flask','streamlit']
                ml_kw             = ['machine learning','ml','scikit-learn','xgboost','lightgbm']
                web_kw            = ['react','django','node js','javascript','vue','angular']
                fullstack_kw      = ['react','node js','django','flask','vue','angular']
                android_kw        = ['android','flutter','kotlin']
                ios_kw            = ['ios','swift','xcode']
                uiux_kw           = ['ux','ui','figma','adobe xd','sketch']
                devops_kw         = ['docker','kubernetes','jenkins','ansible','terraform','ci/cd']
                qa_kw             = ['selenium','pytest','cucumber','jmeter','postman','robot framework']
                dataeng_kw        = ['spark','hadoop','airflow','etl','kafka']
                cloud_kw          = ['aws','azure','gcp','lambda','ec2','cloud']
                biz_analytics_kw  = ['excel','power bi','tableau','qlik','power query','sap bi']
                cybersec_kw       = ['cybersecurity','penetration testing','nmap','wireshark','kali','metasploit']
                blockchain_kw     = ['blockchain','solidity','ethereum','smart contract','web3']
                game_dev_kw       = ['unity','unreal','game engine','cocos2d','godot']
                embedded_kw       = ['embedded','rtos','microcontroller','arduino','raspberry pi']
                network_kw        = ['cisco','ccna','network','router','switch','tcp/ip']
                pm_kw             = ['product management','roadmap','jira','confluence','agile','scrum']
                
                reco_field, rec_skills, rec_course = '', [], []
                for s in skills:
                    sl = s.lower()
                    if sl in ml_kw:
                        reco_field = 'Machine Learning Engineering'
                        rec_skills = ['Scikit‑learn','TensorFlow','XGBoost']
                        rec_course = ml_course
                        break
                    if sl in ds_kw:
                        reco_field = 'Data Science'
                        rec_skills = ['Pandas','NumPy','Scikit‑learn']
                        rec_course = ds_course
                        break
                    if sl in fullstack_kw:
                        reco_field = 'Full‑Stack Development'
                        rec_skills = ['React','Node.js','Django']
                        rec_course = fullstack_course
                        break
                    if sl in web_kw:
                        reco_field = 'Web Development'
                        rec_skills = ['React','Node.js']
                        rec_course = web_course
                        break
                    if sl in android_kw:
                        reco_field = 'Android Development'
                        rec_skills = ['Kotlin','Jetpack']
                        rec_course = android_course
                        break
                    if sl in ios_kw:
                        reco_field = 'iOS Development'
                        rec_skills = ['SwiftUI','Combine']
                        rec_course = ios_course
                        break
                    if sl in uiux_kw:
                        reco_field = 'UI/UX Design'
                        rec_skills = ['Figma','Prototyping']
                        rec_course = uiux_course
                        break
                    if sl in devops_kw:
                        reco_field = 'DevOps Engineering'
                        rec_skills = ['Docker','Kubernetes','CI/CD']
                        rec_course = devops_course
                        break
                    if sl in qa_kw:
                        reco_field = 'QA Automation'
                        rec_skills = ['Selenium','pytest','Postman']
                        rec_course = qa_course
                        break
                    if sl in dataeng_kw:
                        reco_field = 'Data Engineering'
                        rec_skills = ['Apache Spark','Airflow','ETL']
                        rec_course = dataeng_course
                        break
                    if sl in cloud_kw:
                        reco_field = 'Cloud Engineering'
                        rec_skills = ['AWS EC2','Azure Functions','GCP Compute']
                        rec_course = cloud_course
                        break
                    if sl in biz_analytics_kw:
                        reco_field = 'Business Analytics'
                        rec_skills = ['Tableau','Power BI','Excel']
                        rec_course = biz_analytics_course
                        break
                    if sl in cybersec_kw:
                        reco_field = 'Cybersecurity'
                        rec_skills = ['Nmap','Wireshark','Pen Testing']
                        rec_course = cybersec_course
                        break
                    if sl in blockchain_kw:
                        reco_field = 'Blockchain Development'
                        rec_skills = ['Solidity','Ethereum','Smart Contracts']
                        rec_course = blockchain_course
                        break
                    if sl in game_dev_kw:
                        reco_field = 'Game Development'
                        rec_skills = ['Unity','Unreal Engine','C#']
                        rec_course = game_dev_course
                        break
                    if sl in embedded_kw:
                        reco_field = 'Embedded Systems'
                        rec_skills = ['C','RTOS','Microcontrollers']
                        rec_course = embedded_course
                        break
                    if sl in network_kw:
                        reco_field = 'Network Engineering'
                        rec_skills = ['Cisco IOS','TCP/IP','Routing & Switching']
                        rec_course = network_course
                        break
                    if sl in pm_kw:
                        reco_field = 'Product Management'
                        rec_skills = ['Roadmapping','Scrum','JIRA']
                        rec_course = pm_course
                        break
                
                if reco_field:
                    st.success(f"📌 Detected field: {reco_field}")
                    st_tags(label='Recommended Skills', text='Add these', value=rec_skills, key='reco')
                    rec_course = course_recommender(rec_course)
                
                # 5) Advanced Resume Tips & Strength Meter
                st.subheader("🎯 Advanced Resume Tips & Suggestions")
                sections = {
                    "Objective": {"keywords": ["objective"], "score":15},
                    "Achievements": {"keywords": ["achievements"], "score":15},
                    "Projects": {"keywords": ["projects"], "score":15},
                    "Skills": {"keywords": ["skills"], "score":10},
                    "Education": {"keywords": ["education"], "score":10},
                    "Experience": {"keywords": ["experience"], "score":10},
                    "Certifications": {"keywords": ["certifications"], "score":10},
                    "Contact Info": {"keywords": ["email","phone"], "score":5},
                }
                total, max_s = 0, sum(v['score'] for v in sections.values())
                for name, info in sections.items():
                    found = any(k in resume_text for k in info['keywords'])
                    if found:
                        total += info['score']
                        st.markdown(f"<span style='color:#1ed760;'>✅ {name}</span>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<span style='color:#ff4b4b;'>⚠️ {name}</span>", unsafe_allow_html=True)
                pct = int((total/max_s)*100)
                st.subheader("📊 Resume Strength Meter")
                st.markdown("""
                    <style>
                        .stProgress > div > div > div > div {
                            background: linear-gradient(to right, #d73b5c, #f39c12, #27ae60);
                        }
                    </style>""", unsafe_allow_html=True)
                bar = st.progress(0)
                for i in range(pct+1): time.sleep(0.02); bar.progress(i)
                if pct>=85: st.success(f"🌟 {pct}/100")
                elif pct>=60: st.warning(f"✅ {pct}/100")
                else: st.error(f"🚧 {pct}/100")
                st.info("Based on section presence only.")
                st.markdown("---")
                cover_letter_generator(resume_data)
                
                
                # 6) Persist data
                timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
                cursor.execute(
                    "INSERT INTO user_data "
                    "(Name,Email_ID,resume_score,Timestamp,Page_no,Predicted_Field,User_level,Actual_skills,Recommended_skills,"
                    "Recommended_courses) "
                    "VALUES ("+
                    "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)" ,
                    (
                        resume_data.get('name',''), resume_data.get('email',''), str(pct),
                        timestamp, str(pages), reco_field, level,
                        str(skills), str(rec_skills), str(rec_course)
                    )
                )
                connection.commit()

    else:
        st.subheader("👨‍💼 Admin Dashboard")
        
        # Input fields
        user = st.text_input("Username")
        pwd = st.text_input("Password", type='password')
        
        # Add button click state tracker
        if 'login_clicked' not in st.session_state:
            st.session_state['login_clicked'] = False
        
        # Check button click
        if st.button('Login'):
            st.session_state['login_clicked'] = True
        
        # Only validate after button is clicked
        if st.session_state['login_clicked']:
            if user == 'admin' and pwd == 'admin':
                df = pd.read_sql("SELECT * FROM user_data;", connection)
                
                # Decode bytes columns
                df = df.applymap(lambda x: x.decode('utf-8', errors='ignore')
                if isinstance(x, (bytes, bytearray)) else x)
                
                st.subheader("📊 User Data")
                st.dataframe(df)
                st.markdown(get_table_download_link(df, 'report.csv', 'Download CSV'),
                            unsafe_allow_html=True)
                
                # Basic Metrics
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Users", len(df))
                
                # Most Common Field with fallback
                most_common_field = 'N/A'
                if 'Predicted_Field' in df.columns and not df['Predicted_Field'].empty:
                    most_common_field = df['Predicted_Field'].mode()[0] if not df['Predicted_Field'].mode().empty else 'N/A'
                col2.metric("Most Common Field", most_common_field)
                
                # User Level Metrics with numerical conversion
                level_mapping = {'Fresher': 1, 'Intermediate': 2, 'Experienced': 3}
                avg_level = 'N/A'
                common_level = 'N/A'
                
                if 'User_level' in df.columns:
                    try:
                        df['Level_Score'] = df['User_level'].map(level_mapping)
                        avg_level = f"{df['Level_Score'].mean():.1f}/3" if not df['Level_Score'].empty else 'N/A'
                        common_level = df['User_level'].mode()[0] if not df['User_level'].mode().empty else 'N/A'
                    except Exception as e:
                        st.error(f"Error processing levels: {str(e)}")
                
                col3.metric("Avg User Level", avg_level, f"Most common: {common_level}")
                
                # Visualization Section
                st.subheader("📈 Data Visualizations")
                
                # Pie Charts with existence checks
                if 'Predicted_Field' in df.columns:
                    c1, c2 = st.columns(2)
                    with c1:
                        st.plotly_chart(px.pie(df, names='Predicted_Field',
                                               title='Field Distribution'))
                    with c2:
                        if 'User_level' in df.columns:
                            st.plotly_chart(px.pie(df, names='User_level',
                                                   title='User Levels'))
                
                # Enhanced Bar Chart with safety checks
                if 'Predicted_Field' in df.columns and 'User_level' in df.columns:
                    st.plotly_chart(px.bar(df, x='Predicted_Field',
                                           color='User_level',
                                           title='Field Distribution by User Level',
                                           labels={'User_level': 'Experience Level'}))
                
                # Time Series with proper datetime handling
                if 'Timestamp' in df.columns:
                    try:
                        df['Timestamp'] = df['Timestamp'].str.replace('_', ' ').pipe(pd.to_datetime)
                        time_series = df.set_index('Timestamp').resample('D').size()
                        st.plotly_chart(px.line(time_series,
                                                title='Daily User Registrations',
                                                labels={'value': 'Registrations'}))
                    except Exception as e:
                        st.error(f"Error processing timestamps: {str(e)}")
                
                # Correlation Heatmap with numerical checks
                numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
                numerical_cols = list(dict.fromkeys(numerical_cols))
                
                if len(numerical_cols) > 1:
                    st.plotly_chart(px.imshow(df[numerical_cols].corr(),
                                              title='Feature Correlation Heatmap',
                                              labels=dict(x="Features", y="Features", color="Correlation")))
                else:
                    st.warning("Insufficient numerical data for correlation heatmap")
            else:
                st.error("Invalid credentials")
   
if __name__ == '__main__':
    run()
