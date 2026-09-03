# 🤖 AI Interview Evaluator

An AI-powered interview preparation and evaluation application built with **Python, Hugging Face Transformers, Sentence Transformers, and Streamlit**.

The application analyzes a candidate's resume against a job description, identifies skill gaps, generates technical interview questions, evaluates candidate answers, and provides AI-powered feedback with improved answers.

---

## 🚀 Project Overview

The **AI Interview Evaluator** simulates an AI-powered technical interview.

### Workflow

```text
Resume PDF
    ↓
Resume Text Extraction
    ↓
Job Description
    ↓
Resume Match Score
    ↓
Skill Gap Analysis
    ↓
AI Interview Questions
    ↓
Candidate Answers
    ↓
Reference Answers
    ↓
Semantic Similarity + Keyword Coverage
    ↓
Answer Score
    ↓
AI Feedback
    ↓
Improved Interview Answer
```

---

## ✨ Features

### 📄 Resume Analysis

* Upload a Resume PDF
* Extract resume text automatically
* Compare resume with a Job Description
* Generate a Resume Match Score

### 🧠 Skill Gap Analysis

Identifies:

* Skills already present in the resume
* Missing skills required for the role

Example:

```text
Skills Found:
✓ Python
✓ SQL
✓ Machine Learning
✓ Streamlit

Missing Skills:
✗ RAG
✗ Vector Database
✗ Docker
```

### 🎤 AI Interview Question Generator

Generates technical interview questions based on:

* Job Description
* Candidate skills
* AI/ML concepts
* Python
* NLP
* Generative AI
* RAG
* SQL

### 📝 Answer Evaluation

The application evaluates candidate answers using:

**Semantic Similarity**

and

**Keyword Coverage**

Final score:

```text
Answer Score =
Semantic Similarity × 70
+
Keyword Coverage × 30
```

### 💡 AI Feedback

The application provides:

* Strengths
* Weaknesses
* Improvement suggestions
* Interview tips

### ✨ Improved Answer

The candidate's answer is rewritten into a more technically correct and interview-ready response.

---

## 🛠️ Tech Stack

| Technology                | Purpose                                        |
| ------------------------- | ---------------------------------------------- |
| Python                    | Application development                        |
| Streamlit                 | Web application                                |
| PyPDF                     | Resume PDF text extraction                     |
| Hugging Face Transformers | AI text generation                             |
| FLAN-T5                   | Interview question/reference answer generation |
| Sentence Transformers     | Semantic embeddings                            |
| all-MiniLM-L6-v2          | Text similarity                                |
| Scikit-learn              | Cosine similarity                              |
| Pandas                    | Data processing                                |

---

## 🧠 AI Models

### Text Generation

```text
google/flan-t5-base
```

Used for:

* Interview question generation
* Reference answer generation
* Candidate feedback
* Improved answers

### Embedding Model

```text
sentence-transformers/all-MiniLM-L6-v2
```

Used for:

* Resume/JD similarity
* Candidate/reference answer similarity

---

## 📂 Project Structure

```text
AI-Interview-Evaluator/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
└── sample/
    └── sample_job_description.txt
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/ai-interview-evaluator.git
```

### 2. Open the project

```bash
cd ai-interview-evaluator
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

### 4. Activate the environment

### Windows

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

Start Streamlit:

```bash
streamlit run app.py
```

The application will open in your browser.

Default address:

```text
http://localhost:8501
```

---

## 📋 How to Use

### Step 1 — Upload Resume

Upload your resume in PDF format.

### Step 2 — Enter Job Description

Paste the target job description into the Job Description field.

### Step 3 — Analyze Resume

Click:

```text
🚀 Analyze Resume
```

The application calculates:

* Resume Match %
* Skills Found
* Missing Skills

### Step 4 — Start Interview

The AI generates interview questions based on the job description and candidate skills.

### Step 5 — Answer Questions

Enter your answer below each question.

### Step 6 — Evaluate Answer

Click:

```text
Evaluate Answer
```

The system calculates an answer score.

### Step 7 — Review Feedback

The AI provides:

```text
Strengths
Weaknesses
Improvement
Interview Tip
```

### Step 8 — Learn the Improved Answer

The application generates an improved interview-quality answer.

---

## 📊 Scoring Method

The project combines two approaches.

### 1. Semantic Similarity

Candidate and reference answers are converted into vector embeddings using:

```text
all-MiniLM-L6-v2
```

Cosine similarity measures how semantically similar the answers are.

### 2. Keyword Coverage

Important words from the reference answer are compared with the candidate answer.

### Final Score

```text
Final Score =
(Semantic Similarity × 70)
+
(Keyword Coverage × 30)
```

The final result is displayed as:

```text
85.5 / 100
```

---

## 🎯 Example

### Interview Question

```text
What is RAG in Generative AI?
```

### Candidate Answer

```text
RAG combines a language model with external documents.
It retrieves relevant information and provides it to the model
before generating an answer.
```

### AI Evaluation

```text
Answer Score: 88/100
```

### Feedback

```text
Strengths:
Good explanation of retrieval and external knowledge.

Weaknesses:
Could explain the generation step in more detail.

Improvement:
Mention that retrieved documents are added as context
to the LLM prompt.

Interview Tip:
Give a simple real-world example when explaining RAG.
```

---

## ☁️ Deploy on Streamlit Cloud

The application can be deployed using Streamlit Cloud without an API key.

### Step 1

Push the project to GitHub.

### Step 2

Create a new Streamlit application.

Select:

```text
Repository: ai-interview-evaluator
Branch: main
Main file: app.py
```

### Step 3

Deploy the application.

No Groq API key is required because the application uses Hugging Face models directly.

---

## 🔐 Security

Do not upload sensitive information to GitHub.

Never commit:

```text
.env
API keys
Passwords
Private resumes
Candidate answers
Personal documents
```

The `.gitignore` file prevents common temporary and sensitive files from being committed.

---

## 🚀 Future Enhancements

Possible improvements include:

* 🎙️ Voice-based interview
* 🗣️ Speech-to-text answer evaluation
* 📈 Interview performance dashboard
* 📊 Question-wise score charts
* 📥 Download interview report as PDF
* 💾 Save interview history
* 🎯 Personalized learning recommendations
* 🧩 Role-specific question banks
* 🧠 Difficulty levels: Easy / Medium / Hard
* 👨‍💻 Coding question evaluation
* 🔍 Better resume skill extraction
* 📚 RAG-based interview question generation
* 🗃️ Vector database integration

---

## 💼 Resume Project Description

**AI Interview Evaluator | Python, NLP, Hugging Face, Sentence Transformers, Streamlit**

```text
Developed an AI-powered interview evaluation application that analyzes
resumes against job descriptions, identifies skill gaps, generates
role-specific technical interview questions, and evaluates candidate
answers using semantic similarity and keyword coverage. Implemented
Hugging Face FLAN-T5 for AI-generated questions, feedback and improved
answers, and Sentence Transformers for semantic text evaluation.
Deployed the application using Streamlit.
```

---

## 🎓 Key Skills Demonstrated

This project demonstrates practical knowledge of:

```text
Python
NLP
Generative AI
Large Language Models
Transformers
Sentence Transformers
Semantic Similarity
Cosine Similarity
Prompt Engineering
Resume Parsing
Machine Learning
Streamlit
Git & GitHub
AI Application Deployment
```

---

## 👨‍💻 Author

**Harish**

B.Tech – Artificial Intelligence and Data Science

Sri Venkateswaraa College of Technology

---

## ⭐ Project Goal

The goal of this project is to help freshers practice technical interviews using an AI-powered system that provides **objective scoring, personalized feedback, and improved answers**.

If you find this project useful, consider giving the repository a ⭐ on GitHub.
