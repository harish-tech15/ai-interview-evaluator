import re
import pandas as pd
import streamlit as st

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline


# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="AI Interview Evaluator",
    page_icon="🤖",
    layout="wide"
)


# --------------------------------------------------
# LOAD MODELS
# --------------------------------------------------

@st.cache_resource
def load_models():

    embedding_model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    question_generator = pipeline(
        "text2text-generation",
        model="google/flan-t5-base"
    )

    return embedding_model, question_generator


model, question_generator = load_models()


# --------------------------------------------------
# FUNCTIONS
# --------------------------------------------------

def extract_resume_text(pdf_file):

    reader = PdfReader(pdf_file)

    text = ""

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text


def calculate_match_score(resume_text, job_description):

    resume_embedding = model.encode([resume_text])
    job_embedding = model.encode([job_description])

    similarity = cosine_similarity(
        resume_embedding,
        job_embedding
    )[0][0]

    return round(similarity * 100, 2)


def find_skills(resume_text):

    skills = [
        "Python",
        "Machine Learning",
        "Deep Learning",
        "NLP",
        "Generative AI",
        "LLM",
        "RAG",
        "SQL",
        "Git",
        "Streamlit",
        "Prompt Engineering",
        "Vector Database",
        "Data Preprocessing",
        "TensorFlow",
        "PyTorch",
        "AWS",
        "Docker"
    ]

    resume_lower = resume_text.lower()

    found = []
    missing = []

    for skill in skills:

        if skill.lower() in resume_lower:
            found.append(skill)
        else:
            missing.append(skill)

    return found, missing


def generate_questions(job_description, skills):

    skills_text = ", ".join(skills)

    prompt = f"""
You are an expert technical interviewer.

Create 5 interview questions for a Junior AI Engineer.

Job Description:
{job_description}

Candidate Skills:
{skills_text}

Requirements:
- Include technical questions
- Include scenario-based questions
- Include project-based questions
- Focus on relevant AI and programming skills
- Suitable for a fresher
- Return only questions
- One question per line
"""

    result = question_generator(
        prompt,
        max_new_tokens=250,
        do_sample=False
    )

    return clean_questions(
        result[0]["generated_text"]
    )


def clean_questions(text):

    lines = text.split("\n")

    questions = []

    for line in lines:

        line = line.strip()

        line = re.sub(
            r"^\d+[\.\)]\s*",
            "",
            line
        )

        if len(line) > 15:
            questions.append(line)

    return questions[:5]


def generate_reference_answer(question):

    prompt = f"""
Give a technically correct interview answer
for a Junior AI Engineer fresher.

Question:
{question}

Keep the answer concise and clear.
"""

    result = question_generator(
        prompt,
        max_new_tokens=200,
        do_sample=False
    )

    return result[0]["generated_text"]


def extract_keywords(text):

    words = re.findall(
        r"\b[a-zA-Z][a-zA-Z+#.-]{2,}\b",
        text.lower()
    )

    stop_words = {
        "the", "and", "for", "with",
        "that", "this", "from", "are",
        "was", "what", "how", "you",
        "can", "into", "use"
    }

    return {
        word for word in words
        if word not in stop_words
    }


def evaluate_answer(
    candidate_answer,
    reference_answer
):

    candidate_embedding = model.encode(
        [candidate_answer]
    )

    reference_embedding = model.encode(
        [reference_answer]
    )

    semantic_similarity = cosine_similarity(
        candidate_embedding,
        reference_embedding
    )[0][0]

    reference_keywords = extract_keywords(
        reference_answer
    )

    candidate_keywords = extract_keywords(
        candidate_answer
    )

    matched_keywords = (
        reference_keywords &
        candidate_keywords
    )

    if reference_keywords:

        keyword_coverage = (
            len(matched_keywords) /
            len(reference_keywords)
        )

    else:

        keyword_coverage = 0

    score = (
        semantic_similarity * 70 +
        keyword_coverage * 30
    )

    return round(score, 2)


def generate_feedback(
    question,
    candidate_answer,
    reference_answer,
    score
):

    prompt = f"""
You are an expert technical interviewer.

Question:
{question}

Candidate Answer:
{candidate_answer}

Ideal Answer:
{reference_answer}

Score:
{score}/100

Give concise feedback with:

Strengths:
Weaknesses:
Improvement:
Interview Tip:

Use simple English.
"""

    result = question_generator(
        prompt,
        max_new_tokens=250,
        do_sample=False
    )

    return result[0]["generated_text"]


def generate_improved_answer(
    question,
    candidate_answer
):

    prompt = f"""
You are an interview coach.

Question:
{question}

Candidate Answer:
{candidate_answer}

Rewrite the answer into a technically
correct interview-quality answer.

Use simple English.
Suitable for a fresher.
Return only the improved answer.
"""

    result = question_generator(
        prompt,
        max_new_tokens=250,
        do_sample=False
    )

    return result[0]["generated_text"]


# --------------------------------------------------
# UI
# --------------------------------------------------

st.title("🤖 AI Interview Evaluator")

st.write(
    "Resume analysis + interview questions + "
    "AI answer evaluation"
)

st.divider()


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.header("📄 Resume")

    resume_file = st.file_uploader(
        "Upload Resume PDF",
        type=["pdf"]
    )

    st.header("💼 Job Description")

    job_description = st.text_area(
        "Paste Job Description",
        height=250
    )


# --------------------------------------------------
# RESUME ANALYSIS
# --------------------------------------------------

if resume_file and job_description:

    if st.button(
        "🚀 Analyze Resume",
        use_container_width=True
    ):

        with st.spinner(
            "Analyzing resume..."
        ):

            resume_text = extract_resume_text(
                resume_file
            )

            match_score = calculate_match_score(
                resume_text,
                job_description
            )

            found_skills, missing_skills = (
                find_skills(resume_text)
            )

            st.session_state[
                "resume_text"
            ] = resume_text

            st.session_state[
                "match_score"
            ] = match_score

            st.session_state[
                "found_skills"
            ] = found_skills

            st.session_state[
                "missing_skills"
            ] = missing_skills

            st.session_state[
                "questions"
            ] = generate_questions(
                job_description,
                found_skills
            )


# --------------------------------------------------
# DISPLAY RESUME RESULTS
# --------------------------------------------------

if "match_score" in st.session_state:

    st.header("📊 Resume Analysis")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Resume Match",
            f"{st.session_state['match_score']}%"
        )

    with col2:

        st.metric(
            "Skills Found",
            len(st.session_state["found_skills"])
        )

    with col3:

        st.metric(
            "Skills Missing",
            len(st.session_state["missing_skills"])
        )


    st.subheader("✅ Skills Found")

    st.write(
        ", ".join(
            st.session_state["found_skills"]
        )
    )


    st.subheader("❌ Missing Skills")

    st.write(
        ", ".join(
            st.session_state["missing_skills"]
        )
    )


    st.divider()


# --------------------------------------------------
# INTERVIEW
# --------------------------------------------------

if "questions" in st.session_state:

    st.header("🎤 AI Interview")

    questions = st.session_state["questions"]

    for i, question in enumerate(
        questions
    ):

        st.subheader(
            f"Question {i + 1}"
        )

        st.write(question)

        answer = st.text_area(
            "Your Answer",
            key=f"answer_{i}",
            height=120
        )

        if st.button(
            f"Evaluate Answer {i + 1}",
            key=f"evaluate_{i}"
        ):

            if not answer.strip():

                st.warning(
                    "Please enter your answer first."
                )

            else:

                with st.spinner(
                    "Evaluating your answer..."
                ):

                    reference = (
                        generate_reference_answer(
                            question
                        )
                    )

                    score = evaluate_answer(
                        answer,
                        reference
                    )

                    feedback = (
                        generate_feedback(
                            question,
                            answer,
                            reference,
                            score
                        )
                    )

                    improved = (
                        generate_improved_answer(
                            question,
                            answer
                        )
                    )


                st.metric(
                    "Answer Score",
                    f"{score}/100"
                )

                st.subheader(
                    "💡 AI Feedback"
                )

                st.write(feedback)

                st.subheader(
                    "✨ Improved Answer"
                )

                st.info(improved)

                with st.expander(
                    "View Reference Answer"
                ):

                    st.write(reference)

                st.divider()
