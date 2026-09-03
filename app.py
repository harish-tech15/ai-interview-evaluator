
import re
import streamlit as st

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="AI Interview Evaluator",
    page_icon="🤖",
    layout="wide"
)


# ==================================================
# LOAD AI MODELS
# ==================================================

@st.cache_resource
def load_models():

    # Embedding model
    embedding_model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    # FLAN-T5 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "google/flan-t5-base"
    )

    # FLAN-T5 model
    text_model = AutoModelForSeq2SeqLM.from_pretrained(
        "google/flan-t5-base"
    )

    return embedding_model, tokenizer, text_model


# Load models
model, tokenizer, text_model = load_models()


# ==================================================
# TEXT GENERATION
# ==================================================

def generate_text(prompt, max_new_tokens=250):

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    outputs = text_model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False
    )

    generated_text = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    return generated_text


# ==================================================
# RESUME TEXT EXTRACTION
# ==================================================

def extract_resume_text(pdf_file):

    reader = PdfReader(pdf_file)

    text = ""

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text


# ==================================================
# RESUME MATCH SCORE
# ==================================================

def calculate_match_score(
    resume_text,
    job_description
):

    resume_embedding = model.encode(
        [resume_text]
    )

    job_embedding = model.encode(
        [job_description]
    )

    similarity = cosine_similarity(
        resume_embedding,
        job_embedding
    )[0][0]

    # Keep score between 0 and 100
    similarity = max(
        0,
        min(similarity, 1)
    )

    return round(
        similarity * 100,
        2
    )


# ==================================================
# SKILL GAP ANALYSIS
# ==================================================

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
        "Docker",
        "Pandas",
        "NumPy",
        "Scikit-learn",
        "Flask",
        "Django"

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


# ==================================================
# GENERATE INTERVIEW QUESTIONS
# ==================================================

def generate_questions(
    job_description,
    skills
):

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
- Cover Python, Machine Learning, NLP,
  Generative AI, SQL and RAG when relevant
- Suitable for a fresher
- Return only questions
- One question per line
"""

    generated = generate_text(
        prompt,
        max_new_tokens=250
    )

    return clean_questions(
        generated
    )


# ==================================================
# CLEAN QUESTIONS
# ==================================================

def clean_questions(text):

    lines = text.split("\n")

    questions = []

    for line in lines:

        line = line.strip()

        if not line:
            continue

        # Remove numbering
        line = re.sub(
            r"^\d+[\.\)\-:]\s*",
            "",
            line
        )

        # Remove bullet points
        line = re.sub(
            r"^[\-\*\•]\s*",
            "",
            line
        )

        if len(line) > 15:

            questions.append(line)

    return questions[:5]


# ==================================================
# GENERATE REFERENCE ANSWER
# ==================================================

def generate_reference_answer(question):

    prompt = f"""
You are an expert AI Engineer and technical interviewer.

Interview Question:

{question}

Create a technically correct ideal answer
for a Junior AI Engineer fresher.

Requirements:

- Use simple English
- Include important technical concepts
- Keep the answer concise
- Make it suitable for an interview
"""

    return generate_text(
        prompt,
        max_new_tokens=200
    )


# ==================================================
# KEYWORD EXTRACTION
# ==================================================

def extract_keywords(text):

    words = re.findall(
        r"\b[a-zA-Z][a-zA-Z+#.-]{2,}\b",
        text.lower()
    )

    stop_words = {

        "the",
        "and",
        "for",
        "with",
        "that",
        "this",
        "from",
        "are",
        "was",
        "were",
        "what",
        "how",
        "you",
        "can",
        "into",
        "use",
        "using",
        "your",
        "their",
        "they",
        "have",
        "has",
        "will",
        "about",
        "which",
        "when",
        "where",
        "should"

    }

    keywords = {

        word

        for word in words

        if word not in stop_words

    }

    return keywords


# ==================================================
# ANSWER EVALUATION
# ==================================================

def evaluate_answer(
    candidate_answer,
    reference_answer
):

    # Semantic embeddings

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

    # Prevent negative score
    semantic_similarity = max(
        0,
        min(
            semantic_similarity,
            1
        )
    )

    # Keywords

    reference_keywords = extract_keywords(
        reference_answer
    )

    candidate_keywords = extract_keywords(
        candidate_answer
    )

    matched_keywords = (
        reference_keywords
        &
        candidate_keywords
    )

    if reference_keywords:

        keyword_coverage = (
            len(matched_keywords)
            /
            len(reference_keywords)
        )

    else:

        keyword_coverage = 0

    # Final score

    score = (

        semantic_similarity * 70

        +

        keyword_coverage * 30

    )

    return round(
        score,
        2
    )


# ==================================================
# AI FEEDBACK
# ==================================================

def generate_feedback(
    question,
    candidate_answer,
    reference_answer,
    score
):

    prompt = f"""
You are an expert technical interviewer.

Interview Question:

{question}

Candidate Answer:

{candidate_answer}

Ideal Answer:

{reference_answer}

Candidate Score:

{score}/100

Analyze the candidate answer.

Provide:

Strengths:
Mention what the candidate explained correctly.

Weaknesses:
Mention important concepts that are missing.

Improvement:
Explain how the candidate can improve.

Interview Tip:
Give one practical interview tip.

Use simple English.
Keep the response concise.
"""

    return generate_text(
        prompt,
        max_new_tokens=300
    )


# ==================================================
# IMPROVED ANSWER
# ==================================================

def generate_improved_answer(
    question,
    candidate_answer
):

    prompt = f"""
You are an expert AI Engineer and interview coach.

Interview Question:

{question}

Candidate Answer:

{candidate_answer}

Rewrite the candidate answer into
a technically correct interview-quality answer.

Requirements:

- Use simple English
- Include important technical concepts
- Suitable for a fresher
- Keep it concise
- Make it easy to speak in an interview
- Return only the improved answer
"""

    return generate_text(
        prompt,
        max_new_tokens=250
    )


# ==================================================
# APPLICATION UI
# ==================================================

st.title(
    "🤖 AI Interview Evaluator"
)

st.write(
    "Resume Analysis + Skill Gap Analysis + "
    "AI Interview + Answer Evaluation"
)

st.divider()


# ==================================================
# SIDEBAR
# ==================================================

with st.sidebar:

    st.header(
        "📄 Resume"
    )

    resume_file = st.file_uploader(
        "Upload Resume PDF",
        type=["pdf"]
    )

    st.header(
        "💼 Job Description"
    )

    job_description = st.text_area(
        "Paste Job Description",
        height=250,
        placeholder=(
            "Paste the complete job description here..."
        )
    )


# ==================================================
# ANALYZE RESUME
# ==================================================

if resume_file and job_description:

    if st.button(
        "🚀 Analyze Resume",
        use_container_width=True
    ):

        with st.spinner(
            "Analyzing resume and generating interview questions..."
        ):

            # Extract resume

            resume_text = extract_resume_text(
                resume_file
            )

            if not resume_text.strip():

                st.error(
                    "Could not extract text from the PDF."
                )

                st.stop()

            # Match score

            match_score = calculate_match_score(
                resume_text,
                job_description
            )

            # Skills

            found_skills, missing_skills = (
                find_skills(
                    resume_text
                )
            )

            # Questions

            questions = generate_questions(
                job_description,
                found_skills
            )

            # Save results

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
            ] = questions

            st.success(
                "Resume analysis completed!"
            )


# ==================================================
# RESUME RESULTS
# ==================================================

if "match_score" in st.session_state:

    st.header(
        "📊 Resume Analysis"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Resume Match",
            f"{st.session_state['match_score']}%"
        )

    with col2:

        st.metric(
            "Skills Found",
            len(
                st.session_state[
                    "found_skills"
                ]
            )
        )

    with col3:

        st.metric(
            "Skills Missing",
            len(
                st.session_state[
                    "missing_skills"
                ]
            )
        )

    st.subheader(
        "✅ Skills Found"
    )

    found = st.session_state[
        "found_skills"
    ]

    if found:

        st.write(
            " • ".join(found)
        )

    else:

        st.info(
            "No matching skills found."
        )

    st.subheader(
        "❌ Missing Skills"
    )

    missing = st.session_state[
        "missing_skills"
    ]

    if missing:

        st.write(
            " • ".join(missing)
        )

    else:

        st.success(
            "No major missing skills detected."
        )

    st.divider()


# ==================================================
# AI INTERVIEW
# ==================================================

if "questions" in st.session_state:

    st.header(
        "🎤 AI Technical Interview"
    )

    questions = st.session_state[
        "questions"
    ]

    if not questions:

        st.warning(
            "No questions were generated. "
            "Try analyzing the resume again."
        )

    for i, question in enumerate(
        questions
    ):

        st.subheader(
            f"Question {i + 1}"
        )

        st.write(
            question
        )

        answer = st.text_area(
            "Your Answer",
            key=f"answer_{i}",
            height=130,
            placeholder=(
                "Type your interview answer here..."
            )
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
                    "AI is evaluating your answer..."
                ):

                    # Reference answer

                    reference = (
                        generate_reference_answer(
                            question
                        )
                    )

                    # Score

                    score = evaluate_answer(
                        answer,
                        reference
                    )

                    # Feedback

                    feedback = (
                        generate_feedback(
                            question,
                            answer,
                            reference,
                            score
                        )
                    )

                    # Improved answer

                    improved = (
                        generate_improved_answer(
                            question,
                            answer
                        )
                    )

                st.success(
                    "Answer evaluated successfully!"
                )

                # Score

                st.metric(
                    "🎯 Answer Score",
                    f"{score}/100"
                )

                # Feedback

                st.subheader(
                    "💡 AI Feedback"
                )

                st.write(
                    feedback
                )

                # Improved answer

                st.subheader(
                    "✨ Improved Answer"
                )

                st.info(
                    improved
                )

                # Reference

                with st.expander(
                    "📚 View Reference Answer"
                ):

                    st.write(
                        reference
                    )

                st.divider()


# ==================================================
# FOOTER
# ==================================================

st.divider()

st.caption(
    "AI Interview Evaluator | "
    "Built with Python, Hugging Face, "
    "Sentence Transformers & Streamlit"
)

