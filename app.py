import re
import streamlit as st

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="AI Interview Evaluator",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

    /* Main background */
    .stApp {
        background: #f7f9fc;
    }

    /* Remove top padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1250px;
    }

    /* Header */
    .hero {
        background: linear-gradient(135deg, #111827, #2563eb);
        padding: 35px;
        border-radius: 22px;
        color: white;
        margin-bottom: 30px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.10);
    }

    .hero h1 {
        font-size: 42px;
        margin-bottom: 8px;
        font-weight: 800;
    }

    .hero p {
        font-size: 17px;
        opacity: 0.9;
        margin-bottom: 0;
    }

    /* Cards */
    .card {
        background: white;
        padding: 25px;
        border-radius: 18px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 5px 18px rgba(0,0,0,0.05);
        margin-bottom: 20px;
    }

    .card-title {
        font-size: 21px;
        font-weight: 700;
        color: #111827;
        margin-bottom: 10px;
    }

    .card-subtitle {
        color: #6b7280;
        font-size: 14px;
        margin-bottom: 18px;
    }

    /* Metric cards */
    .metric-card {
        background: white;
        border-radius: 18px;
        padding: 22px;
        border: 1px solid #e5e7eb;
        text-align: center;
        box-shadow: 0 5px 18px rgba(0,0,0,0.05);
    }

    .metric-icon {
        font-size: 28px;
    }

    .metric-value {
        font-size: 32px;
        font-weight: 800;
        color: #111827;
        margin-top: 5px;
    }

    .metric-label {
        color: #6b7280;
        font-size: 14px;
    }

    /* Skill badges */
    .skill {
        display: inline-block;
        padding: 7px 12px;
        border-radius: 20px;
        margin: 4px;
        font-size: 13px;
        font-weight: 600;
    }

    .skill-found {
        background: #dcfce7;
        color: #166534;
    }

    .skill-missing {
        background: #fee2e2;
        color: #991b1b;
    }

    /* Question box */
    .question-box {
        background: #ffffff;
        padding: 22px;
        border-radius: 16px;
        border-left: 5px solid #2563eb;
        margin-bottom: 18px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
    }

    .question-number {
        color: #2563eb;
        font-size: 13px;
        font-weight: 700;
        text-transform: uppercase;
    }

    .question-text {
        color: #111827;
        font-size: 18px;
        font-weight: 650;
        margin-top: 8px;
    }

    /* Result */
    .result-good {
        background: #dcfce7;
        color: #166534;
        padding: 14px;
        border-radius: 12px;
        text-align: center;
        font-weight: 700;
    }

    .result-average {
        background: #fef3c7;
        color: #92400e;
        padding: 14px;
        border-radius: 12px;
        text-align: center;
        font-weight: 700;
    }

    .result-low {
        background: #fee2e2;
        color: #991b1b;
        padding: 14px;
        border-radius: 12px;
        text-align: center;
        font-weight: 700;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #111827;
    }

    section[data-testid="stSidebar"] * {
        color: white;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 10px;
        font-weight: 700;
        padding: 10px 20px;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #6b7280;
        padding-top: 35px;
        font-size: 13px;
    }

</style>
""", unsafe_allow_html=True)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown("""
    <div style="text-align:center; padding:15px 5px 25px 5px;">
        <div style="font-size:45px;">🤖</div>
        <h2>AI Interview</h2>
        <p style="color:#9ca3af;">Evaluator</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("### 📌 Navigation")

    st.markdown("""
    **🏠 Dashboard**

    **📄 Resume Analysis**

    **🎯 Skill Gap**

    **💬 AI Interview**

    **📊 Performance**
    """)

    st.markdown("---")

    st.markdown("### ⚙️ Technology")

    st.caption("Python")
    st.caption("Streamlit")
    st.caption("Sentence Transformers")
    st.caption("FLAN-T5")
    st.caption("Machine Learning")


# =========================================================
# HEADER
# =========================================================

st.markdown("""
<div class="hero">

    <h1>🤖 AI Interview Evaluator</h1>

    <p>
        Analyze your resume, identify skill gaps,
        practice AI-generated interview questions
        and receive intelligent feedback.
    </p>

</div>
""", unsafe_allow_html=True)


# =========================================================
# MODEL LOADING
# =========================================================

@st.cache_resource
def load_models():

    embedding_model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        "google/flan-t5-base"
    )

    text_model = AutoModelForSeq2SeqLM.from_pretrained(
        "google/flan-t5-base"
    )

    return embedding_model, tokenizer, text_model


with st.spinner("Loading AI models..."):

    model, tokenizer, text_model = load_models()


# =========================================================
# TEXT GENERATION
# =========================================================

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

    return tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )


# =========================================================
# PDF EXTRACTION
# =========================================================

def extract_resume_text(pdf_file):

    reader = PdfReader(pdf_file)

    text = ""

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text


# =========================================================
# SKILLS
# =========================================================

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
    "GitHub",
    "Streamlit",
    "Prompt Engineering",
    "Vector Database",
    "Data Preprocessing",
    "Pandas",
    "NumPy",
    "Scikit-learn",
    "REST APIs",
    "Sentence Transformers"
]


def find_skills(resume_text):

    resume_lower = resume_text.lower()

    found = []
    missing = []

    for skill in skills:

        if skill.lower() in resume_lower:
            found.append(skill)
        else:
            missing.append(skill)

    return found, missing


# =========================================================
# RESUME MATCH
# =========================================================

def calculate_match_score(resume_text, job_description):

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

    score = max(0, min(100, similarity * 100))

    return round(score, 2)


# =========================================================
# QUESTION GENERATOR
# =========================================================

def generate_interview_questions(
    job_description,
    found_skills,
    num_questions=5
):

    skills_text = ", ".join(found_skills)

    prompt = f"""
You are an expert technical interviewer.

Create {num_questions} interview questions
for a Junior AI Engineer fresher.

Job Description:
{job_description}

Candidate Skills:
{skills_text}

Requirements:

- Include Python questions
- Include Machine Learning questions
- Include NLP questions
- Include Generative AI questions
- Include RAG questions
- Include scenario-based questions
- Include project-based questions
- Suitable for a fresher
- Return only questions
- One question per line
"""

    result = generate_text(
        prompt,
        max_new_tokens=300
    )

    return result


def clean_questions(text):

    lines = text.split("\n")

    questions = []

    for line in lines:

        line = line.strip()

        if not line:
            continue

        line = re.sub(
            r"^\d+[\.\)]\s*",
            "",
            line
        )

        if len(line) > 15:

            questions.append(line)

    return questions[:5]


# =========================================================
# REFERENCE ANSWER
# =========================================================

def generate_reference_answer(question):

    prompt = f"""
You are an expert AI interviewer.

Interview Question:
{question}

Create a concise and technically correct
ideal answer for a Junior AI Engineer fresher.

Use simple English.
Include important technical concepts.
"""

    return generate_text(
        prompt,
        max_new_tokens=250
    )


# =========================================================
# KEYWORD EXTRACTION
# =========================================================

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
        "what",
        "how",
        "you",
        "can",
        "into",
        "use",
        "using",
        "your",
        "about"
    }

    return set(
        word
        for word in words
        if word not in stop_words
    )


# =========================================================
# ANSWER EVALUATION
# =========================================================

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

    semantic_similarity = max(
        0,
        min(1, semantic_similarity)
    )

    reference_keywords = extract_keywords(
        reference_answer
    )

    candidate_keywords = extract_keywords(
        candidate_answer
    )

    matched_keywords = (
        reference_keywords
        .intersection(candidate_keywords)
    )

    if len(reference_keywords) > 0:

        keyword_coverage = (
            len(matched_keywords)
            / len(reference_keywords)
        )

    else:

        keyword_coverage = 0

    final_score = (
        semantic_similarity * 70
        + keyword_coverage * 30
    )

    return round(
        min(100, final_score),
        2
    )


# =========================================================
# AI FEEDBACK
# =========================================================

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

Provide:

STRENGTHS:
What was explained correctly.

WEAKNESSES:
What important concepts are missing.

IMPROVEMENT:
How the candidate can improve.

INTERVIEW TIP:
One practical interview tip.

Keep everything concise and simple.
"""

    return generate_text(
        prompt,
        max_new_tokens=300
    )


# =========================================================
# IMPROVED ANSWER
# =========================================================

def generate_improved_answer(
    question,
    candidate_answer
):

    prompt = f"""
You are an expert AI Engineer
and interview coach.

Interview Question:
{question}

Candidate Answer:
{candidate_answer}

Rewrite the answer into a better
interview-quality answer.

Requirements:

- Technically correct
- Simple English
- Suitable for a fresher
- Include important concepts
- Not unnecessarily long
- Return only the improved answer
"""

    return generate_text(
        prompt,
        max_new_tokens=300
    )


# =========================================================
# INPUT SECTION
# =========================================================

st.markdown("""
<div class="card">

<div class="card-title">
📄 Step 1 — Upload Resume & Job Description
</div>

<div class="card-subtitle">
Upload your resume and paste the job description
to analyze your profile.
</div>

</div>
""", unsafe_allow_html=True)


col1, col2 = st.columns(2)


with col1:

    uploaded_resume = st.file_uploader(
        "📄 Upload Resume PDF",
        type=["pdf"]
    )


with col2:

    job_description = st.text_area(
        "💼 Job Description",
        height=220,
        placeholder="Paste the job description here..."
    )


# =========================================================
# ANALYZE BUTTON
# =========================================================

if st.button(
    "🚀 Analyze My Resume",
    use_container_width=True
):

    if uploaded_resume is None:

        st.error(
            "Please upload your resume PDF."
        )

    elif not job_description.strip():

        st.error(
            "Please enter the job description."
        )

    else:

        with st.spinner(
            "Analyzing your resume with AI..."
        ):

            resume_text = extract_resume_text(
                uploaded_resume
            )

            match_score = calculate_match_score(
                resume_text,
                job_description
            )

            found_skills, missing_skills = find_skills(
                resume_text
            )

            st.session_state.resume_text = resume_text
            st.session_state.job_description = job_description
            st.session_state.match_score = match_score
            st.session_state.found_skills = found_skills
            st.session_state.missing_skills = missing_skills

            st.session_state.analyzed = True

        st.success(
            "Resume analysis completed successfully! 🎉"
        )


# =========================================================
# DASHBOARD
# =========================================================

if st.session_state.get("analyzed", False):

    st.markdown(
        "<br>",
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class="card-title">
    📊 Resume Analysis Dashboard
    </div>
    """, unsafe_allow_html=True)

    score = st.session_state.match_score
    found = st.session_state.found_skills
    missing = st.session_state.missing_skills

    total_skills = len(found) + len(missing)

    skill_percentage = (
        len(found) / total_skills * 100
        if total_skills > 0
        else 0
    )

    metric1, metric2, metric3, metric4 = st.columns(4)

    with metric1:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-icon">🎯</div>
                <div class="metric-value">{score}%</div>
                <div class="metric-label">Resume Match</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with metric2:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-icon">✅</div>
                <div class="metric-value">{len(found)}</div>
                <div class="metric-label">Skills Found</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with metric3:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-icon">⚠️</div>
                <div class="metric-value">{len(missing)}</div>
                <div class="metric-label">Skills Missing</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with metric4:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-icon">📈</div>
                <div class="metric-value">{skill_percentage:.0f}%</div>
                <div class="metric-label">Skill Coverage</div>
            </div>
            """,
            unsafe_allow_html=True
        )


    # =====================================================
    # SKILL GAP
    # =====================================================

    st.markdown("<br>", unsafe_allow_html=True)

    skill_col1, skill_col2 = st.columns(2)


    with skill_col1:

        st.markdown("""
        <div class="card">

        <div class="card-title">
        ✅ Matching Skills
        </div>

        """, unsafe_allow_html=True)

        if found:

            badges = ""

            for skill in found:

                badges += (
                    f'<span class="skill skill-found">'
                    f'✓ {skill}'
                    f'</span>'
                )

            st.markdown(
                badges,
                unsafe_allow_html=True
            )

        else:

            st.info(
                "No matching skills detected."
            )

        st.markdown(
            "</div>",
            unsafe_allow_html=True
        )


    with skill_col2:

        st.markdown("""
        <div class="card">

        <div class="card-title">
        🎯 Skill Gaps
        </div>

        """, unsafe_allow_html=True)

        if missing:

            badges = ""

            for skill in missing:

                badges += (
                    f'<span class="skill skill-missing">'
                    f'✗ {skill}'
                    f'</span>'
                )

            st.markdown(
                badges,
                unsafe_allow_html=True
            )

        else:

            st.success(
                "Excellent! No major skill gaps detected."
            )

        st.markdown(
            "</div>",
            unsafe_allow_html=True
        )


    # =====================================================
    # INTERVIEW QUESTIONS
    # =====================================================

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
    <div class="card">

    <div class="card-title">
    💬 Step 2 — AI Interview
    </div>

    <div class="card-subtitle">
    Generate personalized technical interview questions
    based on the job description and your skills.
    </div>

    </div>
    """, unsafe_allow_html=True)


    if st.button(
        "🧠 Generate Interview Questions",
        use_container_width=True
    ):

        with st.spinner(
            "AI is preparing your interview..."
        ):

            raw_questions = generate_interview_questions(
                st.session_state.job_description,
                st.session_state.found_skills
            )

            question_list = clean_questions(
                raw_questions
            )

            if len(question_list) < 5:

                fallback_questions = [
                    "What is the difference between Machine Learning and Deep Learning?",
                    "What is NLP and where is it used?",
                    "What is Generative AI and how does an LLM work?",
                    "What is RAG and why is it useful?",
                    "Explain one AI project you have worked on."
                ]

                for q in fallback_questions:

                    if q not in question_list:

                        question_list.append(q)

                    if len(question_list) >= 5:
                        break

            st.session_state.questions = question_list[:5]

        st.success(
            "Interview questions generated! 🎉"
        )


    # =====================================================
    # INTERVIEW
    # =====================================================

    if "questions" in st.session_state:

        st.markdown("<br>", unsafe_allow_html=True)

        for i, question in enumerate(
            st.session_state.questions
        ):

            st.markdown(
                f"""
                <div class="question-box">

                    <div class="question-number">
                    QUESTION {i + 1}
                    </div>

                    <div class="question-text">
                    {question}
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )

            answer_key = f"answer_{i}"

            st.text_area(
                "✍️ Your Answer",
                key=answer_key,
                height=130,
                placeholder="Type your interview answer here..."
            )

            if st.button(
                f"🔍 Evaluate Answer {i + 1}",
                key=f"evaluate_{i}",
                use_container_width=True
            ):

                candidate_answer = st.session_state[
                    answer_key
                ]

                if not candidate_answer.strip():

                    st.warning(
                        "Please enter your answer first."
                    )

                else:

                    with st.spinner(
                        "Evaluating your answer..."
                    ):

                        reference_answer = (
                            generate_reference_answer(
                                question
                            )
                        )

                        answer_score = evaluate_answer(
                            candidate_answer,
                            reference_answer
                        )

                        feedback = generate_feedback(
                            question,
                            candidate_answer,
                            reference_answer,
                            answer_score
                        )

                        improved_answer = (
                            generate_improved_answer(
                                question,
                                candidate_answer
                            )
                        )

                    st.markdown("<br>", unsafe_allow_html=True)

                    # Score
                    if answer_score >= 85:

                        st.markdown(
                            f"""
                            <div class="result-good">
                            ⭐ Excellent Answer — {answer_score}/100
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                    elif answer_score >= 60:

                        st.markdown(
                            f"""
                            <div class="result-average">
                            👍 Good Answer — {answer_score}/100
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                    else:

                        st.markdown(
                            f"""
                            <div class="result-low">
                            📚 Needs Improvement — {answer_score}/100
                            </div>
                            """,
                            unsafe_allow_html=True
                        )


                    st.markdown("<br>", unsafe_allow_html=True)

                    feedback_col, answer_col = st.columns(2)


                    with feedback_col:

                        st.markdown("""
                        <div class="card">

                        <div class="card-title">
                        🤖 AI Feedback
                        </div>

                        """, unsafe_allow_html=True)

                        st.write(feedback)

                        st.markdown(
                            "</div>",
                            unsafe_allow_html=True
                        )


                    with answer_col:

                        st.markdown("""
                        <div class="card">

                        <div class="card-title">
                        ✨ Improved Answer
                        </div>

                        """, unsafe_allow_html=True)

                        st.write(improved_answer)

                        st.markdown(
                            "</div>",
                            unsafe_allow_html=True
                        )


    # =====================================================
    # FOOTER
    # =====================================================

    st.markdown("""
    <div class="footer">

        🤖 AI Interview Evaluator
        <br>
        Built with Python • Streamlit • NLP • Transformers
        <br><br>
        © 2026 AI Interview Evaluator

    </div>
    """, unsafe_allow_html=True)
