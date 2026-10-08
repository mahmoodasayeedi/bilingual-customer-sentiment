

import streamlit as st
from transformers import pipeline, AutoTokenizer

st.set_page_config(
    page_title="Bilingual Customer Sentiment Analysis",
    page_icon="🌍",
    layout="wide"
)

# ---------- DESIGN ----------

st.markdown("""
<style>
.stApp {
    background-color: #f5f6f9;
    color: #0f172a;
}

.block-container {
    max-width: 1120px;
    padding-top: 1.5rem;
    padding-bottom: 3rem;
}

h1, h2, h3, p, label {
    color: #0f172a;
}

[data-testid="stVerticalBlockBorderWrapper"] > div {
    border-radius: 13px;
}

div.stButton > button[kind="primary"] {
    background: #f97316;
    border: 1px solid #f97316;
    color: white;
    font-weight: 700;
    border-radius: 8px;
}

div.stButton > button[kind="primary"]:hover {
    background: #ea580c;
    border-color: #ea580c;
    color: white;
}

div.stButton > button[kind="secondary"] {
    background: #e5e7eb;
    border: 1px solid #e5e7eb;
    color: #0f172a;
    font-weight: 700;
    border-radius: 8px;
}

div.stButton > button[kind="secondary"]:hover {
    background: #d1d5db;
    color: #0f172a;
}

.result-label {
    font-size: 13px;
    color: #64748b;
}

.result-value {
    font-size: 35px;
    font-weight: 800;
    line-height: 1.2;
}

.mini-card {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 9px;
    padding: 12px;
    min-height: 95px;
}

.mini-title {
    font-size: 12px;
    color: #64748b;
    margin-bottom: 6px;
}

.mini-text {
    color: #0f172a;
    font-size: 14px;
}

.score-track {
    background: #e5e7eb;
    border-radius: 999px;
    height: 12px;
    overflow: hidden;
    margin: 5px 0 14px;
}

.score-fill {
    height: 100%;
    border-radius: 999px;
}

footer {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)

# ---------- MODELS ----------

TRANSLATION_MODEL = "Helsinki-NLP/opus-mt-de-en"
SENTIMENT_MODEL = "Roberto-Vargas/roberta-amazon-sentiment"

@st.cache_resource(show_spinner="Loading AI models...")
def load_models():
    translator = pipeline(
        "translation_de_to_en",
        model=TRANSLATION_MODEL,
        device=-1
    )

    tokenizer = AutoTokenizer.from_pretrained(
        "roberta-base",
        use_fast=False
    )

    classifier = pipeline(
        "text-classification",
        model=SENTIMENT_MODEL,
        tokenizer=tokenizer,
        top_k=None,
        device=-1
    )

    return translator, classifier


# ---------- HELPERS ----------

def score_bar(label, score, color):
    percentage = score * 100

    st.markdown(
        f"""
        <div style="display:flex;
                    justify-content:space-between;
                    font-size:14px;
                    font-weight:600;">
            <span>{label}</span>
            <span>{percentage:.2f}%</span>
        </div>
        <div class="score-track">
            <div class="score-fill"
                 style="width:{percentage:.2f}%;
                        background:{color};">
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


EXAMPLES = {
    "English": {
        "Positive": (
            "I love this product. It works perfectly "
            "and is easy to use."
        ),
        "Neutral": (
            "The product is okay. It does what I "
            "expected, nothing special."
        ),
        "Negative": (
            "This product is terrible. It stopped "
            "working after two days."
        ),
    },
    "German": {
        "Positive": (
            "Ich liebe dieses Produkt. Es funktioniert "
            "perfekt und ist einfach zu bedienen."
        ),
        "Neutral": (
            "Das Produkt ist okay. Es erfüllt meine "
            "Erwartungen, aber nichts Besonderes."
        ),
        "Negative": (
            "Dieses Produkt ist schrecklich. Es hat "
            "nach zwei Tagen aufgehört zu funktionieren."
        ),
    }
}

# ---------- SESSION STATE ----------

if "review" not in st.session_state:
    st.session_state.review = EXAMPLES["English"]["Positive"]

if "result" not in st.session_state:
    st.session_state.result = None

if "language" not in st.session_state:
    st.session_state.language = "English"


def clear_review():
    st.session_state.review = ""
    st.session_state.result = None


def choose_example(sentiment):
    language = st.session_state.language
    st.session_state.review = EXAMPLES[language][sentiment]
    st.session_state.result = None


# ---------- HEADER ----------

st.title("🌍 Bilingual Customer Sentiment Analysis")

st.markdown(
    "**MarianMT Translation + Fine-tuned RoBERTa "
    "Sentiment Classification**"
)

st.write(
    "Enter a customer review below to classify it as "
    "**Positive**, **Neutral**, or **Negative** "
    "and view the model's confidence scores."
)

st.markdown(
    "🔹 **RoBERTa • Amazon Customer Reviews • English + German**"
)

st.write("")

# ---------- REVIEW CARD ----------

with st.container(border=True):
    st.markdown("**Customer Review**")

    language = st.selectbox(
        "Select Review Language",
        ["English", "German"],
        key="language"
    )

    st.text_area(
        "Enter your review",
        key="review",
        height=155,
        label_visibility="collapsed"
    )

    analyze_col, clear_col = st.columns([2, 1])

    with analyze_col:
        analyze = st.button(
            "Analyze Sentiment",
            type="primary",
            use_container_width=True
        )

    with clear_col:
        st.button(
            "Clear",
            type="secondary",
            use_container_width=True,
            on_click=clear_review
        )

    if analyze:
        review_text = st.session_state.review.strip()

        if not review_text:
            st.warning("Please enter a customer review.")
        else:
            try:
                with st.spinner("Analyzing your review..."):
                    translator, classifier = load_models()

                    english_text = review_text

                    if language == "German":
                        translated = translator(
                            review_text,
                            max_length=256
                        )
                        english_text = translated[0][
                            "translation_text"
                        ]

                    predictions = classifier(
                        english_text,
                        truncation=True,
                        max_length=128
                    )[0]

                    label_map = {
                        "LABEL_0": "Negative",
                        "LABEL_1": "Neutral",
                        "LABEL_2": "Positive",
                        "negative": "Negative",
                        "neutral": "Neutral",
                        "positive": "Positive",
                    }

                    scores = {}

                    for item in predictions:
                        raw = item["label"]
                        label = label_map.get(
                            raw,
                            raw.capitalize()
                        )
                        scores[label] = float(item["score"])

                    best_label = max(scores, key=scores.get)

                    st.session_state.result = {
                        "label": best_label,
                        "confidence": scores[best_label],
                        "scores": scores,
                        "translation": (
                            english_text
                            if language == "German"
                            else None
                        ),
                    }

                st.success("Analysis complete.")

            except Exception as error:
                st.session_state.result = None
                st.error("Analysis could not be completed.")
                st.exception(error)

# ---------- RESULTS ----------

result = st.session_state.result

if result is not None:

    if result["translation"]:
        with st.container(border=True):
            st.markdown("**English Translation**")
            st.info(result["translation"])

    with st.container(border=True):
        st.markdown("**Sentiment Prediction**")

        label = result["label"]
        confidence = result["confidence"]
        scores = result["scores"]

        colors = {
            "Positive": "#00894d",
            "Neutral": "#b7791f",
            "Negative": "#dc2626"
        }

        left, right = st.columns([3, 1])

        with left:
            st.markdown(
                '<div class="result-label">'
                'Predicted Sentiment</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                f'<div class="result-value" '
                f'style="color:{colors.get(label, "#0f172a")}">'
                f'{label}</div>',
                unsafe_allow_html=True
            )

        with right:
            st.markdown(
                '<div class="result-label">'
                'Confidence</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                f'<div class="result-value">'
                f'{confidence:.2%}</div>',
                unsafe_allow_html=True
            )

        st.write("")

        score_bar(
            "Positive",
            scores.get("Positive", 0),
            "#22c55e"
        )

        score_bar(
            "Neutral",
            scores.get("Neutral", 0),
            "#f59e0b"
        )

        score_bar(
            "Negative",
            scores.get("Negative", 0),
            "#ef4444"
        )

# ---------- EXAMPLE REVIEWS ----------

with st.container(border=True):
    st.markdown("**Example Reviews**")

    cols = st.columns(3)

    for col, sentiment in zip(
        cols,
        ["Positive", "Neutral", "Negative"]
    ):
        with col:
            st.markdown(
                f"""
                <div class="mini-card">
                    <div class="mini-title">
                        {sentiment.upper()} EXAMPLE
                    </div>
                    <div class="mini-text">
                        {EXAMPLES[language][sentiment]}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.button(
                f"Use {sentiment} Example",
                key=f"example_{sentiment}",
                use_container_width=True,
                on_click=choose_example,
                args=(sentiment,)
            )

# ---------- MODEL INFORMATION ----------

with st.container(border=True):
    st.markdown("**Model Information**")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
            <div class="mini-card">
                <div class="mini-title">SENTIMENT MODEL</div>
                <div class="mini-text">
                    <b>Fine-tuned RoBERTa</b>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            """
            <div class="mini-card">
                <div class="mini-title">LANGUAGE SUPPORT</div>
                <div class="mini-text">
                    <b>English • German</b>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            """
            <div class="mini-card">
                <div class="mini-title">PROJECT</div>
                <div class="mini-text">
                    <b>NLP Automated Customer Reviews</b>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.caption(
        "German reviews are translated to English using "
        "MarianMT before sentiment classification. "
        "Confidence scores are model estimates, "
        "not guarantees of correctness."
    )

# ---------- FOOTER ----------

st.markdown(
    """
    <p style="text-align:center;
              color:#94a3b8;
              margin-top:25px;
              font-size:13px;">
        NLP Automated Customer Reviews • Sentiment Analysis
    </p>
    """,
    unsafe_allow_html=True
)
