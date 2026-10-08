
import streamlit as st
from transformers import pipeline, AutoTokenizer
from urllib.parse import quote

st.set_page_config(
    page_title="Bilingual Customer Sentiment Analysis",
    page_icon="🌍",
    layout="wide"
)

# ---------- STYLE ----------

st.markdown("""
<style>
.stApp {
    background: #fafafa;
    color: #111827;
}

.block-container {
    max-width: 1100px;
    padding-top: 1.6rem;
}

h1, h2, h3 {
    color: #111827;
}

div.stButton > button[kind="primary"] {
    background: #f97316;
    border-color: #f97316;
    color: white;
    font-weight: 700;
}

div.stButton > button[kind="primary"]:hover {
    background: #ea580c;
    border-color: #ea580c;
    color: white;
}

div.stButton > button[kind="secondary"] {
    background: #e5e7eb;
    border-color: #e5e7eb;
    color: #111827;
}

.prediction {
    text-align: center;
    font-size: 32px;
    font-weight: 800;
    margin: 15px 0;
}

.score-track {
    background: #e5e7eb;
    height: 8px;
    border-radius: 999px;
    overflow: hidden;
    margin: 6px 0 18px;
}

.score-fill {
    height: 100%;
    background: #fb923c;
    border-radius: 999px;
}
</style>
""", unsafe_allow_html=True)

# ---------- MODELS ----------

@st.cache_resource(show_spinner="Loading models...")
def load_models():
    translator = pipeline(
        "translation_de_to_en",
        model="Helsinki-NLP/opus-mt-de-en",
        device=-1
    )

    tokenizer = AutoTokenizer.from_pretrained(
        "roberta-base",
        use_fast=False
    )

    classifier = pipeline(
        "text-classification",
        model="Roberto-Vargas/roberta-amazon-sentiment",
        tokenizer=tokenizer,
        top_k=None,
        device=-1
    )

    return translator, classifier

# ---------- EXAMPLES ----------

EXAMPLES = {
    "English": [
        "Love this tablet. The screen is bright and easy to use.",
        "It stopped working after two weeks.",
        "It's okay. Does what it says, but nothing special."
    ],
    "German": [
        "Ich liebe dieses Tablet. Der Bildschirm ist hell.",
        "Es funktioniert nach zwei Wochen nicht mehr.",
        "Es ist okay, aber nichts Besonderes."
    ]
}

# ---------- SESSION ----------

if "review" not in st.session_state:
    st.session_state.review = (
        "Love this tablet. The screen is bright and easy to use."
    )

if "language" not in st.session_state:
    st.session_state.language = "English"

if "result" not in st.session_state:
    st.session_state.result = None


def clear_review():
    st.session_state.review = ""
    st.session_state.result = None


def set_example(text):
    st.session_state.review = text
    st.session_state.result = None


# ---------- HEADER ----------

st.title("🌍 Bilingual Customer Sentiment Analysis")

st.markdown(
    "Classifies a customer review as "
    "**negative, neutral, or positive**. "
    "German reviews are translated to English "
    "before sentiment analysis."
)

st.caption(
    "MarianMT Translation + Fine-tuned RoBERTa"
)

st.write("")

# ---------- MAIN LAYOUT ----------

left, right = st.columns([1, 1], gap="medium")

with left:
    with st.container(border=True):
        st.markdown("**Customer review**")

        language = st.selectbox(
            "Select Review Language",
            ["English", "German"],
            key="language"
        )

        st.text_area(
            "Enter your review",
            key="review",
            height=180,
            label_visibility="collapsed"
        )

        clear_col, analyze_col = st.columns([1, 1.5])

        with clear_col:
            st.button(
                "Clear",
                use_container_width=True,
                on_click=clear_review
            )

        with analyze_col:
            analyze = st.button(
                "Analyze Sentiment",
                type="primary",
                use_container_width=True
            )

with right:
    with st.container(border=True):
        st.markdown("**Predicted sentiment**")

        if st.session_state.result:
            result = st.session_state.result

            label = result["label"]
            scores = result["scores"]

            colors = {
                "Positive": "#15803d",
                "Neutral": "#b45309",
                "Negative": "#dc2626"
            }

            st.markdown(
                f'<div class="prediction" '
                f'style="color:{colors.get(label, "#111827")}">'
                f'{label.lower()}</div>',
                unsafe_allow_html=True
            )

            st.divider()

            for name in ["Positive", "Neutral", "Negative"]:
                score = scores.get(name, 0.0)

                st.markdown(
                    f"""
                    <div style="display:flex;
                                justify-content:space-between;">
                        <span>{name.lower()}</span>
                        <span>{score:.0%}</span>
                    </div>
                    <div class="score-track">
                        <div class="score-fill"
                             style="width:{score * 100:.2f}%">
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        else:
            st.info(
                "Enter a review and click Analyze Sentiment "
                "to see the prediction."
            )

# ---------- RUN ANALYSIS ----------

if analyze:
    review_text = st.session_state.review.strip()

    if not review_text:
        st.warning("Please enter a customer review.")
    else:
        try:
            with st.spinner("Analyzing review..."):
                translator, classifier = load_models()

                english_text = review_text

                if language == "German":
                    translation = translator(
                        review_text,
                        max_length=256
                    )
                    english_text = translation[0]["translation_text"]

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
                    "positive": "Positive"
                }

                scores = {
                    label_map.get(
                        p["label"],
                        p["label"].capitalize()
                    ): float(p["score"])
                    for p in predictions
                }

                label = max(scores, key=scores.get)

                st.session_state.result = {
                    "label": label,
                    "scores": scores,
                    "translation": (
                        english_text
                        if language == "German"
                        else None
                    )
                }

            st.rerun()

        except Exception as error:
            st.error("Analysis failed.")
            st.exception(error)

# ---------- TRANSLATION ----------

if st.session_state.result:
    translation = st.session_state.result["translation"]

    if translation:
        with st.container(border=True):
            st.markdown("**English translation**")
            st.info(translation)

# ---------- EXAMPLES ----------

st.write("")
st.markdown("**Examples**")

for example in EXAMPLES[language]:
    st.button(
        example,
        key=f"example_{language}_{example}",
        on_click=set_example,
        args=(example,)
    )

# ---------- SHARE ----------

st.divider()

review_value = quote(st.session_state.review, safe="")
language_value = quote(language, safe="")

share_url = (
    "https://bilingual-customer-sentiment.streamlit.app/"
    f"?language={language_value}&review={review_value}"
)

st.link_button(
    "🔗 Share review via link",
    share_url,
    use_container_width=True
)

st.caption(
    "Sentiment model: Roberto's fine-tuned RoBERTa | "
    "Translation model: MarianMT"
)
