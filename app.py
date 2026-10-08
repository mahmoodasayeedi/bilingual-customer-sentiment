
import streamlit as st
from transformers import pipeline, AutoTokenizer

# -------------------------------------------------
# 1. Streamlit page configuration
# -------------------------------------------------

st.set_page_config(
    page_title="Bilingual Customer Sentiment",
    page_icon="🌍",
    layout="centered"
)

st.title("🌍 Bilingual Customer Sentiment Analysis")

st.write(
    "Analyze English and German customer reviews "
    "using MarianMT translation and RoBERTa sentiment analysis."
)

# -------------------------------------------------
# 2. Hugging Face model identifiers
# -------------------------------------------------

TRANSLATION_MODEL = "Helsinki-NLP/opus-mt-de-en"

SENTIMENT_MODEL = (
    "Roberto-Vargas/roberta-amazon-sentiment"
)

# Roberto's model was fine-tuned from roberta-base.
# Use the original tokenizer as a fallback.
TOKENIZER_MODEL = "roberta-base"

# -------------------------------------------------
# 3. Load and cache models
# -------------------------------------------------

@st.cache_resource
def load_models():

    # German -> English translation
    translator = pipeline(
        task="translation_de_to_en",
        model=TRANSLATION_MODEL,
        device=-1
    )

    # Load original RoBERTa tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_MODEL,
        use_fast=False
    )

    # Load Roberto's fine-tuned sentiment model
    classifier = pipeline(
        task="text-classification",
        model=SENTIMENT_MODEL,
        tokenizer=tokenizer,
        top_k=None,
        device=-1
    )

    return translator, classifier


# -------------------------------------------------
# 4. User input
# -------------------------------------------------

st.subheader("Enter a Customer Review")

language = st.selectbox(
    "Select Review Language",
    ["English", "German"]
)

review = st.text_area(
    "Customer Review",
    placeholder=(
        "Example: Das Tablet ist sehr langsam "
        "und ich bin nicht zufrieden."
    ),
    height=140
)

# -------------------------------------------------
# 5. Analyze sentiment
# -------------------------------------------------

if st.button(
    "🔍 Analyze Sentiment",
    type="primary",
    use_container_width=True
):

    if not review.strip():
        st.warning("Please enter a customer review.")

    else:

        try:

            with st.spinner(
                "Loading models and analyzing review..."
            ):

                translator, classifier = load_models()

                english_review = review.strip()

                # Translate German to English
                if language == "German":

                    translation_result = translator(
                        english_review,
                        max_length=256
                    )

                    english_review = (
                        translation_result[0]["translation_text"]
                    )

                # Classify English review
                predictions = classifier(
                    english_review,
                    truncation=True,
                    max_length=128
                )[0]

            # ---------------------------------
            # 6. Show translation
            # ---------------------------------

            if language == "German":

                st.subheader("🇬🇧 English Translation")

                st.info(english_review)

            # ---------------------------------
            # 7. Show sentiment result
            # ---------------------------------

            best_prediction = max(
                predictions,
                key=lambda item: item["score"]
            )

            sentiment = best_prediction["label"].capitalize()
            confidence = best_prediction["score"]

            st.subheader("Sentiment Prediction")

            if sentiment.lower() == "positive":

                st.success(
                    f"😊 POSITIVE — {confidence:.1%} confidence"
                )

            elif sentiment.lower() == "negative":

                st.error(
                    f"😞 NEGATIVE — {confidence:.1%} confidence"
                )

            elif sentiment.lower() == "neutral":

                st.warning(
                    f"😐 NEUTRAL — {confidence:.1%} confidence"
                )

            else:

                st.info(
                    f"{sentiment} — {confidence:.1%} confidence"
                )

            # ---------------------------------
            # 8. Show all sentiment scores
            # ---------------------------------

            st.subheader("Sentiment Confidence Scores")

            for result in predictions:

                label = result["label"].capitalize()
                score = result["score"]

                st.write(f"**{label}: {score:.1%}**")

                st.progress(
                    min(max(float(score), 0.0), 1.0)
                )

            st.caption(
                "Sentiment model: Roberto-Vargas/"
                "roberta-amazon-sentiment"
            )

        except Exception as error:

            st.error(
                "An error occurred while loading "
                "the models or analyzing the review."
            )

            st.exception(error)


# -------------------------------------------------
# 9. Footer
# -------------------------------------------------

st.divider()

st.caption(
    "German-to-English translation: "
    "Helsinki-NLP/opus-mt-de-en"
)

st.caption(
    "Sentiment classification: "
    "Roberto-Vargas/roberta-amazon-sentiment"
)
