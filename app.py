

import streamlit as st
from transformers import pipeline

# Configure the application
st.set_page_config(
    page_title="Bilingual Customer Sentiment",
    page_icon="🌍"
)

st.title("🌍 Bilingual Customer Sentiment Analysis")

st.write(
    "Analyze English and German customer reviews "
    "using MarianMT and RoBERTa."
)

# Load the models once and reuse them
@st.cache_resource
def load_models():
    translator = pipeline(
    "translation",
    model="Helsinki-NLP/opus-mt-de-en",
    device=-1
)

   
from transformers import AutoTokenizer

model_name = "Roberto-Vargas/roberta-amazon-sentiment"

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    use_fast=False
)

classifier = pipeline(
    "text-classification",
    model=model_name,
    tokenizer=tokenizer,
    top_k=None,
    device=-1
)


    return translator, classifier


# User interface
language = st.selectbox(
    "Select review language",
    ["English", "German"]
)

review = st.text_area(
    "Enter your customer review",
    placeholder="Write your review here..."
)

if st.button("Analyze Sentiment"):

    if not review.strip():
        st.warning("Please enter a customer review.")

    else:
        with st.spinner("Loading models and analyzing review..."):

            translator, classifier = load_models()

            english_review = review.strip()

            # Translate only German reviews
            if language == "German":
                translation = translator(
                    english_review,
                    max_length=256
                )

                english_review = translation[0]["translation_text"]

                st.subheader("English Translation")
                st.info(english_review)

            # Predict sentiment
            predictions = classifier(
                english_review,
                truncation=True,
                max_length=128
            )[0]

            # Show the highest-scoring sentiment
            best = max(
                predictions,
                key=lambda item: item["score"]
            )

            st.subheader("Sentiment Prediction")
            st.success(
                f"{best['label'].capitalize()} "
                f"({best['score']:.1%} confidence)"
            )

            st.subheader("All Sentiment Scores")

            for result in predictions:
                st.write(
                    f"{result['label'].capitalize()}: "
                    f"{result['score']:.1%}"
                )
