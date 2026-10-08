
import streamlit as st

st.set_page_config(
    page_title="Bilingual Customer Sentiment",
    page_icon="🌍"
)

st.title("🌍 Bilingual Customer Sentiment Analysis")

st.write(
    "Analyze customer reviews in English or German "
    "using MarianMT and RoBERTa."
)

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
        st.info("Model integration coming next!")
