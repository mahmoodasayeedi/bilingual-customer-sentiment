🌍 Bilingual Customer Sentiment Analysis
An interactive English–German customer review sentiment analyzer powered by MarianMT and a fine-tuned RoBERTa classifier. Enter a review in either language and receive a Positive, Neutral, or Negative prediction with confidence scores. German reviews are translated into English before classification.
🚀 Try the live Streamlit app
Project overview
Customer reviews contain valuable feedback, but reading and categorizing them manually is time-consuming. This application demonstrates a simple NLP workflow for analyzing English and German customer reviews through one interface.
Features
- Select English or German from the language dropdown.
- Enter a customer review or choose an example.
- Translate German reviews into English using MarianMT.
- Predict sentiment using a fine-tuned RoBERTa model.
- View the predicted class, individual confidence scores, and the English translation when applicable.
- Clear the input and try another review.
How it works
Customer review
      |
      v
Select language (English / German)
      |
      +-- English ----------------------------+
      |                                       |
      +-- German --> MarianMT translation ----+
                                              |
                                              v
                                 Fine-tuned RoBERTa
                                              |
                                              v
                              Negative / Neutral / Positive
                                   + confidence scores
Models
Component	Hugging Face model	Purpose
Translation	Helsinki-NLP/opus-mt-de-en	Translate German reviews into English
Sentiment classification	Roberto-Vargas/roberta-amazon-sentiment	Predict three sentiment classes
Tokenizer fallback	roberta-base	Tokenize English text for RoBERTa


The application uses the roberta-base tokenizer as a fallback for the fine-tuned model. This assumes the fine-tuning process retained the original RoBERTa vocabulary and token IDs.
Dataset
The underlying sentiment-classification work used Amazon product reviews from the project's 1429_1.csv dataset (approximately 34,597 cleaned reviews). Reviews were prepared for three-class classification using their star ratings:
Star rating	Sentiment
1–2 stars	Negative
3 stars	Neutral
4–5 stars	Positive


The cleaned data was strongly imbalanced: approximately 93.33% Positive, 4.33% Neutral, and 2.34% Negative. This is why macro F1-score is important alongside accuracy: it gives equal weight to each class rather than allowing the majority class to dominate the headline metric.
The dataset is not included in this app repository. Refer to the team project repository for the broader NLP project and data preparation work.
Method and model selection
1. Prepare and label customer reviews using the three-class rating mapping.
2. Compare a traditional TF-IDF + balanced Logistic Regression baseline against a fine-tuned RoBERTa classifier.
3. Select the fine-tuned RoBERTa model for the sentiment analysis application.
4. Add MarianMT German-to-English translation as an inference-time extension; the sentiment classifier itself still processes English text.
5. Serve both models through a Streamlit web interface, caching loaded models for repeated predictions.
Results and key findings
The following results describe the English-language held-out evaluation of the sentiment models in the underlying team project. They are not an independent evaluation of the German translation-and-classification pipeline.
Model	Test accuracy	Macro F1-score
Always-positive majority baseline	93.3%	~0.32
TF-IDF + balanced Logistic Regression	93.6%	0.654
Fine-tuned RoBERTa	94.6%	0.709


For the fine-tuned RoBERTa model, reported class F1-scores were 0.717 (Negative), 0.435 (Neutral), and 0.976 (Positive).
Key finding: High accuracy alone can be misleading on imbalanced review data. RoBERTa improved macro F1-score over the traditional baseline, although neutral reviews remain the hardest class to classify.
Live demo
App: https://bilingual-customer-sentiment.streamlit.app/
To try it:
1. Open the app.
2. Choose English or German in Select Review Language.
3. Type or select an example review.
4. Click Analyze Sentiment.
5. Inspect the predicted sentiment and confidence scores. German inputs also show their English translation.
Example German review:
Das Tablet ist sehr langsam und ich bin nicht zufrieden.

This review expresses dissatisfaction and would ordinarily be expected to have negative sentiment; the actual prediction and scores depend on the running models.
Run locally
Requirements: Python 3.11, an internet connection for the first model download, and sufficient RAM to load both transformer models.
# Clone the repository
git clone https://github.com/mahmoodasayeedi/bilingual-customer-sentiment.git
cd bilingual-customer-sentiment

# Create and activate a virtual environment
python -m venv .venv

# macOS / Linux
source .venv/bin/activate

# Windows (PowerShell)
 .venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Start the app
streamlit run app.py
The browser will open the Streamlit interface. The first prediction may take longer because the Hugging Face models must be downloaded and initialized. No API key is required for the public model repositories used here.
Project structure
bilingual-customer-sentiment/
├── app.py             # Streamlit UI, translation and sentiment inference
├── requirements.txt   # Python dependencies
└── README.md          # Project documentation
Large datasets and pretrained model weights are not committed to this application repository; models are loaded from Hugging Face.
Limitations and future improvements
- German accuracy has not been separately measured. Translation may change nuances, tone, idioms, or negation.
- Neutral reviews are challenging, as reflected by the lower neutral-class F1-score.
- Dataset imbalance means accuracy alone is not a sufficient measure of model quality.
- First-run loading can be slow, especially on resource-limited Streamlit hosting.
- Tokenizer compatibility should be confirmed against the original fine-tuning setup; the app currently uses the base RoBERTa tokenizer as a fallback.
- Confidence scores are model outputs, not guarantees that predictions are correct.
Potential next steps include evaluating on a labeled German review set, improving neutral sentiment classification, verifying calibration, and optimizing model loading.
Credits and references
- NLP Automated Customer Reviews — team project repository
- Sentiment model: Roberto Vargas — Roberto-Vargas/roberta-amazon-sentiment
- Translation model: Helsinki-NLP/opus-mt-de-en
- RoBERTa base model: FacebookAI/roberta-base
- Libraries: Streamlit, Hugging Face Transformers, PyTorch
This repository is the bilingual Streamlit demo. It builds on the team's sentiment analysis work and does not modify the original team repository or Roberto's hosted model.

## 👥 Project Team

This project was developed collaboratively as part of the **NLP Automated Customer Reviews** project.

- **Mahmooda Sayeedir** — Bilingual Streamlit Application Development and Deployment
- **Roberto Vargas** — Project Coordinator and Team Collaboration

### Acknowledgments

Special thanks to **Roberto Vargas** for developing and fine-tuning the RoBERTa sentiment classification model used in this application.
