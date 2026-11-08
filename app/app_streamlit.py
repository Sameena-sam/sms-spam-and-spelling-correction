"""
app_streamlit.py
================
Deployment web page for the SMS spam classifier (Half 2, deployment marks).

It loads the model exported by the notebook (`spam_model.joblib`), takes a
message typed by the user, and predicts ham/spam with a confidence score.

Run:
    pip install streamlit scikit-learn joblib
    streamlit run app/app_streamlit.py
"""

import os

import numpy as np
import streamlit as st
import joblib

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "spam_model.joblib")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def confidence(model, text):
    """Return P(spam) in [0, 1]. Uses predict_proba when available, otherwise
    squashes the SVM decision_function through a logistic to get a 0-1 score."""
    if hasattr(model, "predict_proba"):
        return float(model.predict_proba([text])[0][1])
    score = float(model.decision_function([text])[0])
    return 1.0 / (1.0 + np.exp(-score))


st.set_page_config(page_title="SMS Spam Classifier", page_icon="📩")
st.title("📩 SMS Spam Classifier")
st.write("Type a message and the model will predict whether it is **spam** or "
         "**ham** (legitimate).")

try:
    model = load_model()
except Exception as e:
    st.error(f"Could not load '{MODEL_PATH}'. Run the notebook first to create it.\n\n{e}")
    st.stop()

message = st.text_area("Message", height=140,
                       placeholder="e.g. Congratulations! You won a free prize...")

if st.button("Classify", type="primary"):
    if not message.strip():
        st.warning("Please enter a message.")
    else:
        pred = model.predict([message])[0]
        p_spam = confidence(model, message)
        if pred == 1:
            st.error(f"🚫 SPAM  —  confidence {p_spam:.1%}")
        else:
            st.success(f"✅ HAM (legitimate)  —  spam probability {p_spam:.1%}")
        st.progress(p_spam)

st.caption("Model: tuned TF-IDF + linear classifier, trained on the UCI SMS "
           "Spam Collection.")
