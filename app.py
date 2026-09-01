import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="Inbox Inspection — Spam Detector", page_icon="📮", layout="centered")


@st.cache_resource
def load_model():
    df = pd.read_csv("SMSSpamCollection", sep="\t", header=None, names=["label", "message"])
    vectorizer = TfidfVectorizer(stop_words="english")
    X = vectorizer.fit_transform(df["message"])
    y = df["label"]
    model = LogisticRegression(class_weight="balanced")
    model.fit(X, y)
    return vectorizer, model


vectorizer, model = load_model()

# ---- Styling ----
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@500;700&family=IBM+Plex+Sans:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}

.stApp {
    background-color: #E4DCC8;
    color: #2B2620;
}

.inspection-header {
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 700;
    font-size: 2.1rem;
    letter-spacing: -0.01em;
    color: #2B2620;
    border-bottom: 3px solid #2B2620;
    padding-bottom: 0.6rem;
    margin-bottom: 0.3rem;
}

.inspection-sub {
    font-family: 'IBM Plex Sans', sans-serif;
    color: #5C5648;
    font-size: 1rem;
    margin-bottom: 2rem;
}

.stTextArea textarea {
    background-color: #FAF6EE !important;
    border: 1.5px dashed #8A8270 !important;
    border-radius: 2px !important;
    font-family: 'IBM Plex Mono', monospace;
    color: #2B2620 !important;
}

.stButton button {
    background-color: #2B2620;
    color: #E4DCC8;
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 700;
    letter-spacing: 0.05em;
    border-radius: 2px;
    border: none;
    padding: 0.5rem 1.6rem;
}

.stButton button:hover {
    background-color: #4A4436;
    color: #E4DCC8;
}

.stamp-box {
    font-family: 'IBM Plex Mono', monospace;
    border: 3px solid;
    border-radius: 4px;
    padding: 1.2rem 1.5rem;
    margin-top: 1.5rem;
    font-weight: 700;
    font-size: 1.3rem;
    letter-spacing: 0.08em;
    transform: rotate(-1deg);
    display: inline-block;
}

.stamp-rejected {
    border-color: #B3261E;
    color: #B3261E;
    background-color: rgba(179, 38, 30, 0.08);
}

.stamp-cleared {
    border-color: #2D6A4F;
    color: #2D6A4F;
    background-color: rgba(45, 106, 79, 0.08);
}
</style>
""",
    unsafe_allow_html=True,
)

# ---- Content ----
st.markdown('<div class="inspection-header">Inbox Inspection</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="inspection-sub">Drop a message below. It gets read, checked against known spam patterns, and stamped.</div>',
    unsafe_allow_html=True,
)

user_input = st.text_area(
    "Message to inspect",
    height=140,
    label_visibility="collapsed",
    placeholder="Paste or type a message here...",
)

check = st.button("Inspect message")

if check:
    if user_input.strip() == "":
        st.warning("Nothing to inspect yet — enter a message first.")
    else:
        vector = vectorizer.transform([user_input])
        prediction = model.predict(vector)[0]
        if prediction == "spam":
            st.markdown('<div class="stamp-box stamp-rejected">✕ REJECTED — SPAM</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="stamp-box stamp-cleared">✓ CLEARED — LEGITIMATE</div>', unsafe_allow_html=True)
