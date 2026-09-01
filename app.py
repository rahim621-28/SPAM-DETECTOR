import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load and prepare data
df = pd.read_csv("SMSSpamCollection", sep="\t", header=None, names=["label", "message"])

vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['message'])
y = df['label']

model = LogisticRegression(class_weight='balanced')
model.fit(X, y)

# Build the web page
st.title("Spam Message Detector")
st.write("Type a message below and see if it's classified as spam or not.")

user_input = st.text_area("Enter a message:")

if st.button("Check"):
    if user_input.strip() == "":
        st.warning("Please enter a message first.")
    else:
        input_vector = vectorizer.transform([user_input])
        prediction = model.predict(input_vector)[0]
        if prediction == "spam":
            st.error("This looks like SPAM 🚫")
        else:
            st.success("This looks like a legitimate message ✅")
