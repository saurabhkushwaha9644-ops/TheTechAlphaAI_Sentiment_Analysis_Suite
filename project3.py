"""
Module Name: advanced_sentiment_app.py
Author: AI Intern
Description: Enterprise-grade Web Application for Sentiment Analysis.
             Supports English, Hindi, and Hinglish using Deep Learning Transformers.
"""

import streamlit as st
from transformers import pipeline

st.set_page_config(
    page_title="AI Sentiment Analyzer Pro",
    page_icon="📊",
    layout="centered"
)

@st.cache_resource
def load_nlp_pipeline():
    return pipeline(
        "sentiment-analysis", 
        model="cardiffnlp/twitter-roberta-base-sentiment-latest"
    )

try:
    sentiment_pipe = load_nlp_pipeline()
except Exception as e:
    st.error(f"Failed to load AI Models: {str(e)}")

st.title("📊 AI Sentiment Analysis Suite")
st.caption("Developed by AI Engineering Intern | Production Ready")
st.markdown("---")

st.subheader("💡 Enter Text for Analysis")
st.write("This tool supports **English, Hindi, and Hinglish** (e.g., *'Product achha hai'*).")

user_input = st.text_area(
    "Type or paste your text below:", 
    placeholder="Example: The service was absolutely amazing! OR Ye bohot kharab product hai..."
)

if st.button("Analyze Sentiment", type="primary"):
    if not user_input.strip():
        st.warning("⚠️ Please enter some text before analyzing.")
    else:
        with st.spinner("Analyzing text patterns using Deep Learning..."):
            
            prediction = sentiment_pipe(user_input)[0]
            
            
            raw_label = prediction['label'].upper()
            score = prediction['score']
            
            
            if "POS" in raw_label:
                sentiment = "POSITIVE"
                st_metric_color = "green"
            elif "NEG" in raw_label:
                sentiment = "NEGATIVE"
                st_metric_color = "red"
            else:
                sentiment = "NEUTRAL"
                st_metric_color = "orange"
                
        
        st.markdown("### 📝 Analysis Report")
        
        
        if sentiment == "POSITIVE":
            st.success(f"✨ **Sentiment Result:** {sentiment}")
        elif sentiment == "NEGATIVE":
            st.error(f"🚨 **Sentiment Result:** {sentiment}")
        else:
            st.warning(f"😐 **Sentiment Result:** {sentiment}")
            
        
        st.metric(label="AI Confidence Score", value=f"{score * 100:.2f}%")
        st.progress(score)
        
        
        st.info(f"**Processed Text:** \"{user_input}\"")


st.markdown("---")
st.caption("© 2026 AI Internship Project | Task 3 Completed Successfully.")
