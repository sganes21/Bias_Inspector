import streamlit as st
from newspaper import Article
from urllib.parse import urlparse
import os
from openai import OpenAI
import nltk

# Downloading NLTK resource 

nltk.download('punkt')
nltk.download('punkt_tab')

# Bias Inspector Title & Subtitle

st.title("Bias Inspector")
st.write("Let's Get Biased!")

# Taking in API key as environmental variable

api_key = st.secrets["OPENAI_API_KEY"]
client = OpenAI(api_key=api_key)

# Setting up pre-trained model
model_name = "ft:gpt-4.1-mini-2025-04-14:duke-university::C0Gn9Szw"

# Extracting article title, source,heading, tags, text using python library

def scrape_article(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        article.nlp()
    except Exception as e:
        st.error(f"Error scraping the article: {e}")
        return None
    # Extract main fields
    title = article.title or ""
    text = article.text or ""
    tags = ", ".join(article.keywords) if article.keywords else ""
    heading = article.text.split("\n")[0] if article.text else ""
    source = urlparse(url).netloc or ""
    # Minimal check
    if not text and not title:
        st.error("No meaningful content extracted.")
        return None
    return {
        "title": title,
        "tags": tags,
        "heading": heading,
        "source": source,
        "text": text,
    }

# Setting up prompt

def create_prompt(article_data):
    return f"""
Given the following news article information:

Title: {article_data.get('title', '')}
Tags: {article_data.get('tags', '')}
Heading: {article_data.get('heading', '')}
Source: {article_data.get('source', '')}
Text: {article_data.get('text', '')}

Answer ONLY with EXACTLY one of these bias labels: "left", "center", or "right".
Do NOT provide any additional explanation or text.
"""

# Predicting bias using model

def get_bias_prediction(prompt):
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0
        )
        content = response.choices[0].message.content.strip().lower()
        if content in {"left", "center", "right"}:
            return content
        else:
            return "unknown"
    except Exception as e:
        st.error(f"Error calling bias model: {e}")
        return "error"

# App front end inputs

url_input = st.text_input("Please input the URL to your news article:")

if st.button("Analyze Bias") and url_input:
    with st.spinner("Extracting and analyzing the article..."):
        article_data = scrape_article(url_input)
        if article_data:
            prompt = create_prompt(article_data)
            bias = get_bias_prediction(prompt)
            st.markdown(f"### 🏷️ Predicted Political Bias: **{bias.capitalize()}**")
            # Reveal extracted data optionally
            with st.expander("Show extracted article details"):
                st.write(article_data)
        else:
            st.warning("Could not extract article content. Try a different link.")
elif url_input:
    st.info("After entering a URL, click 'Analyze Bias' to get the prediction.")

st.caption("Powered by OpenAI. Supports major news sites. Python newspaper4k library manages text extraction. Results may depend on extraction quality.")




