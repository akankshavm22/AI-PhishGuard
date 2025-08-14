import pandas as pd
import numpy as np
import re
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import joblib
from nltk.corpus import stopwords
from urllib.parse import urlparse
import requests
import nltk
import matplotlib.pyplot as plt
import seaborn as sns

nltk.download('stopwords')
STOPWORDS = set(stopwords.words("english"))

# ----------------- Email Cleaning ----------------- #
def clean_email_body(email_body):
    email_body = re.sub(r'[^a-zA-Z\s]', '', email_body)
    email_body = ' '.join([word.lower() for word in email_body.split() if word.lower() not in STOPWORDS])
    return email_body

# ----------------- URL Features ----------------- #
def extract_url_features(email_body):
    urls = re.findall(r'(https?://[^\s]+)', email_body)
    features = []
    for url in urls:
        parsed_url = urlparse(url)
        domain_length = len(parsed_url.netloc)
        path_length = len(parsed_url.path)
        protocol = 1 if parsed_url.scheme in ['http', 'https'] else 0
        try:
            status_code = requests.head(url, timeout=3, allow_redirects=True).status_code
        except:
            status_code = 0
        features.append([domain_length, path_length, protocol, status_code])
    return np.mean(features, axis=0) if features else 0

# ----------------- Train Model ----------------- #
def train_model(data):
    data['cleaned_body'] = data['email_body'].apply(clean_email_body)
    data['url_features'] = data['email_body'].apply(extract_url_features)
    
    X = pd.concat([data['cleaned_body'], data['url_features']], axis=1)
    X.columns = ['email_body', 'url_features']
    y = data['label']
    
    body_vectorizer = CountVectorizer()
    model = make_pipeline(body_vectorizer, MultinomialNB())
    
    progress_text = "Training in progress..."
    my_bar = st.progress(0, text=progress_text)
    import time
    for percent_complete in range(0, 101, 20):
        time.sleep(0.2)
        my_bar.progress(percent_complete, text=progress_text)
    
    model.fit(X['email_body'], y)
    joblib.dump(model, 'phishing_model.pkl')
    my_bar.progress(100, text="Training Complete!")
    st.success("Model trained and saved successfully!")

# ----------------- Predict ----------------- #
def predict_phishing(model, email_body):
    cleaned_body = clean_email_body(email_body)
    url_features = extract_url_features(email_body)
    X = pd.DataFrame([[cleaned_body, url_features]], columns=['email_body', 'url_features'])
    prediction = model.predict(X['email_body'])
    return 'Phishing' if prediction[0] == 1 else 'Safe'

# ----------------- Batch Analysis ----------------- #
def batch_analysis(model, df):
    results = []
    for index, row in df.iterrows():
        email_text = row['email_body']
        prediction = predict_phishing(model, email_text)
        urls = re.findall(r'(https?://[^\s]+)', email_text)
        results.append({
            'Email Body': email_text[:100] + "..." if len(email_text) > 100 else email_text,
            'Prediction': prediction,
            'Extracted URLs': ", ".join(urls),
            'URL Count': len(urls)
        })
    return pd.DataFrame(results)

# ----------------- Themes ----------------- #
def apply_theme(theme):
    if theme == "Dark":
        st.markdown("""
        <style>
        .stApp {background-color: #1e1e2f; color: #f0f0f0;}
        .stButton>button {background-color: #0a84ff; color: white;}
        .stDataFrame td {color: #f0f0f0;}
        </style>
        """, unsafe_allow_html=True)
        return {'bg': '#2e2e3f', 'phishing': '#ff4b4b', 'safe': '#4cd964', 'plot_phishing':'#ff4b4b', 'plot_safe':'#4cd964'}
    elif theme == "Light":
        st.markdown("""
        <style>
        .stApp {background-color: #f9f9f9; color: #1e1e2f;}
        .stButton>button {background-color: #4CAF50; color: white;}
        .stDataFrame td {color: #1e1e2f;}
        </style>
        """, unsafe_allow_html=True)
        return {'bg': '#ffffff', 'phishing': '#ff0000', 'safe': '#00aa00', 'plot_phishing':'#ff0000', 'plot_safe':'#00aa00'}
    elif theme == "Cyberpunk":
        st.markdown("""
        <style>
        .stApp {background-color: #0f0f1a; color: #ff00ff;}
        .stButton>button {background-color: #00ffff; color: #0f0f1a;}
        .stDataFrame td {color: #ff00ff;}
        </style>
        """, unsafe_allow_html=True)
        return {'bg': '#1a0f2e', 'phishing': '#ff00ff', 'safe': '#00ffff', 'plot_phishing':'#ff00ff', 'plot_safe':'#00ffff'}
    elif theme == "Matrix":
        st.markdown("""
        <style>
        .stApp {background-color: #000000; color: #00ff00;}
        .stButton>button {background-color: #00ff00; color: black;}
        .stDataFrame td {color: #00ff00;}
        </style>
        """, unsafe_allow_html=True)
        return {'bg': '#001100', 'phishing': '#ff0000', 'safe': '#00ff00', 'plot_phishing':'#ff0000', 'plot_safe':'#00ff00'}

# ----------------- Analytics ----------------- #
def plot_history_charts(history, colors):
    if history.empty:
        st.info("No data to plot yet.")
        return
    
    # Pie Chart
    phishing_count = history['Prediction'].value_counts()
    plt.figure(figsize=(5,5))
    plt.pie(phishing_count, labels=phishing_count.index, autopct='%1.1f%%',
            colors=[colors['plot_phishing'], colors['plot_safe']])
    st.subheader("📊 Phishing vs Safe Emails")
    st.pyplot(plt.gcf())
    
    # Bar Chart: URL count
    if 'URL Count' in history.columns:
        plt.figure(figsize=(6,4))
        sns.histplot(history['URL Count'], bins=10, color=colors['safe'])
        plt.title("🔗 Number of URLs per Email")
        plt.xlabel("URL Count")
        plt.ylabel("Frequency")
        st.pyplot(plt.gcf())

# ----------------- Dashboard ----------------- #
def main():
    st.set_page_config(page_title="AI-PhishGuard", layout="wide")
    
    if 'history' not in st.session_state:
        st.session_state.history = pd.DataFrame(columns=['Email Body', 'Prediction', 'Extracted URLs', 'URL Count'])
    
    # Sidebar
    st.sidebar.image("logo.png", width=180)
    st.sidebar.title("Navigation & Settings")
    
    page = st.sidebar.radio("Go to:", ["Check Email", "Batch Analysis", "Train Model", "History & Analytics", "Instructions"])
    theme = st.sidebar.selectbox("Select Theme", ["Dark", "Light", "Cyberpunk", "Matrix"])
    colors = apply_theme(theme)
    
    # ---------------- Single Email ---------------- #
    if page == "Check Email":
        st.header("🛡️ Check Single Email")
        email_input = st.text_area("Paste the Email Body Here:", height=200)
        if st.button("Analyze Email"):
            try:
                model = joblib.load('phishing_model.pkl')
                result = predict_phishing(model, email_input)
                icon = "⚠️" if result=='Phishing' else "✅"
                st.markdown(f"""
                    <div style='border-radius:10px; padding:20px; background-color:{colors["bg"]}'>
                        <h3 style='color:{colors["phishing"] if result=="Phishing" else colors["safe"]}; text-align:center'>{icon} {result}</h3>
                    </div>
                """, unsafe_allow_html=True)
                urls = re.findall(r'(https?://[^\s]+)', email_input)
                if urls:
                    with st.expander("🔗 Extracted URLs"):
                        for u in urls:
                            st.markdown(f"[{u}]({u})", unsafe_allow_html=True)
                st.session_state.history = pd.concat([st.session_state.history,
                                                     pd.DataFrame([{
                                                         'Email Body': email_input[:100] + "...",
                                                         'Prediction': result,
                                                         'Extracted URLs': ", ".join(urls),
                                                         'URL Count': len(urls)
                                                     }])], ignore_index=True)
            except FileNotFoundError:
                st.warning("No trained model found. Please train first.")
    
    # ---------------- Batch Analysis ---------------- #
    elif page == "Batch Analysis":
        st.header("📊 Batch Email Analysis")
        uploaded_file = st.file_uploader("Upload CSV with 'email_body' column", type='csv')
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            if 'email_body' in df.columns:
                try:
                    model = joblib.load('phishing_model.pkl')
                    st.write("Analyzing emails...")
                    results_df = batch_analysis(model, df)
                    results_df['Prediction'] = results_df['Prediction'].apply(lambda x: f"⚠️ {x}" if x=='Phishing' else f"✅ {x}")
                    st.dataframe(results_df)
                    st.download_button("Download Results CSV", results_df.to_csv(index=False), file_name="phishing_results.csv")
                    st.session_state.history = pd.concat([st.session_state.history, results_df], ignore_index=True)
                except FileNotFoundError:
                    st.warning("No trained model found. Please train first.")
            else:
                st.error("CSV must contain 'email_body' column")
    
    # ---------------- Train Model ---------------- #
    elif page == "Train Model":
        st.header("📊 Train Model")
        uploaded_file = st.file_uploader("Upload CSV with 'email_body' and 'label' columns", type='csv', key="train_upload")
        if uploaded_file:
            data = pd.read_csv(uploaded_file)
            if 'email_body' in data.columns and 'label' in data.columns:
                train_model(data)
            else:
                st.error("CSV must contain 'email_body' and 'label' columns")
    
    # ---------------- History & Analytics ---------------- #
    elif page == "History & Analytics":
        st.header("📚 Analysis History & Analytics")
        if not st.session_state.history.empty:
            st.dataframe(st.session_state.history)
            st.download_button("Download Full History CSV", st.session_state.history.to_csv(index=False), file_name="analysis_history.csv")
            plot_history_charts(st.session_state.history, colors)
        else:
            st.info("No history yet. Perform email checks or batch analysis to see results here.")
    
    # ---------------- Instructions ---------------- #
    elif page == "Instructions":
        st.header("📖 How to Use Dashboard")
        st.markdown("""
        1. **Check Email:** Single email phishing analysis.  
        2. **Batch Analysis:** CSV with multiple emails.  
        3. **Train Model:** Upload labeled dataset to retrain AI.  
        4. **History & Analytics:** View session history and charts; download CSV.  
        5. **Themes:** Switch Dark, Light, Cyberpunk, Matrix from sidebar.  
        6. **Result Colors:** Red/⚠️ = Phishing, Green/✅ = Safe.  
        """)
        
if __name__ == "__main__":
    main()
