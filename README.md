# AI-PhishGuard 🛡️
AI-PhishGuard is an AI-powered phishing email detection tool with a beautiful Streamlit dashboard.   It uses Natural Language Processing (NLP) and Machine Learning to classify emails as **Phishing** or **Safe**, and provides URL extraction, batch analysis, training capability, and analytics.

## 🚀 Features
- **Single Email Analysis**: Check if an email is phishing or safe.
- **Batch Analysis**: Upload a CSV of multiple emails for bulk detection.
- **Train Your Model**: Upload a labeled dataset to retrain the AI.
- **Theme Support**: Dark, Light, Cyberpunk, and Matrix modes.
- **Analytics Dashboard**: Pie charts, bar charts, and downloadable history.

## 📂 Folder Structure
<pre>
AI-PhishGuard/
│
├── phishing_detector.py  # Main dashboard script
├── phishing_model.pkl   # Pre-trained ML model
├── requirements.txt   # Required Python packages
├── logo.png      # App logo
├── README.md   # Documentation
└── sample_dataset.csv   # Example dataset for training & testing
</pre>

## 🛠 Installation
1. Clone the repository:
   <pre>
     git clone https://github.com/yourusername/AI-PhishGuard.git
     cd AI-PhishGuard
   </pre>
3. Install Dependenices:
   <pre>
     pip install -r requirements.txt
   </pre>
5. Run the app:
   <pre>
     streamlit run phishing_detector.py
   </pre>

## 🚀 Usage
1. Single Email Analysis
   - Select Check Email in the sidebar.
   - Paste the email text in the input box.
   - Click Analyze Email to get results.

2. Batch Email Analysis
   - Select Batch Analysis in the sidebar.
   - Upload a CSV file containing an email_body column.
   - View predictions and download results.

3. Train a Model
   - Select Train Model in the sidebar.
   - Upload a CSV file with columns email_body and label (1 = Phishing, 0 = Safe).
   - The model will be retrained and saved as phishing_model.pkl.

4. View Analytics
   - Select History & Analytics to see:
     - Phishing vs Safe email distribution
     - URL count distribution

## 🗂 Dataset Format
Example sample_dataset.csv:
| email_body | label |
|------------|-------|
| Dear user, your account will be suspended unless you verify now at http://phishy-bank-login.com | 1 |
| Meeting scheduled for tomorrow at 10 AM. Please confirm your attendance. | 0 |
| URGENT: Your PayPal account has been locked. Click here to restore access: https://secure-paypal-reset.com | 1 |

Column Descriptions:
- email_body: Text of the email
- label:
  - 1 = Phishing
  - 0 = Safe
 
## 🤖 Model Details (phishing_model.pkl)
- Type: scikit-learn Pipeline
- Steps:
  1. CountVectorizer
     - Converts email text into numerical feature vectors (Bag-of-Words approach).
     - Token pattern: Words with at least 2 letters.
     - Lowercase conversion enabled.
  3. Multinomial Naive Bayes Classifier
     - Classes: 0 = Safe, 1 = Phishing.
     - rained on labeled email datasets.
- Usage:
  - Loaded at runtime using joblib.load()
  - Takes processed text as input
  - Returns prediction as Phishing or Safe
 
## 🖼 Screenshots
<img width="1918" height="1020" alt="image" src="https://github.com/user-attachments/assets/aca14181-d124-4da1-b615-e8bec7d16a71" />
<img width="1918" height="1013" alt="image" src="https://github.com/user-attachments/assets/4e3b3070-3455-4153-b292-4e83be706dc1" />
<img width="1918" height="1017" alt="image" src="https://github.com/user-attachments/assets/dd76c5ff-09e4-4b3d-a399-7d7c69d54aa5" />


