# UPI-_SHIELD
🛡️ UPI Shield: A Real-Time Fraud Detection System UPI Shield is an end-to-end machine learning application that demonstrates a real-world fraud detection system. It uses an XGBoost model to analyze transaction data and predict the probability of fraud in real-time. The system is served via a high-performance FastAPI backend and features an interactive web interface built with Streamlit for easy demonstration.

(Action Required: This is a sample screenshot. Replace the link above with a link to your own screenshot of the Streamlit app! You can upload your screenshot to a site like Imgur to get a link.)

✨ Core Features Real-Time Predictions: Get instant fraud risk scores for transaction data submitted through the UI or API.

High-Performance API: Built with FastAPI for fast, asynchronous request handling, making it suitable for production-like environments.

Powerful ML Model: Utilizes an XGBoost Classifier, a state-of-the-art algorithm optimized to handle the highly imbalanced nature of fraud datasets.

Interactive UI: A user-friendly web interface built with Streamlit allows for easy input of transaction data and visual feedback on the model's predictions.

Complete ML Workflow: Covers the entire machine learning lifecycle, from data preprocessing and model training to API deployment and front-end integration.

🛠️ Tech Stack Backend: Python, FastAPI

Machine Learning: Scikit-learn, XGBoost, Pandas, Joblib

Frontend: Streamlit

API Server: Uvicorn

Data Communication: Requests

💾 Dataset This project uses the "Credit Card Fraud Detection" dataset from Kaggle.

Important: Due to its large size (144 MB), the creditcard.csv data file is not included in this repository, as it exceeds GitHub's file size limit. You must download it manually and place it in the root of the project folder to run the training script.

Download Link: Kaggle Credit Card Fraud Detection Dataset

🚀 Local Setup and How to Run Follow these steps to set up and run the project on your local machine (instructions are for Windows using PowerShell).

Prerequisites Python 3.8+
Git

Clone the Repository git clone https://github.com/architpr/UPI-_SHIELD.git cd UPI-_SHIELD

Create and Activate a Virtual Environment python -m venv venv .\venv\Scripts\Activate.ps1

Install Dependencies A requirements.txt file is included for easy installation of all required packages.

pip install -r requirements.txt

(If you haven't created one, run pip freeze > requirements.txt first.)

Download the Dataset Download the creditcard.csv file from the Kaggle link and place it in the UPI-_SHIELD root directory.

Train the Model (One-time setup) Run the training script. This will create the fraud_model.pkl file that the API uses.

python train_model.py

Run the Application You need to run two servers in two separate terminals.
Terminal 1: Start the FastAPI Backend

uvicorn main:app --reload

The API will be running at http://127.0.0.1:8000.

Terminal 2: Start the Streamlit Frontend

streamlit run streamlit_app.py

The web application will automatically open in your browser.

🔗 API Endpoint The FastAPI server provides an interactive documentation page to test the API.

Interactive Docs: http://127.0.0.1:8000/docs

Endpoint: POST /predict

Description: Analyzes transaction data and returns a fraud prediction and probability score.

Request Body: A JSON object containing 30 features (Time, V1-V28, Amount).
