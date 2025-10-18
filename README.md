# TremorSense - Parkinson's Screening via Voice Analysis 🧠🔊

TremorSense is a web application designed to aid in the preliminary screening of Parkinson's Disease by analyzing vocal biomarkers from audio recordings. This project uses a machine learning model (Random Forest) trained on vocal features extracted using Parselmouth.

## Features ✨

* User registration and secure login.
* Upload `.wav` audio files containing sustained vowel sounds.
* Analysis using a pre-trained machine learning model.
* Results displayed on an interactive dashboard including:
    * Prediction (Positive/Negative)
    * Confidence Score (Gauge Chart)
    * Key Vocal Biomarkers (Jitter, Shimmer, HNR - Bar Graphs)
* User profile management with password change functionality.

## Tech Stack 🛠️

* **Backend:** Python, Flask, Flask-SQLAlchemy
* **Machine Learning:** Scikit-learn, Parselmouth (Praat)
* **Frontend:** HTML, CSS, JavaScript, Chart.js
* **Database:** SQLite

## Setup and Installation 🚀

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/NaveenRajanKS004/TREMORSENSE---Parkinson-s-voice-analysis.git]
    cd TREMORSENSE---Parkinson-s-voice-analysis
    ```
2.  **Create and activate a virtual environment:**
    ```bash
    # Use Python 3.11
    python -m venv .venv
    # Windows
    .venv\Scripts\activate
    # macOS/Linux
    # source .venv/bin/activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Run the application:**
    ```bash
    python app.py
    ```
5.  Open your web browser and go to `http://127.0.0.1:5000`.

## Disclaimer ⚠️

This tool is for informational and educational purposes only. It is **not a substitute** for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.