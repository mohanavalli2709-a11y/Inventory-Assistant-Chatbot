🧠 Inventory Assistant – ML-Powered Decision Support System

An AI-driven Inventory Assistant built using **Machine Learning (XGBoost)** and **Streamlit** to help retailers make smarter stock decisions by predicting product performance (STOCK / DON’T STOCK).
## 🚀 Live Demo
👉 https://inventory-assistant-1.streamlit.app/

📌 Project Overview
Inventory management decisions are often reactive and manual.  
This project introduces a **chatbot-style inventory assistant** that:

- Predicts whether a product is **High-performing** or **Low-performing**
- Helps avoid overstocking and understocking
- Converts ML predictions into **actionable business decisions**

The system is designed for **non-technical users**, enabling quick insights via a clean UI.

🧩 Features
- 🔍 Product performance prediction (STOCK / DON’T STOCK)
- 🤖 Chatbot-style interactive interface
- 📊 Prediction history tracking
- 📁 CSV export of predictions
- ⚡ Real-time inference using trained ML model

🛠 Tech Stack
- **Frontend**: Streamlit  
- **Machine Learning**: XGBoost  
- **Data Processing**: pandas, numpy  
- **Model Persistence**: joblib  
- **Language**: Python 3.11  

📂 Project Structure
```text
inventory-assistant/
│── app.py                  # Streamlit application
│── model.pkl / model.joblib # Trained ML model
│── requirements.txt        # Project dependencies
│── README.md               # Project documentation
│── .gitignore              # Ignored files (secrets, venv, etc.)
