# 🚗 DriveMind AI
### AI-Powered Automotive Intelligence Platform for Driver Behaviour Analysis & Natural Language Vehicle Insights

DriveMind AI is an end-to-end automotive analytics platform that combines **Generative AI, Machine Learning, and Interactive Data Analytics** to help users understand vehicle behaviour using natural language.

The platform enables users to query large-scale automotive telemetry data in plain English, predicts aggressive driving behaviour using Machine Learning, and provides interactive visual analytics for better decision making.

Built using **Streamlit, Llama 2, Ollama, Random Forest, SQLite, and Python**, DriveMind AI demonstrates how AI can simplify automotive data analysis while improving vehicle safety insights.

---

# 📌 Problem Statement

Modern vehicles continuously generate massive amounts of telemetry data such as speed, engine temperature, brake pressure, steering angle, RPM, GPS location, and weather conditions.

Although this data is valuable, extracting meaningful insights usually requires technical knowledge of SQL or data analysis tools.

Additionally,

- Unsafe driving behaviour often remains unnoticed.
- Large datasets are difficult to analyze manually.
- Non-technical users cannot easily query automotive databases.
- Raw sensor data provides little actionable insight without AI.

This creates a need for an intelligent platform capable of transforming raw vehicle data into meaningful business insights.

---

# 💡 Solution

DriveMind AI addresses these challenges by integrating **Large Language Models (LLMs)** with **Machine Learning** and **Interactive Analytics**.

The platform allows users to:

- Ask questions in natural language.
- Automatically generate SQL queries.
- Analyze over **150,000 automotive records**.
- Detect aggressive driving behaviour.
- Generate AI-powered risk insights.
- Visualize automotive trends through interactive dashboards.

Instead of writing SQL manually, users simply ask questions like:

> Find vehicles with brake pressure above 80.

or

> Show aggressive driving records.

The system converts these requests into SQL queries, retrieves the required data, performs analysis, and presents AI-generated insights.

---

# ✨ Key Features

## 🤖 Natural Language to SQL

- Converts plain English queries into SQL using Llama 2.
- Executes SQL on a 150K+ automotive dataset.
- Displays generated SQL.
- Returns filtered vehicle records.
- Generates AI-powered textual insights.

---

## 🚗 Driver Behaviour Detection

Random Forest model classifies driving behaviour into:

- Normal Driving
- Aggressive Driving

Prediction is based on:

- Speed
- Acceleration
- Brake Pressure
- Steering Angle
- Engine Temperature
- RPM

---

## 📊 Interactive Analytics Dashboard

Provides:

- Dataset exploration
- Driver behaviour distribution
- Vehicle analytics
- Histograms
- Scatter plots
- Box plots
- Performance metrics

---

## 📈 Machine Learning Performance

The platform displays:

- Accuracy
- F1 Score
- ROC-AUC
- Confusion Matrix
- Feature Importance
- Classification Report

---

## ⚡ Live Driving Prediction

Users can manually adjust vehicle sensor values and instantly predict whether the driving behaviour is:

- Normal
- Aggressive

The prediction page also provides:

- Confidence score
- Radar chart
- Safe operating ranges
- Prediction probability

---

## 💬 AI Insights

Beyond raw query results, DriveMind AI automatically summarizes findings by calculating:

- Average speed
- Average RPM
- Average brake pressure
- Average acceleration

It also generates intelligent alerts such as:

> High brake pressure detected — possible aggressive braking.

or

> High engine temperature detected — possible overheating risk.

---

# 🏗 System Architecture

(Architecture Diagram Here)

```
User
   │
   ▼
Streamlit Web Application
   │
   ├──────────────► Llama 2 (Ollama)
   │                    │
   │                    ▼
   │              SQL Query Generation
   │
   ▼
SQLite Database (150K+ Records)
   │
   ▼
Analytics Engine
   │
   ├── Visualizations
   ├── AI Insights
   ├── ML Prediction
   ▼
Interactive Dashboard
```

---

# 🧠 Machine Learning Model

Algorithm:

**Random Forest Classifier**

Prediction Classes:

- Normal Driving
- Aggressive Driving

Input Features:

- Speed
- Acceleration
- Brake Pressure
- Steering Angle
- RPM
- Engine Temperature

---

## 📊 Model Performance

| Metric | Value |
|----------|---------|
| Accuracy | **99.99%** |
| F1 Score | **99.98%** |
| ROC-AUC | **1.00** |
| Test Size | **30,000 Records** |

Only **3 incorrect predictions** were observed during testing.

---

## 🔍 Feature Importance

Top contributing features:

1. Acceleration — **48.6%**
2. Steering Angle — **45.9%**
3. Brake Pressure
4. Speed
5. Engine Temperature
6. RPM

---

# 📂 Dataset

Dataset Size:

**150,000+ vehicle telemetry records**

Unique Vehicles:

**500**

Data Duration:

January 2024

Features include:

- Vehicle ID
- Session ID
- Timestamp
- Speed
- Acceleration
- Brake Pressure
- Steering Angle
- Engine Temperature
- RPM
- Fuel Level
- GPS Coordinates
- Road Type
- Weather
- Driving Behaviour Label

---

# 📊 Dashboard Modules

## 1️⃣ Natural Language Query Engine

- Natural Language → SQL
- SQL Preview
- Query Result Table
- AI Insights
- Pie Chart
- Histograms

---

## 2️⃣ ML Performance Dashboard

- Accuracy
- F1 Score
- ROC-AUC
- Confusion Matrix
- Feature Importance
- Classification Report

---

## 3️⃣ Live Prediction

- Real-time prediction
- Sensor sliders
- Confidence score
- Radar chart
- Safe operating ranges

---

## 4️⃣ Data Explorer

- Summary statistics
- Pie chart
- Scatter plots
- Box plots
- Raw dataset explorer

---

# 🛠 Tech Stack

### Programming

- Python

### Framework

- Streamlit

### Machine Learning

- Scikit-learn
- Random Forest

### Generative AI

- Llama 2
- Ollama

### Database

- SQLite

### Libraries

- Pandas
- NumPy
- Plotly
- Matplotlib

---

# 🚀 Future Enhancements

Planned improvements include:

- Weather-aware vehicle health prediction.
- Accident probability estimation using environmental conditions.
- Predictive vehicle maintenance.
- OBD-II real-time vehicle integration.
- Cloud deployment with REST APIs.
- Fleet management dashboard.
- Voice-enabled automotive assistant.

---

# 👩‍💻 Author

**Sakshi Zagade**

Computer Engineer

Passionate about Artificial Intelligence, Machine Learning, Generative AI, and Full-Stack Development.

📧 LinkedIn: *https://www.linkedin.com/in/sakshi-zagade-175016249/*

📂 GitHub: *https://github.com/SakshiZagade*

---

## ⭐ If you found this project interesting, consider giving it a star!
