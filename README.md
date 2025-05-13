
# **AI-Based Air Quality Prediction System**  

This repository contains the implementation of an AI-driven solution designed to predict the **Air Quality Index (AQI)** for major cities using historical air pollutant data and weather information. The project leverages machine learning algorithms for both **regression** (predicting exact AQI values) and **classification** (categorizing AQI levels), enabling better urban planning and proactive public health measures.  

---

## **Table of Contents**  
1. [Problem Statement](#problem-statement)  
2. [Project Structure](#project-structure)  
3. [System Requirements](#system-requirements)  
4. [Data Description](#data-description)  
5. [Installation](#installation)  
6. [Usage](#usage)  
7. [Modeling Techniques](#modeling-techniques)  
8. [Evaluation Metrics](#evaluation-metrics)  
9. [Results and Insights](#results-and-insights)  
10. [Deployment](#deployment)  
11. [Future Scope](#future-scope)  
12. [Contributors](#contributors)  

---

## **Problem Statement**  
Air pollution is a critical environmental and public health concern, with harmful pollutants affecting the air quality of urban and industrial areas. This project aims to predict the **Air Quality Index (AQI)** using machine learning algorithms to assist in early warning systems, urban planning, and health advisories.  

---

## **Project Structure**  
```
├── AQI_Prediction_Colab_Notebook.ipynb
├── data
│   ├── raw_data.csv
│   └── processed_data.csv
│
├── models
│   ├── regression_model.pkl
│   └── classification_model.pkl
│
├── README.md
└── requirements.txt
```

---

## **System Requirements**  
- **Hardware:**  
    - Minimum 8GB RAM  
    - Intel i5 Processor or equivalent  

- **Software:**  
    - Python 3.8+  
    - IDE: Google Colab / Jupyter Notebook  
    - Libraries: pandas, numpy, scikit-learn, matplotlib, seaborn, Flask (for deployment)  

---

## **Data Description**  
- **Source:** Kaggle – "Air Quality Data in India"  
- **Attributes:** PM2.5, PM10, CO, NO2, O3, SO2, temperature, humidity, timestamp, location  
- **Size and structure:** Multi-city, time-series pollutant data  

---

## **Installation**  
Clone the repository and navigate to the project directory:  
```bash
git clone https://github.com/kalamaashique-glitch/kalam.git
cd kalam
```

Install the required dependencies:  
```bash
pip install -r requirements.txt
```

---

## **Usage**  
Open the notebook in Google Colab or Jupyter Notebook and run all cells in sequence:  
```bash
jupyter notebook AQI_Prediction_Colab_Notebook.ipynb
```

---

## **Modeling Techniques**  
- **Regression Models:**  
    - Linear Regression  
    - Random Forest Regressor  
    - XGBoost Regressor  

- **Classification Models:**  
    - Logistic Regression  
    - Random Forest Classifier  
    - SVM  
    - Naive Bayes  

---

## **Evaluation Metrics**  
- **Regression:**  
    - Mean Absolute Error (MAE)  
    - Root Mean Squared Error (RMSE)  
    - R² Score  

- **Classification:**  
    - Accuracy  
    - Precision  
    - Recall  
    - F1-Score  

---

## **Results and Insights**  
- Achieved high predictive accuracy for AQI forecasting.  
- PM2.5 and PM10 were identified as the most significant contributors to air quality.  
- Seasonal trends showed pollution peaks during winter due to atmospheric inversion.  
- Geographic patterns highlighted industrial hubs with consistently poor air quality.  

---

## **Deployment**  
- The model is ready for deployment using **Flask** or **Streamlit** for interactive web-based prediction.  

---

## **Future Scope**  
- Integration with real-time air quality APIs for live predictions.  
- Expansion to more cities globally for broader applicability.  
- Enhancement of the model with real-time anomaly detection for extreme pollution events.  

---

## **Contributors**  
- **KALAM AASHIQUE BEIG.K** — Team Lead, Model Building, Evaluation  
- **SYED ABDULLA.H** — Data Collection, Preprocessing, EDA  
- **PRASANNA.K** — Feature Engineering, Visualization, Deployment Exploration  
