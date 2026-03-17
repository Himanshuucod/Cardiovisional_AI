# CardioPredict AI – Cardiovascular Disease Prediction System

## Overview

CardioPredict AI is a machine learning based web application designed to predict the risk of cardiovascular disease using patient health parameters. The system analyzes medical attributes and provides a prediction result through an interactive web interface.

The goal of this project is to assist in early detection of heart disease by using data-driven insights and machine learning algorithms.

---

## Features

* Heart disease risk prediction using Machine Learning
* Web interface built with Flask
* User authentication system (Login / Register)
* Interactive dashboard for prediction
* Database storage for user accounts
* Clean and responsive UI
* Scalable architecture for healthcare analytics

---

## Technologies Used

### Backend

* Python
* Flask

### Machine Learning

* Scikit-learn
* XGBoost
* Pandas
* NumPy

### Frontend

* HTML
* CSS
* JavaScript

### Database

* SQLite

---

## Input Parameters Used for Prediction

The prediction model analyzes the following medical features:

* Age
* Gender
* Height
* Weight
* Systolic Blood Pressure
* Diastolic Blood Pressure
* Cholesterol Level
* Glucose Level
* Smoking Habit
* Alcohol Intake
* Physical Activity

---

## Project Structure

```
CardioPredict-AI
│
├── app.py
├── preprocess.py
├── train_model.py
│
├── templates
│   ├── dashboard.html
│   ├── login.html
│   ├── predict.html
│   ├── register.html
│   └── result.html
│
├── static
│
├── model
│   ├── model.pkl
│   ├── scaler.pkl
│
├── database
│   └── users.db
│
└── README.md
```

---

## Installation

1. Clone the repository

```
git clone https://github.com/yourusername/CardioPredict-AI.git
```

2. Navigate to the project folder

```
cd CardioPredict-AI
```

3. Install required dependencies

```
pip install -r requirements.txt
```

4. Run the application

```
python app.py
```

5. Open the browser and go to

```
http://127.0.0.1:5000
```

---

## Future Improvements

* PDF report generation for predictions
* Visualization dashboard for medical analytics
* Integration with larger healthcare datasets
* Cloud deployment for global accessibility
* Explainable AI for medical interpretation

---

## Author

Himanshu Sahu
