# White Box - Supervised ML Project  
**Name:** [Your Full Name]  
**Institution:** [Your Institution Name]  
**Date:** November 2025  

---

## Project Overview

This repository contains two machine learning projects completed as part of the **White Box Supervised ML Assignment**.  
Both projects apply **interpretable (white-box)** machine learning models to real-world datasets — one for **classification** and the other for **regression**.

The work demonstrates the end-to-end data science process:
- Data Cleaning & Preprocessing  
- Exploratory Data Analysis (EDA)  
- Model Building (Logistic/Linear Regression & KNN)  
- Model Evaluation and Comparison  
- Interpretability and Insight Communication  

---

## 1. Classification Topic – Credit Score Prediction

### Problem Statement  
A large financial institution aims to automate its credit scoring process to eliminate bias and human error.  
The goal is to classify customers into **Good**, **Standard**, or **Poor** credit score categories based on financial and behavioral data.

### Dataset  
**Source:** [Kaggle – Credit Score Classification (by Paris Rohan)](https://www.kaggle.com/datasets/parisrohan/credit-score-classification)

### Models Used  
- **Logistic Regression**  
- **K-Nearest Neighbors (KNN) Classifier**

### Features Tested  
- **3-feature models:** `Outstanding_Debt`, `Delay_from_due_date`, `Changed_Credit_Limit`  
- **5-feature models:** + `Monthly_Inhand_Salary`, `Credit_Utilization_Ratio`

### Evaluation Metrics  
- Accuracy  
- Precision, Recall, F1-Score  
- Confusion Matrix  

### Results Summary  
| Model | Features | Accuracy | Key Observation |
|--------|-----------|-----------|----------------|
| Logistic Regression | 3 | 58.7% | Baseline model, limited non-linear capture |
| Logistic Regression | 5 | 58.9% | Minor improvement |
| KNN Classifier | 3 | **67.4%** | Best performer; balanced and explainable |
| KNN Classifier | 5 | 59.5% | Accuracy dropped due to correlated/noisy features |

**Conclusion:**  
The **KNN model with 3 key behavioral features** provided the most balanced and interpretable predictions.  
It demonstrates that customer behavior metrics are stronger indicators of credit reliability than income-related variables.

---

## 2. Regression Topic – Video Game Sales Prediction

### Problem Statement  
A video game company wants to predict which games will be **successful in global markets** to optimize investment and marketing.  
The goal is to predict **Global_Sales** using regional sales and release information.

### Dataset  
**Source:** [Kaggle – Video Game Sales (by Gregorut)](https://www.kaggle.com/datasets/gregorut/videogamesales)

### Models Used  
- **Linear Regression**  
- **K-Nearest Neighbours (KNN) Regressor**

### Features Tested  
- **3-feature models:** `NA_Sales`, `EU_Sales`, `JP_Sales`  
- **5-feature models:** + `Other_Sales`, `Year`

### Evaluation Metrics  
- R² (Coefficient of Determination)  
- RMSE (Root Mean Squared Error)

### Results Summary  
| Model | Features | R² | RMSE | Observation |
|--------|-----------|------|-------|-------------|
| Linear Regression | 3 | 0.9952 | 0.1438 | Strong fit using main regional sales |
| Linear Regression | 5 | **1.0000** | **0.0054** | Perfect linear relationship |
| KNN Regressor | 3 | 0.8014 | 0.9218 | Good fit but less precise |
| KNN Regressor | 5 | 0.7759 | 0.9792 | Slight overfitting from extra features |

**Conclusion:**  
Since `Global_Sales` is a direct sum of regional sales, **Linear Regression** perfectly models the relationship.  
KNN performs well but is less efficient for purely linear patterns.

---

## Tools and Libraries

- Python 3.11+  
- Pandas, NumPy – Data processing  
- Matplotlib, Seaborn – Visualization  
- Scikit-learn – Model building and evaluation  
- Jupyter Notebook – Experimentation and documentation  

---

## Key Learnings

- Simpler, interpretable models (Logistic/Linear Regression) often outperform complex ones when data relationships are clear.  
- Data cleaning and feature selection strongly influence performance.  
- Correlation and multicollinearity must be reviewed before modeling.  
- Visual analysis and interpretability are crucial for “white box” decision-making.

---

## References

- Paris Rohan, *Credit Score Classification Dataset*, Kaggle  
- Gregorut, *Video Game Sales Dataset*, Kaggle  
- Scikit-learn Documentation: https://scikit-learn.org/stable/  
- Matplotlib & Seaborn Documentation

---

**Author:** Abdulrahman Abdulla Alali  
**Date:** November 2025

