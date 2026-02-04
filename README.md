**Problem Statement**

The objective of this project is to solve a supervised machine learning classification problem using different classification algorithms such as Logistic Regression, Decision Tree, kNN, Naive Bayes, Random Forest (Ensemble) and XGBoost (Ensemble). The dataset is analyzed, preprocessed, and used to train on these six machine learning models. The trained models are further evaluated using metrics such as Accuracy, AUC Score, Precision, Recall, F1 Score and Matthews Correlation Coefficient (MCC Score). GitHub Repository is then created and then deployed using a Streamlit web application to provide an interactive interface for model comparison and prediction.
**Dataset Description**

The dataset used for this assignment was obtained from Kaggle. It contains 9.709 instances and 19 input features. The dataset includes both numerical and categorical attributes related to customer information. The target variable represents whether credit card is approved to a particular customer, making this a classification problem. Total number of customers for whom cards were rejected is 8,426 and for 1,283 customers, it got approved. This makes the dataset an imbalanced one.
**Models Used** 

The following machine learning classification models were implemented and evaluated on the same dataset:
1. Logistic Regression  
2. Decision Tree Classifier  
3. K-Nearest Neighbors (KNN)  
4. Naive Bayes Classifier  
5. Random Forest (Ensemble Model)  
6. XGBoost (Ensemble Model)
**Comparison Table**

 <img width="637" height="371" alt="image" src="https://github.com/user-attachments/assets/7e1456cf-570b-4bc1-ae67-c242f0b897db" />

**Observations**

ML Model Name	Observation about model performance
Logistic Regression:	Accuracy is low (55.8%) and precision (0.15) is very poor but moderate recall (0.50). F1 and MCC are very low. LR model underperforms and struggles to detect minority class.
Decision Tree: 	Accuracy is higher (76.98%) but recall is very low (0.17). Precision is slightly better than Logistic Regression. F1 and MCC are low, which indicates overfitting and poor generalization.
kNN: Higher accuracy (85.53%) but extremely low recall (0.01) and F1 (0.02). Model mostly predicts majority class and fails on minority class, which is unsuitable for imbalanced data.
Naive Bayes: Accuracy is moderate (80.74%), precision is slightly better (0.19), but recall is low (0.15). F1 and MCC are also low. NB model performs better than kNN and Logistic Regression in precision but still weak overall.
Random Forest (Ensemble): The highest accuracy among all the models (86.77%) and good precision (0.50), but recall is almost zero (0.0039). F1 and MCC are low. RF model performs well on majority class but fails to capture minority class.
XGBoost (Ensemble): Slightly better than Random Forest in precision (0.57) and MCC (0.0779). Accuracy is high (86.82%) but recall is very low (0.0156) and F1 is poor. XGB model is effective on majority class but not balanced predictions.
