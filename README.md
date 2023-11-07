# Credit Card Approval Prediction Model

## Project for MLZoomcamp course midterm Credit Card Approval

## Problem Statement:
Should a customer credit card application be approved or denied.
Dataset consists of target variable to identify if a credit card application is approved/denied to a customer. We will be using machince learning techniques to predict if an application is approved for a customer on denied. Machine learning make help us with learning from past applications data and make decision in realtime which helps with reduce human effort and increases efficiency

## Dataset:
Dataset is obtained from Kaggle - https://www.kaggle.com/datasets/youssefaboelwafa/credit-card-approval

Credit Card Approval datasets consists of the columns listed below:

'Gender','Age','Debt','Married','BankCustomer','EducationLevel','Ethnicity',
'YearsEmployed','PriorDefault','Employed','CreditScore','DriversLicense','Citizen',
'ZipCode','Income','ApprovalStatus'

## Solution:
Credit Card Approval is classification problem. We will exploring the various classification algorithms Logistic Regression, Decision Tree Classifier, Random Forest Classifier and Xgboost Classifier. Based on the evaluation metrics we will show the best performing model. 

# Getting Started:
To run the project, please follow the below steps:

## Prerequisties:
- Docker: you will need to have Docker installed on your computer. 

## Installation:
1. Clone the repo.

2. Build the Docker image using the follwing command:
    ```
    docker build -t creditcard-approval .
    ```
3. Run Docker container using the below command:
    ```
    docker run -it --rm -p 9695:9695 creditcard-approval
    ```
4. Once the above statement is executed, you will be noticing, 
   starting guicorn and Listeing at: http://0.0.0.0:9695


## Prediction Service:
Once docker is up and running, you can start using the model by running the below command in a new terminal window or new tab: 
    ```
    python predict-docker.py
    ```



