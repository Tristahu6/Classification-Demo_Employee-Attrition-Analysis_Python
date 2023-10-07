# Classification Demonstration Using IBM.csv Dataset

This repository contains a notebook that demonstrates classification using the IBM.csv dataset. The dataset has basic employee information and attrition status (whether an employee left the company).

## Data Dictionary
- Age: Employee's age
- Attrition: Status of employee attrition (whether they left the company)
- Department: Employee's department
- DistanceFromHome: Distance from home to work
- Education: Employee's education level (1-Below College; 2-College; 3-Bachelor; 4-Master; 5-Doctor)
- EducationField: Field of education
- EnvironmentSatisfaction: Level of satisfaction with the work environment (1-Low; 2-Medium; 3-High; 4-Very High)
- JobSatisfaction: Level of job satisfaction (1-Low; 2-Medium; 3-High; 4-Very High)
- MaritalStatus: Marital status of employee
- MonthlyIncome: Employee's monthly income
- NumCompaniesWorked: Number of companies worked at prior to IBM
- WorkLifeBalance: Work-life balance rating (1-Bad; 2-Good; 3-Better; 4-Best)
- YearsAtCompany: Number of years of service at IBM

## Objective of Analysis
The analysis aims to examine the relationship between various variables and employee attrition, with attrition as the outcome variable. The goal is to identify factors driving employee attrition and predict whether specific employees might leave the company. The notebook uses Logistic Regression and Random Forest Models for predictions.

## Business Value
Through the insights provided by the models, HR departments and businesses can:
- Identify and target employees at higher risk of leaving for retention efforts.
- Implement preventive measures to create a fulfilling work environment that fosters loyalty and reduces attrition.
- Highlight and promote positive company attributes that attract talent during recruitment processes.

## Key Insights & Recommendations
Insights and recommendations based on the analysis are summarized at the end of the notebook.

## Code Usage
Below is an outline of the code:

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import statsmodels.api as sm
import numpy as np 

# Load data
IBM = pd.read_csv('IBM.csv')

# Initial data preparation steps, including:
# - Label encoding
# - Data splitting
# - Dummy variable creation

# ... more code ...

# Run logistic regression
logit_model=sm.Logit(y_train, sm.add_constant(X_train))
result=logit_model.fit()

# ... more code ...

# Random forest model
# - GridSearchCV
# - Feature importance
# - Classification report
