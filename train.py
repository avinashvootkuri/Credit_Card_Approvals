import pickle

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
import xgboost as xgb
from sklearn.metrics import f1_score


# parameters
eta = 0.01
output_file = f'model_eta={eta}.bin'

# data preparation
data = pd.read_csv("Data/cc_approvals.data", header=None)

cols = ['Gender','Age','Debt','Married','BankCustomer','EducationLevel','Ethnicity',
        'YearsEmployed','PriorDefault','Employed','CreditScore','DriversLicense','Citizen',
        'ZipCode','Income','ApprovalStatus']

data.columns = cols

data_df = data.applymap(lambda x:np.nan if x == '?' else x)

data_df2 = data_df.dropna()

print(data_df2.columns)

def data_prep(df):
    ## Change ApprovalStatus from +,- to 1, 0
    df['ApprovalStatus'] = df['ApprovalStatus'].map({'+' : 1, '-': 0})

    ## Change zipcode from object to int
    df['ZipCode'] = df['ZipCode'].astype(int)

    ## Change Age from object to float
    df['Age'] = df['Age'].astype(float)

    return df

data_df_clean = data_prep(df=data_df2)


df_full_train, df_test = train_test_split(data_df_clean, test_size=0.2, random_state=1)

df_full_train = df_full_train.reset_index(drop=True)

df_test = df_test.reset_index(drop=True)

y_train = df_full_train.ApprovalStatus.values
y_test = df_test.ApprovalStatus.values

del df_full_train['ApprovalStatus']
del df_test['ApprovalStatus']


def train(df_full_train, y_train, xgb_params):
    dicts_full_train = df_full_train.to_dict(orient='records')
    dv = DictVectorizer(sparse=False)
    
    X_full_train = dv.fit_transform(dicts_full_train)

    feature_names=dv.get_feature_names_out()

    dfulltrain = xgb.DMatrix(X_full_train, label=y_train)

    model = xgb.train(xgb_params, dfulltrain, num_boost_round=175)

    return dv, model

def predict(df, dv, model):
    dicts_test = df_test.to_dict(orient='records')  
    X_test = dv.transform(dicts_test)

    dtest = xgb.DMatrix(X_test,label=y_test)
    
    y_pred = model.predict(dtest)

    return y_pred
    

xgb_params = {
    'eta': eta, 
    'max_depth': 3,
    'min_child_weight': 10,

    'objective': 'binary:logistic',
    'eval_metric': 'auc',

    'nthread': 8,
    'seed': 1,
    'verbosity': 1,
    }

dv, model = train(df_full_train, y_train, xgb_params=xgb_params)

y_pred = predict(df_test, dv, model)

score = f1_score(y_test, y_pred >= 0.6)

print(f'f1-score={score}')

with open(output_file, 'wb') as f_out:
    pickle.dump((dv, model), f_out)

print(f'the model is saved to {output_file}')
print("Finished running Train.py")


