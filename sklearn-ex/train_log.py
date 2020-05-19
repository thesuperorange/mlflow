
import argparse

import numpy as np
from sklearn.linear_model import LogisticRegression
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score, precision_score,roc_curve,f1_score,auc
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True,  help="input file")
    ap.add_argument("-s", "--split", required=False, default=0.3, help="split ratio ex:0.2=> train/test 0.8/0.2")
    ap.add_argument("-t", "--target", required=False, default='target',  help="target column name")
   
    args = vars(ap.parse_args())

    input_data = args['input']
    split_ratio = float(args['split'])
    target_name = args['target']

    deli = ','
    if input_data.endswith('.tsv'):
        deli = '\t'
    mlflow.log_param("split_ratio", split_ratio)
    # readfile
    
    df = pd.read_csv(input_data,delimiter=deli)
    X = pd.DataFrame( df.drop([target_name], axis=1))
    y = pd.DataFrame(df[target_name])
    # split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_ratio, random_state=42)
    model = LogisticRegression()
    model.fit(X_train, y_train)

    pred = model.predict(X_train)
#acc_train = accuracy_score( y_train,pred)
    score = model.score(X, y)
    fpr, tpr, thresholds = roc_curve(y_train, pred)
    AUC = auc(fpr, tpr)
    F1 =  f1_score(y_train, pred, average='macro')
    precision = precision_score(y_train, pred, average='macro')

    print("train score:",score)
    print("train AUC:",AUC)
    print("train F1:",F1)
    print("train precision:",precision)

    mlflow.log_metric("train ACC",score)
    mlflow.log_metric("train AUC",AUC)

    mlflow.log_metric("train F1", F1)
    mlflow.log_metric("train Prec",precision)

    pred = model.predict(X_test)
    acc_test = accuracy_score( y_test,pred)
    score = model.score(X_test, y_test)

    fpr, tpr, thresholds = roc_curve(y_test,pred)
    AUC = auc(fpr, tpr)
    F1 =  f1_score(y_test, pred, average='macro')
    precision = precision_score(y_test, pred, average='macro')


    print("test score:",score)
    print("test AUC:",AUC)
    print("test F1:",F1)
    print("test precision:",precision)


    mlflow.log_metric("test ACC",score)
    mlflow.log_metric("test AUC",AUC)

    mlflow.log_metric("test F1", F1)
    mlflow.log_metric("test Prec",precision)


    mlflow.sklearn.log_model(model,"model")

