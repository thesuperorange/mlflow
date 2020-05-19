
import argparse

import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.metrics import r2_score
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
    model = LinearRegression()
    model.fit(X_train, y_train)

    print('coefficient:',model.coef_)
    print('截距:',model.intercept_ )
    mse_train = float(np.mean((model.predict(X_train) - y_train) ** 2))
    r_squared = model.score(X_train, y_train)
    print('R2 train:',r_squared)
    print("MSE train:", mse_train)


    score = model.score(X_test,y_test)
    mse_test = float(np.mean((model.predict(X_test) - y_test) ** 2))

    print('R2 test:',score) 
    print("MSE test:",mse_test)
 
    mlflow.log_metric("R2_test",score)
    mlflow.log_metric("R2_train",r_squared)
    mlflow.log_metric("MSE_test",mse_train)
    mlflow.log_metric("MSE_train",mse_test)

    mlflow.sklearn.log_model(model,"model")

