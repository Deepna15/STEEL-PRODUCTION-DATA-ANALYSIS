import numpy as np
import pandas as pd
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from data_loading import load_steel_data
from data_preprocessing import(remove_duplicates,handle_missing_values,detect_outliers,encode_categorical_variables,data_consistency)
from eda import split_and_normalize_data

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR,"results")
os.makedirs(RESULTS_DIR,exist_ok=True)

def train_random_forest(x_train,y_train):
    model = RandomForestRegressor(random_state=42)
    model.fit(x_train,y_train)
    return model
def train_svm(x_train,y_train):
    model = SVR()
    model.fit(x_train,y_train)
    return model
def train_mlp(x_train,y_train):
    model = MLPRegressor(max_iter=500,random_state=42)
    model.fit(x_train,y_train)
    return model
def train_gaussian_process(x_train,y_train):
    model = GaussianProcessRegressor()
    model.fit(x_train,y_train)
    return model

# Model Evaluation
def evaluate_model(model,x_test,y_test):
    y_pred = model.predict(x_test)
    rmse = np.sqrt(mean_squared_error(y_test,y_pred))
    mae = mean_absolute_error(y_test,y_pred)
    r2 = r2_score(y_test,y_pred)
    return rmse,mae,r2,y_pred

if __name__ == "__main__":

    df_train,df_test = load_steel_data()
    df_train = remove_duplicates(df_train)
    df_train = handle_missing_values(df_train)
    df_train = detect_outliers(df_train)
    df_train = encode_categorical_variables(df_train)
    df_train = data_consistency(df_train)
    numeric_cols = df_train.select_dtypes(include=["int64", "float64"]).columns

    TARGET_COLUMN = "output"

    X_train, X_test, X_val, y_train, y_test, y_val, scaler = \
        split_and_normalize_data(df_train, TARGET_COLUMN)
    
    models = {"RANDOM_FOREST": train_random_forest(X_train,y_train), "SVM": train_svm(X_train,y_train),"MLP": train_mlp(X_train,y_train),"GAUSSIAN_PROCESS": train_gaussian_process(X_train,y_train)}
    results = []
    for name, model in models.items():
        rmse,mae,r2,y_pred = evaluate_model(model,X_test,y_test)
        results.append({"Model": name, "RMSE": rmse, "MAE": mae, "R2": r2})
        pred_df = pd.DataFrame({"y_true": y_test, "y_pred":y_pred})
        pred_df.to_csv(os.path.join(RESULTS_DIR, f"{name}_predictions.csv"),index=False)
        

    