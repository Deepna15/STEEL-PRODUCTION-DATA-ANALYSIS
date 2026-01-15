import pandas as pd
import numpy as np
import os
import time
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR,"results")
FIGURES_DIR = os.path.join(BASE_DIR,"figures")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

def calculate_metrics(y_true,y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true,y_pred)
    r2 = r2_score(y_true,y_pred)
    return {"rmse:": rmse, "MAE:": mae, "R2:": r2}
def create_performance_table(df_results):
    df_results.columns = df_results.columns.str.strip()
    file_path = os.path.join(RESULTS_DIR,"performance_table.csv")
    df_results.to_csv(file_path)
    return df_results

# Visulaisations
def plot_model_comparison(df_results):
    df_results.columns = df_results.columns.str.strip()
    per_met = ["RMSE", "MAE", "R2"]
    df_results.set_index("Model")[per_met].plot(kind= "bar",figsize=(10,6))
    plt.xlabel("Models")
    plt.ylabel("Score")
    plt.title("Model Comaprison - Barchart")
    plt.savefig(os.path.join(FIGURES_DIR,"model_comparison.png"))
    plt.close()

def plot_predictions_vs_actual(y_true,y_pred,model_name):
    plt.figure(figsize=(6,6))
    plt.scatter(y_true,y_pred,alpha=0.6)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val],[min_val,max_val], linestyle="--" )
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title(f"{model_name}: Prediction vs Actual")
    plt.savefig(os.path.join(FIGURES_DIR, "predicated_actual.png"))
    plt.close()

def plot_residuals(y_true, y_pred, model_name):
    residuals = y_true - y_pred
    plt.figure(figsize=(6,6))
    plt.scatter(y_pred,residuals,alpha=0.6)
    plt.xlabel("predicated Values")
    plt.ylabel("Residuals")
    plt.title(f"{model_name}: Redidual Plot")
    plt.savefig(os.path.join(FIGURES_DIR, "residuals.png"))
    plt.close()

def fix_results_dataframe(df_results):
    df_results = df_results.rename(columns={"Unnamed: 0": "Metric"})
    df_results = df_results.set_index("Metric").T
    df_results = df_results.reset_index().rename(columns={"index": "Model"})
    return df_results
def convert_metrics_to_numeric(df_results):
    metrics = ["RMSE", "MAE", "R2"]
    for m in metrics:
        df_results[m] = pd.to_numeric(df_results[m], errors="coerce")
    return df_results



if __name__ == "__main__":

   model_evaluation = os.path.join(RESULTS_DIR, "performance_table.csv")
   df_results = pd.read_csv(model_evaluation)
   df_results = fix_results_dataframe(df_results)
   df_results = convert_metrics_to_numeric(df_results)
   print("Columns in df_results")
   print(df_results.columns)
   plot_model_comparison(df_results)

              

