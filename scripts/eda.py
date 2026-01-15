import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from data_loading import load_steel_data
from data_preprocessing import preprocess_data

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURES_DIR = os.path.join(BASE_DIR,"figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

df_train, df_test = load_steel_data()
df_train = preprocess_data(df_train)
df_test = preprocess_data(df_test)

def plot_correlation_matrix(df):
    plt.figure(figsize=(12,10))
    corr = df.corr()
    sns.heatmap(corr, cmap = "viridis",linewidths=0.5)
    plt.title("CORRELATION MATRIX HEATMAP")
    plt.savefig(os.path.join(FIGURES_DIR,"correlation matrix heatmap.png"))
    plt.close()
def plot_feature_distributions(df):
    numeric_cols = df.select_dtypes(include=["int64","float64"]).columns
    df[numeric_cols].hist(bins=30,figsize=(12,10))
    plt.title("HISTOGRAM - FEATURE DISTRIBUTION")
    plt.savefig(os.path.join(FIGURES_DIR,"feature distribution.png"))
    plt.close()
def plot_boxplots(df):
    numeric_cols = df.select_dtypes(include=["int64","float64"]).columns
    plt.figure(figsize=(12,10))
    df[numeric_cols].boxplot(rot=30)
    plt.title("BOXPLOT FOR OUTLIER DETECTION")
    plt.savefig(os.path.join(FIGURES_DIR,"boxplots.png"))
    plt.close()
def plot_pairplots(df):
    numeric_cols = df.select_dtypes(include=["int64","float64"]).columns
    selected_cols = numeric_cols[:10]
    sns.pairplot(df[selected_cols])
    plt.savefig(os.path.join(FIGURES_DIR,"pairplots.png"))
    plt.close()
def plot_target_distribution(df,target_column):
    if target_column in df.columns:
        plt.figure(figsize=(6,4))
        sns.histplot(df[target_column], kde=True)
        plt.title("TARGET VARIABLE DISTRIBUTION")
        plt.savefig(os.path.join(FIGURES_DIR,"target distribution.png"))
        plt.close()

# 2.3) Data splitting and Normalisation

def split_and_normalize_data(df,target_column,test_size = 0.2,val_size=0.2):
    x=df.drop(columns=[target_column])
    y=df[target_column]
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=test_size, random_state=42)
    x_train,x_val,y_train,y_val = train_test_split(x_train,y_train,test_size=val_size,random_state=42)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_val = scaler.transform(x_val)
    x_test = scaler.transform(x_test)
    return x_train,x_test,x_val,y_train,y_test,y_val,scaler


if __name__ == "__main__":
    plot_correlation_matrix(df_train)
    plot_feature_distributions(df_train)
    plot_boxplots(df_train)
    plot_pairplots(df_train)
   
    print(df_train.columns.tolist())
    TARGET_COLUMN = "output"
    plot_target_distribution(df_train,TARGET_COLUMN)
    x_train, x_test, x_val, y_train, y_test, y_val, scaler = \
    split_and_normalize_data(df_train, TARGET_COLUMN)




