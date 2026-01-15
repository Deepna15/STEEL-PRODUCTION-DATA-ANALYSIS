import pandas as pd
import os
def load_steel_data():
    train_path = os.path.join("data", "normalized_train_data.csv")
    test_path = os.path.join("data", "normalized_test_data.csv") 
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    return df_train, df_test
