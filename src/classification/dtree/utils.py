
import pandas as pd

def prepare_WBCD_dataset(data_file):
    headers = ["ID","CT","UCSize","UCShape","MA","SECSize","BN","BC","NN","Mitoses","Diagnosis"]
    data = pd.read_csv(data_file, na_values='?',    
            header=None, index_col=['ID'], names = headers) 
    data = data.reset_index(drop=True)
    data = data.fillna(0)
    X_data = data.iloc[:,:-1]
    y_data = data.iloc[:,-1]
    y_data[y_data==2]=0
    y_data[y_data==4]=1
    return X_data, y_data

