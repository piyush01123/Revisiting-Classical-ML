import pandas as pd
import numpy as np

def prepare_diamonds_dataset(csv_file, fewer_columns=False):
    headers = ["carat",	"cut","color","clarity","depth","table","price","x","y","z"]
    data = pd.read_csv(csv_file, na_values='?', header=None,  names = headers)
    data = data.reset_index(drop=True)
    data = data.iloc[1:]

    data.carat = data.carat.astype(float)
    data.depth = data.depth.astype(float)
    data.table = data.table.astype(float)
    data.price = data.price.astype(float)
    data.x = data.x.astype(float)
    data.y = data.y.astype(float)
    data.z = data.z.astype(float)

    data_n = pd.get_dummies(data)

    if not fewer_columns:
        # Convert to numpy and split
        X_data_pd = data_n.copy()
        del X_data_pd['price']
        print(X_data_pd.columns)

        print("Columns in dataset:")
        for col in X_data_pd.columns:
            print("Column: {}, #NA: {}, dtype: {}".format(col, X_data_pd[col].isnull().sum(), X_data_pd[col].dtype))

        X_data = X_data_pd.to_numpy()
        Y_data = data_n["price"].to_numpy()
        return X_data, Y_data

    else:
        corrs = data_n.corr()['price']
        sorted_corr_indices = corrs.argsort().tolist()[::-1]
        print("Correlation of predictor variables with Y:")
        for k in sorted_corr_indices:
            print(data_n.columns[k], corrs[k])
        print("We see that carat, x, y, z are the best features with correlation above .86 and the rest are below .13",
              "So we choose carat, x, y, z")
        X_data_pd = data_n.copy()
        del X_data_pd['price']

        X_data_pd = X_data_pd[['carat','x','y','z']]
        print(X_data_pd.columns)

        X_data = X_data_pd.to_numpy()
        Y_data = data_n["price"].to_numpy()
        return X_data, Y_data
