import pandas as pd
import numpy as np

def prepare_features (df: pd.DataFrame, target_col: str = "SalePrice"):
    
    # Sikkerhedsnet - modificerer en kopi, og ikke 
    df = df.copy()

    if target_col not in df.columns:
        raise ValueError(f"Target Column '{target_col}' not found in DataFrame")

    # Kategoriske features behandles
    df["Fence"] = df["Fence"].fillna("NoFence")
    df["PoolQC"] = df["PoolQC"].fillna("NoPool")
    df["FireplaceQu"] = df["FireplaceQu"].fillna("NoFireplace")
    df["MasVnrType"] = df["MasVnrType"].fillna("None")
    df["Alley"] = df["Alley"].fillna("NoAlleyAccess")
    # Garage Domæne
    garage_features = ["GarageType", "GarageQual", "GarageFinish", "GarageCond"]
    for col in garage_features:
        df[col] = df[col].fillna("NoGarage")

    df["GarageYrBlt"] = df["GarageYrBlt"].fillna(0)
    df["HasGarage"] = (df["GarageYrBlt"] > 0).astype(int)
    
    # Numeriske Features
    df["LotFrontage"] = df["LotFrontage"].fillna(0.0)
    df["MasVnrArea"] = df["MasVnrArea"].fillna(0.0)


    X = df.drop(columns=[target_col])
    y_log = np.log1p(df[target_col])

    print("Features er forberedt")

    return X, y_log