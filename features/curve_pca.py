from __future__ import annotations
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

def curve_pca_k3(curve_df: pd.DataFrame) -> pd.DataFrame:
    X = curve_df.copy().sort_index()
    X = (X - X.mean(axis=1).values.reshape(-1,1)) / (X.std(axis=1).replace(0,1).values.reshape(-1,1))
    X = X.fillna(0.0)
    pca = PCA(n_components=3, random_state=42)
    F = pca.fit_transform(X.values)
    return pd.DataFrame(F, index=curve_df.index, columns=["curve_level","curve_slope","curve_curv"])
