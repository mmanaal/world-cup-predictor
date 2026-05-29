import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

CLUSTER_FEATURES = ["pace", "shooting", "passing", "dribbling", "defending", "physical"]


def cluster_players(df: pd.DataFrame, n_clusters: int = 4) -> pd.DataFrame:
    available = [c for c in CLUSTER_FEATURES if c in df.columns]
    df = df.copy()
    if not available:
        df["cluster"] = 0
        return df
    scaler = StandardScaler()
    X = scaler.fit_transform(df[available].fillna(0.0))
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    df["cluster"] = km.fit_predict(X).astype(str)
    return df
