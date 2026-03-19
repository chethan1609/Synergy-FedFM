import os
import glob
import pandas as pd
import numpy as np

def read_and_clean_csvs(data_dir: str) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    if not files:
        raise FileNotFoundError(f"No CSVs found in {data_dir}")
    chunks = []
    for f in files:
        df = pd.read_csv(f, low_memory=False)
        df.columns = df.columns.str.strip()
        if "Label" in df.columns:
            df["Label"] = df["Label"].astype(str).str.strip()
            df["Label"] = df["Label"].replace({
                "Web Attack \u2013 Brute Force":  "Web Attack",
                "Web Attack \u2013 XSS":          "Web Attack",
                "Web Attack \u2013 Sql Injection":"Web Attack",
                "DoS Hulk":     "DoS", "DoS GoldenEye": "DoS",
                "DoS Slowloris":"DoS", "DoS Slowhttptest":"DoS",
            })
            df["Label"] = df["Label"].str.title()
        chunks.append(df)
    df = pd.concat(chunks, ignore_index=True)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    if "Timestamp" in df.columns:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
        df = df.sort_values("Timestamp").reset_index(drop=True)
        df.drop(columns=["Timestamp"], inplace=True)
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = df[col].astype(np.float32)
    return df

def save_parquet(df: pd.DataFrame, out_path: str):
    df.to_parquet(out_path, engine="pyarrow", index=False)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--out", default="./cicids2017_clean.parquet")
    args = parser.parse_args()
    df = read_and_clean_csvs(args.data_dir)
    save_parquet(df, args.out)
    print(f"Saved cleaned parquet to {args.out} shape={df.shape}")
