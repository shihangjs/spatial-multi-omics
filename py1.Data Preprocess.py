import os
import numpy as np
import pandas as pd


# convert ssim matrix to flattened vector per sample

base_dir = "d:/gc_project"

data_file = f"{base_dir}/data/survival_data.csv"
ssim_folder = f"{base_dir}/data/ssim_matrices"

df = pd.read_csv(data_file, index_col=0)

final_data = []

row_names = None
column_names = None

for roi_id in df["roi_id"]:
    sample_csv = os.path.join(ssim_folder, f"{roi_id}.csv")

    if not os.path.exists(sample_csv):
        continue

    sample_data = pd.read_csv(sample_csv, index_col=0)
    sample_data = sample_data.fillna(0.0)

    if row_names is None:
        row_names = sample_data.index.tolist()
        column_names = sample_data.columns.tolist()

    flattened_data = sample_data.values.flatten()

    survival_status = df.loc[df["roi_id"] == roi_id, "survival_status"].values[0]
    survival_time = df.loc[df["roi_id"] == roi_id, "survival_time"].values[0]

    sample_row = [roi_id, survival_status, survival_time] + flattened_data.tolist()
    final_data.append(sample_row)

if row_names is None or column_names is None:
    raise ValueError("no valid ssim files found")

feature_names = [f"{r}_{c}" for r in row_names for c in column_names]
columns = ["roi_id", "survival_status", "survival_time"] + feature_names

final_df = pd.DataFrame(final_data, columns=columns)

output_file = f"{base_dir}/data/flattened_ssim_features.csv"
final_df.to_csv(output_file, index=False)