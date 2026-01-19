import os
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import kfold
from torch.optim import adam
import torch.nn.functional as f
from lifelines.utils import concordance_index
from sklearn.cluster import kmeans
from lifelines.statistics import logrank_test
from lifelines import kaplanmeierfitter
import matplotlib.pyplot as plt
from pil import image
from torchvision import models, transforms
from torch import nn
import pickle


model_name = "gcn"
data_type = "mihc_vars"

base_dir = "d:/gc_project"

output_dir = f"{base_dir}/output/survival_results/{model_name}/{data_type}"
os.makedirs(output_dir, exist_ok=True)

data_paths = {
    "imc_all_vars": f"{base_dir}/data/imc_cell_interaction.csv",
    "imc_filter_vars": f"{base_dir}/data/imc_cell_interaction.csv",
    "mihc_vars": f"{base_dir}/data/mihc_survival_data.csv",
}

image_folder_dict = {
    "imc_all_vars": f"{base_dir}/images/imc",
    "imc_filter_vars": f"{base_dir}/images/imc",
    "mihc_vars": f"{base_dir}/images/mihc",
}

celltype_csv_paths = {
    "imc_all_vars": f"{base_dir}/config/cell_types_imc_all.csv",
    "imc_filter_vars": f"{base_dir}/config/cell_types_imc_filter.csv",
    "mihc_vars": f"{base_dir}/config/cell_types_mihc.csv",
}


def load_celltypes_from_csv(csv_path, column_name="cell_type"):
    df_cell = pd.read_csv(csv_path)
    if column_name not in df_cell.columns:
        raise ValueError("cell type csv must contain cell_type column")
    return df_cell[column_name].dropna().astype(str).tolist()


df = pd.read_csv(data_paths[data_type], index_col=0)
celltypes = load_celltypes_from_csv(celltype_csv_paths[data_type])
image_folder = image_folder_dict[data_type]


def build_adjacency_matrix(df, celltypes):
    adj_dict = {}

    for _, row in df.iterrows():
        roi_id = row["roi_id"]
        a = np.zeros((len(celltypes), len(celltypes)), dtype=float)

        for i in range(len(celltypes)):
            for j in range(i + 1, len(celltypes)):
                c1 = f"{celltypes[i]}_{celltypes[j]}"
                c2 = f"{celltypes[j]}_{celltypes[i]}"
                value = row[c1] if c1 in row else row[c2] if c2 in row else 0.0
                a[i, j] = value
                a[j, i] = value

        np.fill_diagonal(a, 1.0)
        adj_dict[roi_id] = a

    return adj_dict


adjacency_matrices = build_adjacency_matrix(df, celltypes)


resnet = models.resnet50(pretrained=True)
resnet = nn.sequential(*list(resnet.children())[:-1])
resnet.eval()

preprocess = transforms.compose([
    transforms.resize(256),
    transforms.centercrop(224),
    transforms.totensor(),
    transforms.normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


def extract_features(image_path):
    img = image.open(image_path).convert("rgb")
    tensor = preprocess(img).unsqueeze(0)
    with torch.no_grad():
        feat = resnet(tensor)
    return feat.flatten().numpy()



feature_dir = f"{output_dir}/features"
os.makedirs(feature_dir, exist_ok=True)
feature_file = f"{feature_dir}/features.pkl"

if os.path.exists(feature_file):
    with open(feature_file, "rb") as f:
        features_dict = pickle.load(f)
else:
    features_dict = {}
    for _, row in df.iterrows():
        roi_id = row["roi_id"]
        feats = []

        for ct in celltypes:
            image_path = f"{image_folder}/{roi_id}/{ct}.csv.png"
            if os.path.exists(image_path):
                feats.append(extract_features(image_path))
            else:
                feats.append(np.zeros(2048))

        features_dict[roi_id] = np.stack(feats)

    with open(feature_file, "wb") as f:
        pickle.dump(features_dict, f)


def plot_survival(clusters, events, times, labels, colors):
    idx = clusters == 0
    t1, t2 = times[idx], times[~idx]
    e1, e2 = events[idx], events[~idx]

    result = logrank_test(t1, t2, e1, e2)

    kmf = kaplanmeierfitter()
    fig, ax = plt.subplots()

    kmf.fit(t1, e1, label=labels[0])
    kmf.plot(ax=ax, ci_show=false, color=colors[0])

    kmf.fit(t2, e2, label=labels[1])
    kmf.plot(ax=ax, ci_show=false, color=colors[1])

    ax.set_title("survival curve")
    ax.set_xlabel("time")
    ax.set_ylabel("survival probability")
    ax.text(0.05, 0.05, f"p-value = {result.p_value:.4e}", transform=ax.transAxes)

    return fig


class gcn(nn.module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.w1 = nn.parameter(torch.randn(input_dim, hidden_dim))
        self.b1 = nn.parameter(torch.zeros(hidden_dim))
        self.w2 = nn.parameter(torch.randn(hidden_dim, output_dim))
        self.b2 = nn.parameter(torch.zeros(output_dim))
        self.fc = nn.linear(output_dim, 1)

    def forward(self, a, x):
        h1 = f.relu(a @ x @ self.w1 + self.b1)
        h2 = a @ h1 @ self.w2 + self.b2
        graph_feat = h2.mean(dim=0)
        node_feat = h2.view(-1)
        out = self.fc(graph_feat)
        return out, node_feat


kf = kfold(n_splits=5, shuffle=true, random_state=42)
results = []

roi_ids = list(features_dict.keys())

for fold, (train_idx, test_idx) in enumerate(kf.split(roi_ids)):
    train_ids = [roi_ids[i] for i in train_idx]
    test_ids = [roi_ids[i] for i in test_idx]

    model = gcn(features_dict[train_ids[0]].shape[1], 64, 64)
    optimizer = adam(model.parameters(), lr=1e-3)

    model.train()
    for _ in range(100):
        optimizer.zero_grad()
        loss_sum = 0.0
        for rid in train_ids:
            x = torch.tensor(features_dict[rid], dtype=torch.float32)
            a = torch.tensor(adjacency_matrices[rid], dtype=torch.float32)
            y = torch.tensor(df.loc[df["roi_id"] == rid, "survival_status"].values[0], dtype=torch.float32)
            out, _ = model(a, x)
            loss_sum += f.binary_cross_entropy_with_logits(out.squeeze(), y)
        loss_sum.backward()
        optimizer.step()

    model.eval()
    preds, events, times, feats = [], [], [], []

    with torch.no_grad():
        for rid in test_ids:
            x = torch.tensor(features_dict[rid], dtype=torch.float32)
            a = torch.tensor(adjacency_matrices[rid], dtype=torch.float32)
            out, feat = model(a, x)
            preds.append(out.item())
            feats.append(feat.numpy())
            events.append(df.loc[df["roi_id"] == rid, "survival_status"].values[0])
            times.append(df.loc[df["roi_id"] == rid, "survival_time"].values[0])

    feats = np.vstack(feats)
    clusters = kmeans(n_clusters=2, random_state=42).fit_predict(feats)

    mean_times = [np.mean(np.array(times)[clusters == i]) for i in [0, 1]]
    risk_labels = ["high risk", "low risk"] if mean_times[0] < mean_times[1] else ["low risk", "high risk"]
    colors = ["red", "blue"]

    fig = plot_survival(clusters, np.array(events), np.array(times), risk_labels, colors)
    fig.savefig(f"{output_dir}/survival_fold_{fold+1}.png")

    c_index = concordance_index(times, preds, events)
    results.append({"fold": fold + 1, "c_index": c_index})

pd.dataframe(results).to_csv(f"{output_dir}/cv_results.csv", index=false)
