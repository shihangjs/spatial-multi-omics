import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import KFold
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from lifelines.statistics import logrank_test
from lifelines import KaplanMeierFitter
from lifelines.utils import concordance_index
import matplotlib.pyplot as plt
import logging


class MLP(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.Sigmoid(),
            nn.Dropout(0.2),
            nn.Linear(32, 64),
            nn.Sigmoid(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.Sigmoid(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.Sigmoid(),
            nn.Dropout(0.2),
        )
        self.classifier = nn.Linear(16, 1)

    def forward(self, x):
        code = self.encoder(x)
        lbl_pred = self.classifier(code)
        return None, code, lbl_pred


def cox_log_rank_cluster(hazards, labels, survtime_all, risk_labels, colors):
    idx = hazards == 0
    t1, t2 = survtime_all[idx], survtime_all[~idx]
    e1, e2 = labels[idx], labels[~idx]

    results = logrank_test(t1, t2, event_observed_A=e1, event_observed_B=e2)
    pvalue_pred = results.p_value

    kmf = KaplanMeierFitter()
    fig = plt.figure()
    ax = plt.subplot(111)

    kmf.fit(t1, event_observed=e1, label=risk_labels[0])
    kmf.plot(ax=ax, show_censors=True, ci_show=False, color=colors[0])

    kmf.fit(t2, event_observed=e2, label=risk_labels[-1])
    kmf.plot(ax=ax, show_censors=True, ci_show=False, color=colors[-1])

    ax.text(0, ax.get_ylim()[0] + 0.05, s=f"p-value = {pvalue_pred:.4e}", fontsize=14)
    ax.grid(color="grey", linestyle="--", linewidth=0.5)
    ax.set_xlabel("time")
    ax.set_ylabel("survival probability")
    ax.set_title("survival plot", fontsize=14)

    return fig, pvalue_pred


def load_feature_list(csv_path: str, column_name: str = "feature") -> list:
    df_feat = pd.read_csv(csv_path)
    if column_name not in df_feat.columns:
        raise ValueError(f"feature csv must contain column '{column_name}'")
    feats = df_feat[column_name].dropna().astype(str).tolist()
    if len(feats) == 0:
        raise ValueError(f"no features found in {csv_path}")
    return feats


def prepare_data(
    df: pd.DataFrame,
    data_type: str = "imc_all_vars",
    standardize: bool = False,
    normalize: bool = False,
    use_vars: str = "all",
    feature_csv_paths: dict | None = None,
):
    required_cols = {"roi_id", "survival_status", "survival_time"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(list(missing))}")

    if feature_csv_paths is None:
        feature_csv_paths = {}

    drop_cols = ["roi_id", "survival_status", "survival_time"]

    if data_type == "imc_all_vars":
        all_cols = df.columns.tolist()
        all_features = [c for c in all_cols if c not in drop_cols]

        gene_expr = load_feature_list(feature_csv_paths["gene_expr"]) if "gene_expr" in feature_csv_paths else []
        cell_interact = load_feature_list(feature_csv_paths["cell_interact"]) if "cell_interact" in feature_csv_paths else []

        if use_vars == "all":
            vars_used = all_features
        elif use_vars == "gene_expr":
            vars_used = gene_expr
        elif use_vars == "cell_interact":
            vars_used = cell_interact
        else:
            raise ValueError("use_vars must be one of: all, gene_expr, cell_interact")

    elif data_type in ["imc_filter_vars", "mihc_vars"]:
        marker = load_feature_list(feature_csv_paths["marker"]) if "marker" in feature_csv_paths else []
        cell_interact = load_feature_list(feature_csv_paths["cell_interact"]) if "cell_interact" in feature_csv_paths else []

        if use_vars == "all":
            vars_used = marker + cell_interact
        elif use_vars == "gene_expr":
            vars_used = []
        elif use_vars == "cell_interact":
            vars_used = cell_interact
        else:
            raise ValueError("use_vars must be one of: all, gene_expr, cell_interact")

    else:
        raise ValueError("data_type must be one of: imc_all_vars, imc_filter_vars, mihc_vars")

    vars_used = [v for v in vars_used if v in df.columns]
    if len(vars_used) == 0:
        raise ValueError("no valid features found after filtering by df columns")

    x = df[vars_used].values
    e = df["survival_status"].values.astype(int)
    t = df["survival_time"].values.astype(float)

    if standardize:
        scaler = StandardScaler()
        x = scaler.fit_transform(x)

    if normalize:
        scaler = MinMaxScaler()
        x = scaler.fit_transform(x)

    return {"x": x, "e": e, "t": t}


def train_model(datasets, num_epochs: int = 500, lr: float = 1e-3, device: str = "cpu"):
    model = MLP(datasets["train"]["x"].shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_cindex = -1.0
    best_state = None

    for _ in range(num_epochs):
        model.train()
        inputs = torch.FloatTensor(datasets["train"]["x"]).to(device)
        lbl_pred = model(inputs)[2]

        theta = lbl_pred.squeeze()
        survtime = torch.FloatTensor(datasets["train"]["t"]).to(device)
        r_matrix = (survtime.unsqueeze(1) >= survtime).float()

        loss = -torch.mean(
            (theta - torch.log(torch.sum(torch.exp(theta) * r_matrix, dim=1) + 1e-12))
            * torch.FloatTensor(datasets["train"]["e"]).to(device)
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            test_inputs = torch.FloatTensor(datasets["test"]["x"]).to(device)
            test_pred = model(test_inputs)[2].squeeze().cpu().numpy()
            cindex = concordance_index(datasets["test"]["t"], -test_pred, datasets["test"]["e"])
            if cindex > best_cindex:
                best_cindex = cindex
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is None:
        best_state = model.state_dict()

    return best_state, best_cindex, model


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    data_type = "imc_all_vars"
    use_vars = "cell_interact"

    base_dir = "d:/gc_project"

    data_paths = {
        "imc_all_vars": f"{base_dir}/data/survival_data_imc_all.csv",
        "imc_filter_vars": f"{base_dir}/data/survival_data_imc_filter.csv",
        "mihc_vars": f"{base_dir}/data/survival_data_mihc.csv",
    }

    feature_csv_paths_by_type = {
        "imc_all_vars": {
            "gene_expr": f"{base_dir}/config/features_gene_expr.csv",
            "cell_interact": f"{base_dir}/config/features_cell_interact.csv",
        },
        "imc_filter_vars": {
            "marker": f"{base_dir}/config/features_marker.csv",
            "cell_interact": f"{base_dir}/config/features_cell_interact.csv",
        },
        "mihc_vars": {
            "marker": f"{base_dir}/config/features_marker.csv",
            "cell_interact": f"{base_dir}/config/features_cell_interact.csv",
        },
    }

    df = pd.read_csv(data_paths[data_type], index_col=0)

    standardize = False
    normalize = False

    full_data = prepare_data(
        df,
        data_type=data_type,
        standardize=standardize,
        normalize=normalize,
        use_vars=use_vars,
        feature_csv_paths=feature_csv_paths_by_type[data_type],
    )

    save_dir = f"{base_dir}/output/survival_analysis_results/{data_type}_use_vars_{use_vars}"
    os.makedirs(save_dir, exist_ok=True)

    seed = 42
    logging.info(f"running with seed: {seed}")
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = "cpu"
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)

    results_df = pd.DataFrame(columns=["seed", "fold", "c_index", "p_value"])
    csv_file_path = f"{save_dir}/survival_analysis_results.csv"

    for fold, (train_idx, test_idx) in enumerate(kf.split(df)):
        logging.info(f"fold {fold + 1}/5")

        datasets = {
            "train": {k: v[train_idx] for k, v in full_data.items()},
            "test": {k: v[test_idx] for k, v in full_data.items()},
        }

        best_state, best_cindex, model = train_model(
            datasets,
            num_epochs=1000,
            lr=1e-3,
            device=device,
        )

        model.load_state_dict(best_state)
        model.eval()

        with torch.no_grad():
            train_inputs = torch.FloatTensor(datasets["train"]["x"]).to(device)
            train_code = model(train_inputs)[1].cpu().numpy()

        cluster_labels = KMeans(n_clusters=2, random_state=seed).fit_predict(train_code)

        avg_survival_time = []
        for cl in np.unique(cluster_labels):
            cl_idx = np.where(cluster_labels == cl)[0]
            cl_times = datasets["train"]["t"][cl_idx]
            avg_survival_time.append(np.mean(cl_times))

        risk_labels = [
            "high risk" if avg_survival_time[i] == min(avg_survival_time) else "low risk"
            for i in range(len(avg_survival_time))
        ]
        colors = ["red" if label == "high risk" else "blue" for label in risk_labels]

        fig, pvalue_pred = cox_log_rank_cluster(
            cluster_labels,
            datasets["train"]["e"],
            datasets["train"]["t"],
            risk_labels,
            colors,
        )

        temp_df = pd.DataFrame(
            {
                "seed": [seed],
                "fold": [fold + 1],
                "c_index": [best_cindex],
                "p_value": [pvalue_pred],
            }
        )

        if os.path.exists(csv_file_path):
            temp_df.to_csv(csv_file_path, mode="a", header=False, index=False)
        else:
            temp_df.to_csv(csv_file_path, mode="w", header=True, index=False)

        fig.savefig(f"{save_dir}/survival_curve_fold_{fold + 1}_seed_{seed}.png")

    results_df.to_csv(csv_file_path, index=False)
