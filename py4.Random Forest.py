import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from lifelines.utils import concordance_index
from lifelines.statistics import logrank_test
from lifelines import KaplanMeierFitter


def prepare_data(df, data_type="imc_all_vars"):
    search_patterns = ["hladrhimφ", "cd11chimφ"]

    if data_type == "imc_all_vars":
        x = df.drop(["roi_id", "survival_status", "survival_time"], axis=1).values

    elif data_type == "imc_filter_vars":
        epithelial_vars = [col for col in df.columns if "epithelial" in col.lower()]
        epithelial_vars = [col for col in epithelial_vars if re.search("|".join(search_patterns), col.lower())]

        fibroblast_vars = [col for col in df.columns if "fibroblast" in col.lower()]
        fibroblast_vars = [col for col in fibroblast_vars if re.search("|".join(search_patterns), col.lower())]

        custom_vars = ["cd11chimφ_hladrhimφ", "hladrhimφ_cd11chimφ"]
        general_vars = ["collagen1", "pdl1", "pan.keratin", "cd68", "cd45", "cd11c", "hladr"]

        var_names = epithelial_vars + fibroblast_vars + custom_vars + general_vars
        var_names = [v for v in var_names if v in df.columns]
        if len(var_names) == 0:
            raise ValueError("no valid columns found for imc_filter_vars")
        x = df[var_names].values

    elif data_type == "mihc_vars":
        x = df.drop(["roi_id", "file_id", "survival_status", "survival_time"], axis=1).values

    else:
        raise ValueError("data_type must be one of: imc_all_vars, imc_filter_vars, mihc_vars")

    return x


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


def run_rf_survival_cv(df, data_type, output_dir, n_splits=5, seed=42):
    x = prepare_data(df, data_type=data_type)
    t = np.array(df["survival_time"], dtype=float)
    e = np.array(df["survival_status"], dtype=int)

    rf_model = RandomForestClassifier(n_estimators=100, random_state=seed)
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    c_index_scores = []
    results_rows = []

    fold = 0
    for train_idx, test_idx in kf.split(x, e):
        x_train, x_test = x[train_idx], x[test_idx]
        e_train, e_test = e[train_idx], e[test_idx]
        t_train, t_test = t[train_idx], t[test_idx]

        rf_model.fit(x_train, e_train)

        y_pred_prob = rf_model.predict_proba(x_test)[:, 1]
        thr = float(np.median(y_pred_prob))
        cluster_labels = (y_pred_prob > thr).astype(int)

        risk_labels = ["low risk", "high risk"]
        colors = ["blue", "red"]

        fig, pvalue_pred = cox_log_rank_cluster(
            cluster_labels.astype(int),
            np.array(e_test, dtype=int),
            np.array(t_test, dtype=float),
            risk_labels,
            colors,
        )
        fig.savefig(f"{output_dir}/survival_curve_fold_{fold + 1}.png")

        c_index = concordance_index(t_test, y_pred_prob, e_test)
        c_index_scores.append(float(c_index))

        results_rows.append(
            {
                "seed": seed,
                "fold": fold + 1,
                "c_index": float(c_index),
                "p_value": float(pvalue_pred),
            }
        )

        fold += 1

    results_df = pd.DataFrame(results_rows)
    results_df.to_csv(f"{output_dir}/cv_results.csv", index=False)

    return c_index_scores, results_df


if __name__ == "__main__":
    model_name = "rf"
    data_type = "imc_all_vars"

    base_dir = "d:/gc_project"
    output_dir = f"{base_dir}/output/other_models_survival/{model_name}/{data_type}"
    os.makedirs(output_dir, exist_ok=True)

    data_paths = {
        "imc_all_vars": f"{base_dir}/data/survival_data_imc_all.csv",
        "imc_filter_vars": f"{base_dir}/data/survival_data_imc_filter.csv",
        "mihc_vars": f"{base_dir}/data/survival_data_mihc.csv",
    }

    df = pd.read_csv(data_paths[data_type], index_col=0)

    c_index_scores, results_df = run_rf_survival_cv(
        df=df,
        data_type=data_type,
        output_dir=output_dir,
        n_splits=5,
        seed=42,
    )
