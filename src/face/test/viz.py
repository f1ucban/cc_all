import os
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix


matplotlib.use("Agg")
os.makedirs("xtras/charts", exist_ok=True)


def plot_tsne(
    embeds, lbls, outpath="xtras/charts/tsne.png", title="t-SNE of Face Embeddings"
):
    tsne = TSNE(n_components=2, random_state=42)
    embeds_2d = tsne.fit_transform(embeds)
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x=embeds_2d[:, 0],
        y=embeds_2d[:, 1],
        hue=lbls,
        palette="viridis",
        alpha=0.7,
    )
    plt.grid(axis="x", linestyle="--", alpha=0.7)
    plt.title(title)
    plt.savefig(outpath, bbox_inches="tight", dpi=300)
    plt.close()


def plot_roc(tpr, fpr, outpath="xtras/charts/roc_curve.png"):
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {np.trapz(tpr, fpr):.2f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend()
    plt.savefig(outpath, bbox_inches="tight", dpi=300)
    plt.close()


def plot_det_curve(fpr, fnr, outpath="xtras/charts/det_curve.png"):
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, fnr)
    plt.xlabel("False Positive Rate (FAR)")
    plt.ylabel("False Negative Rate (FRR)")
    plt.title("Detection Error Tradeoff (DET) Curve")
    plt.savefig(outpath, bbox_inches="tight", dpi=300)
    plt.close()


def plot_performance_metrics(
    metrics_dict, outpath="xtras/charts/performance_metrics.png"
):
    plt.figure(figsize=(14, 8))
    metrics = pd.Series(metrics_dict)
    colors = sns.color_palette("cividis ", metrics)
    ax = metrics.plot(kind="bar", color=colors, width=0.9)
    plt.title("Face Model Performance Summary", fontsize=16)
    plt.ylabel("Rate/Score", fontsize=14)
    plt.xticks(rotation=45, ha="right", fontsize=10)

    for i, v in enumerate(metrics):
        ax.text(i, v, f"{v:.4f}", ha="center", va="bottom")

    plt.tight_layout()
    plt.savefig(outpath, bbox_inches="tight", dpi=300)
    plt.close()


def plot_distance_metrics(
    intra_dist, inter_dist, outpath="xtras/charts/distance_metrics.png"
):
    plt.figure(figsize=(14, 8))
    metrics = pd.Series({"Intra-class": intra_dist, "Inter-class": inter_dist})

    num_metrics = len(metrics)
    colors = sns.color_palette("magma", num_metrics)
    ax = metrics.plot(kind="barh", color=colors, width=0.9)
    plt.grid(axis="x", linestyle="--", alpha=0.7)
    plt.title("Face Similarity vs Difference", fontsize=16)
    plt.xlabel("Average", fontsize=14)
    plt.yticks(fontsize=10)
    plt.xticks(fontsize=10)

    for i, v in enumerate(metrics):
        ax.text(v, i, f"{v:.4f}", ha="left", va="center")

    plt.tight_layout()
    plt.savefig(outpath, bbox_inches="tight", dpi=300)
    plt.close()


def plot_per_subject_accuracy(
    correct, total, outpath="xtras/charts/per_subject_accuracy.png"
):
    accuracies = {
        f"S{(label+1):03d}": correct[label] / total[label] for label in correct.keys()
    }

    sorted_subjects = sorted(accuracies.keys())
    # sorted_accuracies = [accuracies[subject] for subject in sorted_subjects]

    plt.figure(figsize=(18, 12))
    metrics = pd.Series(accuracies)
    num_subjects = len(sorted_subjects)
    color_palette = (
        sns.color_palette("tab20", 20)
        + sns.color_palette("tab20b", 20)
        + sns.color_palette("tab20c", 20)
    )
    colors_to_use = color_palette[:num_subjects]
    ax = metrics.plot(kind="bar", color=colors_to_use)
    plt.title("FR Per-Subject Accuracy", fontsize=16)
    plt.ylabel("Accuracy", fontsize=14, rotation="vertical")
    plt.xlabel("Subject idx", fontsize=14)
    plt.xticks(rotation=60, ha="right", fontsize=10)

    for i, v in enumerate(metrics):
        ax.text(i, v, f"{v:.4f}", ha="center", va="bottom", fontsize=8, rotation=90)

    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(outpath, bbox_inches="tight", dpi=300)
    plt.close()


def plot_confusion_matrix(y_true, y_pred, outpath="xtras/charts/confusion_matrix.png"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(15, 12))
    sns.heatmap(cm, annot=False, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix", fontsize=16)
    plt.xlabel("Predicted", fontsize=14)
    plt.ylabel("True", fontsize=14)
    plt.savefig(outpath, bbox_inches="tight", dpi=300)
    plt.close()


def plot_classification_report(
    report, outpath="xtras/charts/classification_report.png"
):
    report_data = []
    for line in report.split("\n")[2:-3]:
        if line.strip():
            values = line.split()
            if len(values) >= 5:
                report_data.append(
                    {
                        "Class": values[0],
                        "Precision": float(values[1]),
                        "Recall": float(values[2]),
                        "F1-score": float(values[3]),
                        "Support": int(values[4]),
                    }
                )

    df = pd.DataFrame(report_data)
    df = df.set_index("Class")

    plt.figure(figsize=(25, 8))
    ax = df[["Precision", "Recall", "F1-score"]].plot(kind="bar")
    plt.title("Classification Report", fontsize=16)
    plt.ylabel("Score", fontsize=14)
    plt.xlabel("Class", fontsize=14)
    plt.xticks(rotation=60, ha="right", fontsize=8)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(outpath, bbox_inches="tight", dpi=300)
    plt.close()


def plot_rate_metrics(metrics_dict, outpath="xtras/charts/rate_metrics.png"):
    plt.figure(figsize=(14, 8))
    metrics = pd.Series(metrics_dict)
    num_metrics = len(metrics)
    colors = sns.color_palette("magma", num_metrics)
    ax = metrics.plot(kind="barh", color=colors, width=0.9)
    plt.grid(axis="x", linestyle="--", alpha=0.7)
    plt.title("FR Acceptance and Rejection Rates", fontsize=16)
    plt.xlabel("Rate", fontsize=14)
    plt.yticks(fontsize=10)
    plt.xticks(fontsize=10)

    for i, v in enumerate(metrics):
        ax.text(v, i, f"{v:.4f}", ha="left", va="center")

    plt.tight_layout()
    plt.savefig(outpath, bbox_inches="tight", dpi=300)
    plt.close()


def plot_macro_overall_metrics(
    metrics_dict, outpath="xtras/charts/macro_overall_metrics.png"
):
    plt.figure(figsize=(14, 8))
    metrics = pd.Series(metrics_dict)
    num_metrics = len(metrics)
    colors = sns.color_palette("viridis", num_metrics)
    ax = metrics.plot(kind="barh", color=colors, width=0.9)
    plt.grid(axis="x", linestyle="--", alpha=0.7)
    plt.title("FR Perfomance Summary", fontsize=16)
    plt.xlabel("Score", fontsize=14)
    plt.yticks(fontsize=10)
    plt.xticks(fontsize=10)

    for i, v in enumerate(metrics):
        ax.text(v, i, f"{v:.4f}", ha="left", va="center")

    plt.tight_layout()
    plt.savefig(outpath, bbox_inches="tight", dpi=300)
    plt.close()
