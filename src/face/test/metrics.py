import torch
import numpy as np
import torch.nn.functional as F
from scipy.spatial.distance import cdist, pdist
from face.test.viz import plot_roc, plot_det_curve
from sklearn.metrics import (
    roc_curve,
    det_curve,
    classification_report,
    confusion_matrix,
)


def calc_roc_threshold(y_true, y_score):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    if len(thresholds) == 0:
        return fpr, tpr, thresholds, 0.5

    j_scores = tpr - fpr
    opt_idx = np.argmax(j_scores)
    opt_thresh = thresholds[opt_idx] if opt_idx < len(thresholds) else 0.5
    plot_roc(fpr, tpr)
    return fpr, tpr, thresholds, opt_thresh


def calc_det_curve(y_true, y_score):
    fpr, fnr, thresholds = det_curve(y_true, y_score)
    plot_det_curve(fpr, fnr)


def calc_dist_stats(embeds, idx):
    labels = np.unique(idx[idx != -1])
    intra_dist = []
    inter_dist = []

    for lbl in labels:
        class_samples = embeds[idx == lbl]
        other_samples = embeds[(idx != lbl) & (idx != -1)]

        if len(class_samples) > 1:
            intra_dist.extend(pdist(class_samples, "cosine"))

        if len(other_samples) > 0:
            inter_dist.extend(cdist(class_samples, other_samples, "cosine").min(axis=1))

    return np.mean(intra_dist) if intra_dist else 0.0, (
        np.mean(inter_dist) if inter_dist else 0.0
    )


def eval_recog(known, idx, unknown, cls_embeds, opt_thresh):
    correct = {lbl: 0 for lbl in np.unique(idx)}
    total = {lbl: 0 for lbl in np.unique(idx)}
    preds = []

    for i in range(len(known)):
        true_lbl = idx[i]
        similarities = [
            F.cosine_similarity(
                torch.tensor(known[i]).unsqueeze(0),
                torch.tensor(class_embed).unsqueeze(0),
            ).item()
            for class_embed in cls_embeds.values()
        ]
        pred_lbl = (
            list(cls_embeds.keys())[np.argmax(similarities)]
            if max(similarities) >= opt_thresh
            else -1
        )
        preds.append(pred_lbl)
        total[true_lbl] += 1
        if pred_lbl == true_lbl:
            correct[true_lbl] += 1

    for embed in unknown:
        similarities = [
            F.cosine_similarity(
                torch.tensor(embed).unsqueeze(0), torch.tensor(class_embed).unsqueeze(0)
            ).item()
            for class_embed in cls_embeds.values()
        ]
        preds.append(
            -1
            if max(similarities) < opt_thresh
            else list(cls_embeds.keys())[np.argmax(similarities)]
        )

    return correct, total, preds


def calc_metrics(known_labels, unknown, preds):
    n_known = len(known_labels)
    n_unknown = len(unknown)

    false_accepts = sum(1 for p in preds[n_known:] if p != -1)
    false_rejects = sum(
        1 for p, true in zip(preds[:n_known], known_labels) if p != true
    )

    FAR = false_accepts / n_unknown if n_unknown > 0 else 0.0
    FRR = false_rejects / n_known if n_known > 0 else 0.0
    TAR = 1 - FRR

    return FAR, FRR, TAR


def clf_report(idx, preds, idxs):
    target_names = [f"s{label+1:03d}" for label in idxs] + ["unknown"]
    return classification_report(
        np.concatenate([idx, [-1] * (len(preds) - len(idx))]),
        preds,
        target_names=target_names,
        zero_division=0,
    )


def conf_matrix(idx, preds):
    return confusion_matrix(
        np.concatenate([idx, [-1] * (len(preds) - len(idx))]), preds
    )


def calc_macro_overall_metrics(y_true, y_pred, idxs):
    report = classification_report(
        y_true,
        y_pred,
        target_names=[f"s{label+1:03d}" for label in idxs] + ["unknown"],
        zero_division=0,
        output_dict=True,
    )

    macro_precision = report["macro avg"]["precision"]
    macro_recall = report["macro avg"]["recall"]
    macro_f1 = report["macro avg"]["f1-score"]
    overall_accuracy = report["accuracy"]

    return macro_precision, macro_recall, macro_f1, overall_accuracy
