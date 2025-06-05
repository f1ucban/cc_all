import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F

from face.configs.base import cfg
from face.configs.utils import init_dm
from src.utils.logger import setup_logger
from dataloaders.face_loading import dataloaders
from face.test.viz import (
    plot_tsne,
    plot_distance_metrics,
    plot_per_subject_accuracy,
    plot_confusion_matrix,
    plot_classification_report,
    plot_rate_metrics,
    plot_macro_overall_metrics,
)
from face.test.metrics import (
    calc_roc_threshold,
    calc_det_curve,
    calc_dist_stats,
    eval_recog,
    calc_metrics,
    clf_report,
    conf_matrix,
    calc_macro_overall_metrics,
)

logger = setup_logger("__openset__")
val_dl = dataloaders()["val"]
device, model = init_dm()
finetuned = torch.load(cfg.arcIres50, map_location=device, weights_only=False)
model.load_state_dict(finetuned["model_state_dict"])
model.eval()


def main():
    embeds, labels = [], []

    with torch.no_grad():
        for faces, lbls in tqdm(val_dl, desc="Extracting Embeddings"):
            embeds.extend(model(faces.to(device)).cpu().numpy())
            labels.extend(lbls.cpu().numpy())

    embeds, labels = np.array(embeds), np.array(labels)
    mask = labels != -1
    known, known_lbls = embeds[mask], labels[mask]
    unknown = embeds[~mask]
    idxs = np.unique(known_lbls)
    cls_embeds = {lbl: np.mean(known[known_lbls == lbl], axis=0) for lbl in idxs}

    known_unknown = np.concatenate([known, unknown])
    all_cls_embeds = np.array(list(cls_embeds.values()))

    t_embeds = torch.tensor(known_unknown)
    t_cls_embeds = torch.tensor(all_cls_embeds)

    similarities = F.cosine_similarity(
        t_embeds.unsqueeze(1), t_cls_embeds.unsqueeze(0), dim=2
    )
    y_score = similarities.max(dim=1)[0].numpy()
    y_true = np.concatenate([np.ones(len(known)), np.zeros(len(unknown))])

    _, _, _, opt_thresh = calc_roc_threshold(y_true, y_score)
    calc_det_curve(y_true, y_score)
    correct, total, preds = eval_recog(
        known, known_lbls, unknown, cls_embeds, opt_thresh
    )
    FAR, FRR, TAR = calc_metrics(known_lbls, unknown, preds)
    URR = preds[len(known_lbls) :].count(-1) / len(unknown) if len(unknown) > 0 else 0.0

    try:
        intra_dist, inter_dist = calc_dist_stats(
            np.concatenate([known, unknown]),
            np.concatenate([known_lbls, [-1] * len(unknown)]),
        )
    except Exception as e:
        logger.info(f"Error calc dist metrics: {e}")
        intra_dist, inter_dist = 0.0, 0.0

    clfr = clf_report(known_lbls, preds, idxs)
    conf_mat = conf_matrix(known_lbls, preds)
    macro_precision, macro_recall, macro_f1, overall_accuracy = (
        calc_macro_overall_metrics(
            np.concatenate([known_lbls, [-1] * (len(preds) - len(known_lbls))]),
            preds,
            idxs,
        )
    )

    plot_rate_metrics({"FAR": FAR, "FRR": FRR, "TAR": TAR, "URR": URR})
    plot_macro_overall_metrics(
        {
            "Precision (macro)": macro_precision,
            "Recall (macro)": macro_recall,
            "F1 (macro)": macro_f1,
            "Overall Accuracy": overall_accuracy,
        }
    )

    plot_distance_metrics(intra_dist, inter_dist)
    plot_per_subject_accuracy(correct, total)
    plot_confusion_matrix(
        np.concatenate([known_lbls, [-1] * (len(preds) - len(known_lbls))]), preds
    )
    plot_classification_report(clfr)
    plot_tsne(
        np.concatenate([known, unknown]),
        ["known"] * len(known) + ["unknown"] * len(unknown),
    )

    logger.info(f"Optimal Threshold             : {opt_thresh:.4f}")
    logger.info(f"False Acceptance Rate (FAR)   : {FAR:.4f}")
    logger.info(f"False Rejection Rate  (FRR)   : {FRR:.4f}")
    logger.info(f"True Acceptance Rate  (TAR)   : {TAR:.4f}")
    logger.info(f"Unknown Rejection Rate(URR)   : {URR:.4f}")
    logger.info(f"Intra-Class Distance  (AVG)   : {intra_dist:.4f}")
    logger.info(f"Inter-Class Distance  (AVG)   : {inter_dist:.4f}")
    logger.info("Per-Subject Accuracy")
    for label in idxs:
        acc = correct[label] / total[label]
        logger.info(f"S{(label+1):03d}: {acc:.4f}")
    logger.info(f"\nClassification Report:\n{clfr}")

    print(f"All visualizations have been saved to the 'xtras/charts' directory.")


if __name__ == "__main__":
    main()
