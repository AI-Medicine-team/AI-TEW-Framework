"""
Calculate the Youden index, NPV, PPV, etc.
"""

from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, classification_report
import pandas as pd
import numpy as np


def youden_index(y_true, y_score, pos_label=1, start=0, end=1):
    """
    Clinical diagnostic indicators include: PPV, NPV, Sensitivity, Specificity, Youden index,
    and TP, FP, TN, FN.
    Youden Index = sensitivity + specificity - 1
    F1 = 2 * ppv * sensitivity / (ppv + sensitivity)

    Input:
        y_true: the true binary labels, 1D numpy array
        y_score: predicted scores or probabilities
        pos_label: 1 for positive class, 0 for negative class
        start, end: threshold range for evaluation

    Return:
        df: DataFrame containing indicators under different thresholds
        max_ji_val: maximum Youden index
        max_f1_val: maximum F1 score
        roc_auc: AUC of the ROC curve
    """

    fpr, tpr, thr = roc_curve(y_true, y_score, pos_label=pos_label)
    roc_auc = auc(fpr, tpr)

    # generate thresholds
    thrs = list(np.arange(start, end, 0.005))

    # Print header (original Chinese comment removed)
    # print("Threshold\tACC\tPPV\tNPV\tSensitivity\tSpecificity\tYoudenIdx\tF1\tTrueBenign\tTrueMalignant\tPredBenign\tPredMalignant\tTP\tFP\tTN\tFN")
    cols = "Thr\tACC\tPPV\tNPV\tSens(Rec/TPR)\tSpec\tYoudenIdx\tF1\tTrueBen\tTrueMal\tPredBen\tPredMal\tTP\tFP\tTN\tFN"
    columns = cols.split('\t')

    result = list()
    for i, f in enumerate(thrs):
        y_pred = np.zeros(y_score.shape[0], dtype=np.int8)
        idx = np.where(y_score >= f)  # >= threshold gives label 1
        y_pred[idx] = 1

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        _fpr, _tpr = 0.0, 0.0
        if fp + tn != 0:
            _fpr = fp / (fp + tn)
        if tp + fn != 0:
            _tpr = tp / (tp + fn)

        ppv, npv = 0.0, 0.0
        if tp + fp != 0:
            ppv = tp / (tp + fp)
        if tn + fn != 0:
            npv = tn / (tn + fn)

        spec = 1.0 - _fpr
        acc = (tp + tn) / float(tn + fp + fn + tp)
        jord_idx = _tpr + spec - 1
        f1 = 2 * ppv * _tpr / (ppv + _tpr)

        row = [f, acc, ppv, npv, _tpr, spec, jord_idx, f1,
               tp + fn, fp + tn, tp + fp, tn + fn, tp, fp, tn, fn]
        result.append(row)

    df = pd.DataFrame(result, columns=columns)

    max_ji_val = df['YoudenIdx'].max()
    max_f1_val = df['F1'].max()

    return df, max_ji_val, max_f1_val, roc_auc
