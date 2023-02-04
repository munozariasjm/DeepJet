import pandas as pd
import glob
import os
import uproot
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import awkward as ak
from joblib import Parallel, delayed
from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix


class EvalTool:
    def __init__(self, file_path, var=None, savepath=None):
        self.file = uproot.open(file_path)
        self.variables = ['prob_isB', 'prob_isBB', 'prob_isLeptB', 'prob_isC', 'prob_isUDS', 'prob_isG', 'isB', 'isBB', 'isLeptB', 'isC', 'isUDS', 'isG', 'jet_pt', 'jet_eta']
        self.prob_vars = [var for var in self.variables if "prob" in var]
        self.true_vars = [var for var in self.variables if var not in self.prob_vars and not var.startswith("jet")]

        self.true_matrix = self.trues_matrix()
        self.pred_matrix = self.preds_matrix()

    def get_var(self, varname):
        leaf = self.file["tree;1"][varname]
        return leaf.array().to_numpy()

    def trues_matrix(self):
        data = [self.get_var(var) for var in self.true_vars]
        return np.stack(data, axis=1)

    def preds_matrix(self):
        data = [self.get_var(var) for var in self.prob_vars]
        return np.stack(data, axis=1)

    @staticmethod
    def get_class(matrix):
        return np.argmax(matrix, axis=1)

    def get_pt(self):
        return self.get_var("jet_pt")

    def get_pred_classes(self):
        return self.get_class(self.pred_matrix)

    def get_true_classes(self):
        return self.get_class(self.true_matrix)

    def build_report(self, ground_truth: np.ndarray, predicted: np.ndarray):
        predicted_class = np.argmax(self.pred_matrix, axis=1)
        real_class = np.argmax(self.true_matrix, axis=1)
        return classification_report(real_class, predicted_class, target_names=self.prob_vars, output_dict=True)


class CustomMetrics:

    def __init__(self):
        pass

    def get_rates(self, probas: np.array, labels:np.array, threshold:float):
        """
        Get the true positive rate and false positive rate for a given variable

        The proba is the array of probabilities for the given variable
        Labels is the array of labels
        """
        tpr = np.sum((probas > threshold) & (labels == 1)) / np.sum(labels == 1)
        fpr = np.sum((probas > threshold) & (labels == 0)) / np.sum(labels == 0)
        return tpr, fpr


    def versus_rate(self, probas_1, probas_2, labels_1, labels_2, threshold):

        tpr_1, fpr_1 = self.get_rates(probas_1, labels_1, threshold)
        tpr_2, fpr_2 = self.get_rates(probas_2, labels_2, threshold)
        fpvs =  fpr_1 / (fpr_1 + fpr_2)
        tps = tpr_1 / (tpr_1 + tpr_2)
        return fpvs, tps

    def get_curves(self, probas_1, probas_2, labels_1, labels_2, n_split=1):
        """Calculate the rejection rate for a given threshold and probability"""
        thresholds = np.linspace(0.01, 1, 1000)
        all_rejs = []
        all_tprs = []

        rejs = []
        tprs = []
        accuracy = []
        for th in thresholds:
            tps, fps = self.versus_rate( probas_1, probas_2, labels_1, labels_2, th)
            tprs.append(tps)
            rejs.append(1/fps)
        return tprs, rejs



