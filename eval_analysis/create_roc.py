
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import glob
from eval_tool import EvalTool, CustomMetrics
import argparse
import pickle
import os
import sys
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix


def get_path():
    parser.add_argument("-i", type=str, help="Path to the test prediction file")
    parser.add_argument("-o", type=str, help="Path to save the analysis file", default="./analysis.pkl")
    args = parser.parse_args()
    return args

# Create a list of all the files in the directory


if __name__ == "__main__":
    args = get_path()
    TEST_PRED_PATH = args.i
    SAVE_PATH = args.o
    files = glob.glob(TEST_PRED_PATH + "*/pred*")

    print("Found {} files".format(len(files)))
    # Run the analysis on all the files
    ground_truths = []
    predictions = []
    all_probas = []
    all_labels = []
    pts = []

    assert len(files) > 0, "No files found in {}".format(TEST_PRED_PATH)
    for file_path in tqdm(files):
        try:
            evaluator = EvalTool(file_path)
            ground_truths.append(evaluator.get_true_classes())
            predictions.append(evaluator.get_pred_classes())
            all_probas.append(evaluator.pred_matrix)
            all_labels.append(evaluator.true_matrix)
            pts.append(evaluator.get_pt())
        except Exception as e:
            print("Error processing file {}".format(file_path))

    all_ground_truths_cls = np.concatenate(ground_truths)
    all_predicted_cls = np.concatenate(predictions)
    all_probas = np.concatenate(all_probas, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    print