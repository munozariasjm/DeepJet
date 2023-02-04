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

parser = argparse.ArgumentParser()


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
            tqdm.write("Processing file {}".format(file_path))
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
    # Run the metrics

    metrics = CustomMetrics()

    # For b vs udsg
    b_probas = all_probas[evaluator.prob_vars.index("prob_isB")]
    udsg_probas = all_probas[evaluator.prob_vars.index("prob_isUDS")] + all_probas[evaluator.prob_vars.index("prob_isG")]
    b_labels = all_labels[evaluator.true_vars.index("isB")]
    udsg_labels = all_labels[evaluator.true_vars.index("isUDS")] + all_labels[evaluator.true_vars.index("isG")]

    b_vs_udsg_tprs, b_vs_udsg_rejs  = metrics.get_curves(b_probas, udsg_probas, b_labels, udsg_labels)

    # For b vs c
    c_probas = all_probas[evaluator.prob_vars.index("prob_isC")]
    c_labels = all_labels[evaluator.true_vars.index("isC")]

    b_vs_c_tprs, b_vs_c_rejs = metrics.get_curves(b_probas, c_probas, b_labels, c_labels)

    df = pd.DataFrame({"b_vs_udsg_tprs": b_vs_udsg_tprs, "b_vs_udsg_rejs": b_vs_udsg_rejs, "b_vs_c_tprs": b_vs_c_tprs, "b_vs_c_rejs": b_vs_c_rejs})
    df.to_pickle(SAVE_PATH)
    report = classification_report(all_ground_truths_cls, all_predicted_cls, target_names=evaluator.prob_vars, output_dict=True)
    print(report)


