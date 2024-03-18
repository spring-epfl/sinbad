import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import List
from block_classifier.crawl.types import BlockVo, BrowserContext

import joblib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
)
from sklearn.utils import resample
from treeinterpreter import treeinterpreter as ti

from block_classifier.crawl.saliency import saliency_score
from block_classifier.crawl.crawler import load_website, setup_webdriver
from block_classifier.features import get_block_features
from block_classifier.label import draw_site_with_blocks, pre_score_blocks
from .utils import is_notebook

sys.path.insert(1, os.path.join(os.path.split(os.path.abspath(__file__))[0], ".."))

from .logger import LOGGER


def get_perc(num, den):

    """
    Helper function to get percentage value.

    Args:
        num: Numerator
        den: Denominator
    Returns:
        Percentage, rounded to 2 decimal places.
    """

    return str(round(num / den * 100, 2)) + "%"


def print_stats(report, result_dir, avg="macro avg", stats=["mean", "std"]):

    """
    Function to make classification stats report. This gives us the metrics over all folds.

    Args:
        report: Results of all the folds.
        result_dir: Output directory for results.
        avg: Type of average we want (macro).
        stats: Stats we want (mean/std).
    Returns:
        Nothing, writes to a file.
    """

    by_label = report.groupby("label").describe()
    fname = os.path.join(result_dir, "scores")
    with open(fname, "w") as f:
        for stat in stats:
            print(by_label.loc[avg].xs(stat, level=1))
            x = by_label.loc[avg].xs(stat, level=1)
            f.write(by_label.loc[avg].xs(stat, level=1).to_string())
            f.write("\n")


def report_feature_importance(feature_importances, result_dir):

    """
    Function to log feature importances to a file.

    Args:
        feature_importances: Feature importances.
        result_dir: Output directory for results.
    Returns:
        Nothing, writes to a file.
    """

    fname = os.path.join(result_dir, "featimp")
    with open(fname, "a") as f:
        f.write(feature_importances.to_string())
        f.write("\n")


def report_true_pred(y_true, y_pred, name, vid, i, result_dir):

    """
    Function to make truth/prediction output file, and confustion matrix file.

    Args:
        y_true: Truth values.
        y_pred: Predicted values.
        name: Classified resource URLs.
        vid: Visit IDs.
        i: Fold number.
        result_dir: Output directory.
    Returns:
        Nothing, writes to files.
    """

    fname = os.path.join(result_dir, "tp_%s" % str(i))
    with open(fname, "w") as f:
        for i in range(0, len(y_true)):
            f.write(
                "%s |$| %s |$| %s |$| %s\n" % (y_true[i], y_pred[i], name[i], vid[i])
            )

    fname = os.path.join(result_dir, "confusion_matrix")
    with open(fname, "a") as f:
        f.write(
            np.array_str(confusion_matrix(y_true, y_pred, labels=[True, False]))
            + "\n\n"
        )


def describe_classif_reports(results, result_dir):

    """
    Function to make classification stats report over all folds.

    Args:
        results: Results of classification
        result_dir: Output directory
    Returns:
        all_folds: DataFrame of results

    This functions does the following:

    1. Obtains the classification metrics for each fold.
    """

    true_vectors, pred_vectors, name_vectors, vid_vectors = (
        [r[0] for r in results],
        [r[1] for r in results],
        [r[2] for r in results],
        [r[3] for r in results],
    )
    fname = os.path.join(result_dir, "scores")

    all_folds = pd.DataFrame(
        columns=["label", "fold", "precision", "recall", "f1-score", "support"]
    )
    for i, (y_true, y_pred, name, vid) in enumerate(
        zip(true_vectors, pred_vectors, name_vectors, vid_vectors)
    ):
        report_true_pred(y_true, y_pred, name, vid, i, result_dir)
        output = classification_report(y_true, y_pred)
        with open(fname, "a") as f:
            f.write(output)
            f.write("\n\n")
    return all_folds


def log_pred_probability(df_feature_test, y_pred, test_mani, clf, result_dir, tag):

    """
    Function to log prediction probabilities.

    Args:
        df_feature_test: Test feature DataFrame.
        y_pred: Test predictions.
        test_mani: Test feature and labels DataFrame.
        clf: Trained model
        result_dir: Output folder of results.
        tag: Fold number.
    Returns:
        Nothing, writes to file.
    """
    y_pred_prob = clf.predict_proba(df_feature_test)
    fname = os.path.join(result_dir, "predict_prob_" + str(tag))

    with open(fname, "w") as f:
        class_names = [str(x) for x in clf.classes_]
        s = " |$| ".join(class_names)
        f.write("Truth |$| Pred |$| " + s + " |$| Name |S| VID" + "\n")
        truth_labels = [str(x) for x in list(test_mani.label)]
        pred_labels = [str(x) for x in list(y_pred)]
        truth_vids = [str(x) for x in list(test_mani.url)]
        for i in range(0, len(y_pred_prob)):
            preds = [str(x) for x in y_pred_prob[i]]
            preds = " |$| ".join(preds)
            f.write(
                truth_labels[i]
                + " |$| "
                + pred_labels[i]
                + " |$| "
                + preds
                + " |$| "
                + truth_vids[i]
                + "\n"
            )


def log_interpretation(df_feature_test, test_mani, clf, result_dir, tag, cols):

    """
    Function to perform interpretation of test results.

    Args:
        df_feature_test: Test DataFrame.
        clf: Trained model
        result_dir: Output folder of results.
        tag: Fold number.
        cols: Feature column names.
    Returns:
        Nothing, writes to file.
    """

    preds, bias, contributions = ti.predict(clf, df_feature_test)
    fname = os.path.join(result_dir, "interpretation_" + str(tag))
    with open(fname, "w") as f:
        data_dict = {}
        for i in range(len(df_feature_test)):
            name = test_mani.iloc[i]["id"]
            vid = str(test_mani.iloc[i]["url"])
            key = name + "_" + str(vid)
            data_dict[key] = {}
            data_dict[key]["name"] = name
            data_dict[key]["vid"] = vid
            c = list(contributions[i, :, 0])
            c = [round(float(x), 2) for x in c]
            fn = list(cols)
            fn = [str(x) for x in fn]
            feature_contribution = list(zip(c, fn))
            data_dict[key]["contributions"] = feature_contribution
        f.write(json.dumps(data_dict, indent=4))


def load_model(filepath):
    return joblib.load(filepath)


def classify_with_model(clf, test):

    test_mani = test.copy()
    df_feature_test = test_mani.to_numpy()

    return clf.predict(df_feature_test)


def classify(train, test, result_dir, tag, save_model, pred_probability, interpret):

    """
    Function to perform classification.

    Args:
        train: Train data.
        test: Test data.
        result_dir: Output folder for results.
        tag: Fold number.
        save_model: Boolean value indicating whether to save the trained model or not.
        pred_probability: Boolean value indicating whether to save the prediction probabilities or not.
        interpret: Boolean value indicating whether to use tree interpreter on predictions or not.
    Returns:
        list(test_mani.label): Truth labels of test data.
        list(y_pred): Predicted labels of test data.
        list(test_mani.name): URLs of test data.
        list(test_mani.visit_id): Visit IDs of test data.
    """

    train_mani = train.copy()
    test_mani = test.copy()

    clf = RandomForestClassifier(n_estimators=100)
    fields_to_remove = ["id", "url", "label"]

    df_feature_train = train_mani.drop(fields_to_remove, axis=1)
    df_feature_test = test_mani.drop(fields_to_remove, axis=1)

    # remove nan columns
    df_feature_train.dropna(axis=1, inplace=True)
    df_feature_test.dropna(axis=1, inplace=True)

    columns = df_feature_train.columns
    df_feature_train = df_feature_train.to_numpy()
    train_labels = train_mani.label.to_numpy()

    # Perform training
    clf.fit(df_feature_train, train_labels)

    # Obtain feature importances
    feature_importances = pd.DataFrame(
        clf.feature_importances_, index=columns, columns=["importance"]
    ).sort_values("importance", ascending=False)
    report_feature_importance(feature_importances, result_dir)

    # Perform classification and get predictions
    cols = df_feature_test.columns
    df_feature_test = df_feature_test.to_numpy()
    y_pred = clf.predict(df_feature_test)

    acc = accuracy_score(test_mani.label, y_pred)
    prec = precision_score(test_mani.label, y_pred, average="micro")
    rec = recall_score(test_mani.label, y_pred, average="micro")

    # Write accuracy score
    fname = os.path.join(result_dir, "accuracy")
    with open(fname, "a") as f:
        f.write("\nAccuracy score: " + str(round(acc * 100, 3)) + "%" + "\n")
        f.write("Precision score: " + str(round(prec * 100, 3)) + "%" + "\n")
        f.write("Recall score: " + str(round(rec * 100, 3)) + "%" + "\n")

    LOGGER.info("Accuracy Score: %f", acc)

    # Save trained model if save_model is True
    if save_model:
        model_fname = os.path.join(result_dir, "model_" + str(tag) + ".joblib")
        joblib.dump(clf, model_fname)
    if pred_probability:
        log_pred_probability(df_feature_test, y_pred, test_mani, clf, result_dir, tag)
    if interpret:
        log_interpretation(df_feature_test, test_mani, clf, result_dir, tag, cols)

    return (
        list(test_mani.label),
        list(y_pred),
        list(test_mani.id),
        list(test_mani.url),
    )


def classify_crossval(df_labelled, result_dir, save_model, pred_probability, interpret):

    """
    Function to perform cross validation.

    Args:
        df_labelled; DataFrame of features and labels.
        result_dir: Output folder for results.
        save_model: Boolean value indicating whether to save the trained model or not.
        pred_probability: Boolean value indicating whether to save the prediction probabilities or not.
        interpret: Boolean value indicating whether to use tree interpreter on predictions or not.
    Returns:
        results: List of results for each fold.
    """

    train_mani_maj = df_labelled[df_labelled["label"] == 0]
    train_mani_min = df_labelled[df_labelled["label"] == 1]

    train_mani_min = resample(
        train_mani_min, replace=True, n_samples=len(train_mani_maj)
    )

    df_labelled = pd.concat([train_mani_maj, train_mani_min])

    sid_list = list(range(0, len(df_labelled)))
    num_iter = 10
    num_test_sid = int(len(sid_list) / num_iter)
    used_test_ids = []
    results = []

    LOGGER.info("Total Number of sample IDs: %d", len(sid_list))
    LOGGER.info("Number of sample IDs to use in a fold: %d", num_test_sid)

    for i in range(0, num_iter):
        LOGGER.info("Performing fold: %d", i)
        sid_list_iter = list(set(sid_list) - set(used_test_ids))
        chosen_test_sid = random.sample(sid_list_iter, num_test_sid)
        used_test_ids += chosen_test_sid

        df_train = df_labelled.iloc[list(set(sid_list_iter) - set(chosen_test_sid))]
        df_test = df_labelled.iloc[chosen_test_sid]

        fname = os.path.join(result_dir, "composition")
        train_pos = len(df_train[df_train["label"] == 1])
        test_pos = len(df_test[df_test["label"] == 1])

        with open(fname, "a") as f:
            f.write("\nFold " + str(i) + "\n")
            f.write(
                "Train: "
                + str(train_pos)
                + " "
                + get_perc(train_pos, len(df_train))
                + "\n"
            )
            f.write(
                "Test: " + str(test_pos) + " " + get_perc(test_pos, len(df_test)) + "\n"
            )
            f.write("\n")

        result = classify(
            df_train, df_test, result_dir, i, save_model, pred_probability, interpret
        )
        results.append(result)

    return results


def pipeline(dataset_file, result_dir, save_model, pred_probability, interpret):

    """
    Function to run classification pipeline.

    Args:
      feature_file: CSV file of features from feature extraction process.
      label_file: CSV file of labels from labelling process.
      result_dir: Output folder for results.
      save_model: Boolean value indicating whether to save the trained model or not.
      pred_probability: Boolean value indicating whether to save the prediction probabilities or not.
      interpret: Boolean value indicating whether to use tree interpreter on predictions or not.
    Returns:
      Nothing, creates a result directory with all the results.
    """

    if isinstance(dataset_file, str):
        df_labelled = pd.read_csv(dataset_file)
    elif isinstance(dataset_file, pd.DataFrame):
        df_labelled = dataset_file
    else:
        raise TypeError(f"not supported for dataset_file: {type(df_labelled)}")

    Path(result_dir).mkdir(parents=True, exist_ok=True)

    results = classify_crossval(
        df_labelled, result_dir, save_model, pred_probability, interpret
    )
    report = describe_classif_reports(results, result_dir)


def classify_block_vo_list(
    blocks: List[BlockVo],
    browser_context: BrowserContext,
    model: RandomForestClassifier,
) -> dict[str, int]:

    if not len(blocks):
        return {}
    blocks, scores = pre_score_blocks(blocks, browser_context, saliency_score, 0)
    bf = pd.DataFrame(
        [get_block_features(block, blocks, browser_context) for block in blocks]
    )
    if len(bf) == 0:
        return {}
    
    labels = classify_with_model(model, bf)

    return {b.id: int(labels[i]) for i, b in enumerate(blocks)}


# test on a website
def debug_page(url, model_path):
    driver = setup_webdriver()
    (blocks, web_screenshot, browser_context) = load_website(url, driver, wait=30)

    blocks, scores = pre_score_blocks(blocks, browser_context, saliency_score, 0)

    bf = pd.DataFrame(
        [get_block_features(block, blocks, browser_context) for block in blocks]
    )

    clf = load_model(model_path)

    labels = classify_with_model(clf, bf)

    imp_blocks = []
    unimp_blocks = []

    for i, b in enumerate(blocks):

        if labels[i] == 1:
            imp_blocks.append(b)
        else:
            unimp_blocks.append(b)

    fig, ax = plt.subplots(figsize=(14, 10))

    draw_site_with_blocks(ax, imp_blocks, scores, web_screenshot, browser_context)

    if not is_notebook():
        input("type anything to close: ")

    driver.quit()


def main(program: str, args: List[str]):

    parser = argparse.ArgumentParser(
        prog=program, description="Run the WebGraph classification pipeline."
    )

    parser.add_argument(
        "--debug", type=str, help="url to test a trained model on a page", default=None
    )

    parser.add_argument(
        "--trained",
        type=str,
        help="path to a trained model .joblib file",
        default="pretrained-models/model-0.joblib",
    )

    parser.add_argument(
        "--data",
        type=str,
        help="Data CSV file.",
        default="dataset-25.07.2022/features.csv",
    )
    parser.add_argument(
        "--out", type=str, help="Directory to output the results.", default="results"
    )
    parser.add_argument(
        "--save", type=bool, help="Save trained model file.", default=False
    )
    parser.add_argument(
        "--probability", type=bool, help="Log prediction probabilities.", default=False
    )
    parser.add_argument(
        "--interpret", type=bool, help="Log results of tree interpreter.", default=False
    )

    ns = parser.parse_args(args)

    if not ns.debug:
        pipeline(ns.data, ns.out, ns.save, ns.probability, ns.interpret)

    else:
        debug_page(ns.debug, ns.trained)


if __name__ == "__main__":

    main(sys.argv[0], sys.argv[1:])