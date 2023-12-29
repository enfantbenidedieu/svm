# load packages
from sklearn import datasets
import numpy as np
import pandas as pd
from sklearn import metrics
from shiny import ui

def generate_data(n_samples, dataset, noise):
    if dataset == "moons":
        return datasets.make_moons(n_samples=n_samples, noise=noise, random_state=0)

    elif dataset == "circles":
        return datasets.make_circles(
            n_samples=n_samples, noise=noise, factor=0.5, random_state=1
        )

    elif dataset == "linear":
        X, y = datasets.make_classification(n_samples=n_samples,n_features=2,n_redundant=0,n_informative=2,
                        random_state=2,n_clusters_per_class=1,
        )

        rng = np.random.RandomState(2)
        X += noise * rng.uniform(size=X.shape)
        linearly_separable = (X, y)

        return linearly_separable

    else:
        raise ValueError(
            "Data type incorrectly specified. Please choose an existing dataset."
        )
    
# Performance Card
def card_performance(id_name,perf_name,perf_id):
    return ui.card(
                ui.card_header(perf_name,style="text-align:center;"),
                ui.output_text(perf_id),
                theme="purple",
                showcase_layout="bottom",
                full_screen=False,
                id=id_name
            )

# Accuracy
def accuracy(model, xtest, ytest, threshold):
    y_pred_test = (model.decision_function(xtest) > threshold).astype(int)
    test_score = metrics.accuracy_score(y_true=ytest, y_pred=y_pred_test)
    return round(100*test_score,2)

# AUC ROC score
def auc_roc_score(model, xtest, ytest):
    decision_test = model.decision_function(xtest)
    auc_score = metrics.roc_auc_score(y_true=ytest, y_score=decision_test)
    return round(100*auc_score,2)

# Average Precision score
def average_precision_score(model, xtest, ytest):
    decision_test = model.decision_function(xtest)
    average_score = metrics.average_precision_score(y_true=ytest, y_score=decision_test)
    return round(100*average_score,2)

# F1 - score
def f1_score(model, xtest, ytest, threshold):
    y_pred_test = (model.decision_function(xtest) > threshold).astype(int)
    test_score = metrics.f1_score(y_true=ytest, y_pred=y_pred_test)
    return round(100*test_score,2)

# Precision score
def precision_score(model, xtest, ytest, threshold):
    y_pred_test = (model.decision_function(xtest) > threshold).astype(int)
    test_score = metrics.precision_score(y_true=ytest, y_pred=y_pred_test)
    return round(100*test_score,2)

# Recall score
def recall_score(model, xtest, ytest, threshold):
    y_pred_test = (model.decision_function(xtest) > threshold).astype(int)
    test_score = metrics.recall_score(y_true=ytest, y_pred=y_pred_test)
    return round(100*test_score,2)

# Classification Report
def classification_report(model,xtest,ytest,threshold):
    '''Source: https://stackoverflow.com/questions/39662398/scikit-learn-output-metrics-classification-report-into-csv-tab-delimited-format'''
    y_pred_test = (model.decision_function(xtest) > threshold).astype(int)
    report = metrics.classification_report(ytest, y_pred_test,target_names=["normal","abnormal"], output_dict=True)
    df_classification_report = pd.DataFrame(report).transpose()
    #df_classification_report = df_classification_report.sort_values(by=['f1-score'], ascending=False)
    return df_classification_report


