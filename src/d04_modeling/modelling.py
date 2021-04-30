#!/usr/bin/env python
# coding: utf-8

# # Model Building
# ## Table of Contents
# * [1. Import and Prepare Data](#import_data)
# * [2. Modelling Helper Functions](#modelling_helper)
#     * [2.1 Cost-Sensitive Helper Functions](#cost_sensitive)
# * [3. Model Training](#model_training)
#     * [3.1 Logistic Regression](#logistic_regression)
#     * [3.2 Random Forrest](#random_forrest)
#     * [3.3 XGBoost](#xgboost)
#     * [3.4 LightGBM](#lightgbm)
#     * [3.5 CatBoost](#catboost)
# * [4. Model Comparison and Evaluation](#model_comparison)
#     * [4.1 Partial Dependence Analysis](#pdp)
#     * [4.2 SHAP](#shap)
# * [5. Export Predictions](#export)
#         
# 


# Required Packages
#DS Packages
import pandas as pd
import numpy as np


#Utils
from dateutil.relativedelta import relativedelta
from datetime import datetime
import timeit # time model training time
import pickle # save data in pkl format
import os, sys
from tabulate import tabulate # package to display tables
import json # save and load json

#Data Viualization
import seaborn as sns
import matplotlib.pyplot as plt

#Metrics 
from sklearn import metrics
from sklearn.metrics import f1_score
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score,                             precision_recall_curve, roc_curve, accuracy_score
from sklearn.exceptions import NotFittedError
from sklearn.calibration import calibration_curve
from sklearn.metrics import accuracy_score

#Test Train split
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier

from scipy import stats
from scipy.stats import randint
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform

from catboost import CatBoostClassifier

import warnings


def encode_and_bind(original_dataframe, feature_to_encode):
    dummies = pd.get_dummies(original_dataframe[[feature_to_encode]], prefix=feature_to_encode)
    res = pd.concat([original_dataframe, dummies], axis=1)
    res = res.drop([feature_to_encode], axis=1)
    return(res) 

# drops unneeded columns:
def drop_cols(_df):
    _df = _df.drop(labels=['delivery_date',
                           'order_date',
                           'item_id',
                           'item_size',
                           'item_color',
                           'brand_id',
                           'user_id',
                           'user_dob',
                           'user_reg_date',
                           'is_first_purchase',                   
                          ]
                   , axis=1)
    return _df

def confusion_plot(matrices, labels=None, titles=None , nrows=1, ncols = 1):
    """ Display binary confusion matrix as a Seaborn heatmap """
    
    labels = labels if labels else ['Negative (0)', 'Positive (1)']
    
    if nrows != 1 or ncols != 1:
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols)
        for i, j, ax in zip(matrices, titles, axs.flat):
            # Plot heatmap
            ax.set_title(j)
            sns.heatmap(i, ax=ax, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap='Blues', cbar=False)
            ax.set_xlabel('PREDICTED')
            ax.set_ylabel('ACTUAL')
    else:
        fig, ax = plt.subplots(nrows=nrows, ncols=ncols)
        ax.set_title(titles[0])
        sns.heatmap(matrices[0], ax=ax, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap='Blues', cbar=False)
        ax.set_xlabel('PREDICTED')
        ax.set_ylabel('ACTUAL')
    
    plt.close()
    
    return fig

def roc_plot(y_true, y_probs, label, compare=False, ax=None):
    """ Plot Receiver Operating Characteristic (ROC) curve 
        Set `compare=True` to use this function to compare classifiers. """
    
    fpr, tpr, thresh = roc_curve(y_true, y_probs,
                                 drop_intermediate=False)
    auc = round(roc_auc_score(y_true, y_probs), 2)
    
    fig, axis = (None, ax) if ax else plt.subplots(nrows=1, ncols=1)
    label = ' '.join([label, f'({auc})']) if compare else None
    sns.lineplot(x=fpr, y=tpr, ax=axis, label=label)
    
    if compare:
        axis.legend(title='Classifier (AUC)', loc='lower right')
    else:
        axis.text(0.72, 0.05, f'AUC = { auc }', fontsize=12,
                  bbox=dict(facecolor='green', alpha=0.4, pad=5))
            
        # Plot No-Info classifier
        axis.fill_between(fpr, fpr, tpr, alpha=0.3, edgecolor='g',
                          linestyle='--', linewidth=2)
        
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.set_title('ROC Curve')
    axis.set_xlabel('False Positive Rate [FPR]\n(1 - Specificity)')
    axis.set_ylabel('True Positive Rate [TPR]\n(Sensitivity or Recall)')
    
    plt.close()
    
    return axis if ax else fig

def precision_recall_plot(y_true, y_probs, label, compare=False, ax=None):
    """ Plot Precision-Recall curve.
        Set `compare=True` to use this function to compare classifiers. """
    
    p, r, thresh = precision_recall_curve(y_true, y_probs)
    p, r, thresh = list(p), list(r), list(thresh)
    p.pop()
    r.pop()
    
    fig, axis = (None, ax) if ax else plt.subplots(nrows=1, ncols=1)
    
    if compare:
        sns.lineplot(r, p, ax=axis, label=label)
        axis.set_xlabel('Recall')
        axis.set_ylabel('Precision')
        axis.legend(loc='lower left')
    else:
        sns.lineplot(thresh, p, label='Precision', ax=axis)
        axis.set_xlabel('Threshold')
        axis.set_ylabel('Precision')
        axis.legend(loc='lower left')

        axis_twin = axis.twinx()
        sns.lineplot(thresh, r, color='limegreen', label='Recall', ax=axis_twin)
        axis_twin.set_ylabel('Recall')
        axis_twin.set_ylim(0, 1)
        axis_twin.legend(bbox_to_anchor=(0.24, 0.18))
    
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.set_title('Precision Vs Recall')
    
    plt.close()
    
    return axis if ax else fig

def feature_importance_plot(importances, feature_labels, ax=None):
    fig, axis = (None, ax) if ax else plt.subplots(nrows=1, ncols=1, figsize=(5, 10))
    sns.barplot(x=importances, y=feature_labels, ax=axis)
    axis.set_title('Feature Importance Measures')
    
    plt.close()
    
    return axis if ax else fig

def model_memory_size(clf):
    return sys.getsizeof(pickle.dumps(clf))

def tpr_fpr_calc(cutoff, yhat_prob, y_true):

    # temp variable giving distinct predictions based on the cutoff given as an input
    temp = (yhat_prob >= cutoff).astype(bool)

    # Create confusion matrix for this prediction
    cmat = metrics.confusion_matrix(y_true, temp)
    
    # Calculate FPR
    fpr = cmat[0,1] / (cmat[0,1] + cmat[0,0])

    # Calculate TPR
    tpr = cmat[1,1] / (cmat[1,1] + cmat[1,0])

    return tpr, fpr

def report(clf, x_train, y_train, x_test, y_test,
           sample_weight=None, refit=False, importance_plot=False,
           confusion_labels=None, feature_labels=None, verbose=True):
    """ Trains the passed classifier if not already trained and reports
        various metrics of the trained classifier """
    
    dump = dict()
    
    # Train Predictions and Accuracy
    train_predictions = clf.predict(x_train)
    train_acc = accuracy_score(y_train, train_predictions)
    
    ## Testing
    start = timeit.default_timer()
    test_predictions = clf.predict(x_test)
    
    test_acc = accuracy_score(y_test, test_predictions)
    y_probs = clf.predict_proba(x_test)[:, 1]
    
    roc_auc = roc_auc_score(y_test, y_probs)
        
    train_avg_cost = calc_custom_cost_score(y_train.values, train_predictions, list(x_train['item_price']))
    test_avg_cost = calc_custom_cost_score(y_test.values, test_predictions, list(x_test['item_price']))
        
    ## Model Memory
    model_mem = round(model_memory_size(clf) / 1024, 2)
    
    print(clf)
    print("\n=============================> TRAIN-TEST DETAILS <======================================")
    
    ## Metrics
    print(f"Train Size: {x_train.shape[0]} samples")
    print(f" Test Size: {x_test.shape[0]} samples")
    print("---------------------------------------------")
    print("Train Accuracy: ", train_acc)
    print(" Test Accuracy: ", test_acc)
    print("---------------------------------------------")
    print("Train Average Cost: ", train_avg_cost)
    print(" Test Average Cost: ", test_avg_cost)
    print("---------------------------------------------")
    print(" Area Under ROC (test): ", roc_auc)
    print("---------------------------------------------")
    
    print(f"Model Memory Size: {model_mem} kB")
    print("\n=============================> CLASSIFICATION REPORT <===================================")
    
    ## Classification Report
    clf_rep = classification_report(y_test, test_predictions, output_dict=True)
    
    print(classification_report(y_test, test_predictions,
                                target_names=confusion_labels))

    cost_matrix = calc_custom_cost_score(y_test.values, test_predictions, list(x_test['item_price']), matrix = True)

    # Calculate calibration using calibration_curve function
    prob_true, prob_pred = calibration_curve(y_test, y_probs, n_bins = 20)
    # Calculate Bayes optimal threshold
    threshold_bayes = (cost_matrix[1][0]               # C(b,G)
                        /(cost_matrix[1][0]             # C(b,G)
                            +cost_matrix[0][1])).round(5) # C(g,B)
    
    #Find optimal cutoff based on random cutoff values
    possible_cutoffs = np.arange(0.0, 1.0, 0.001)
    costs = {}
    for cutoff in possible_cutoffs:
        pred = np.where(y_probs >= cutoff, 1, 0)
        costs[cutoff] = (calc_custom_cost_score(y_test.values, pred, list(x_test['item_price'])))
        
    threshold_empiric = min(costs, key=costs.get)
    
    # Compare Thresholds
    pred_default = np.where(y_probs >= 0.5, 1, 0) # 0.5 is the default cut-off, equivalant to y_pred from above
    pred_bayes= np.where(y_probs >= threshold_bayes, 1, 0) # Using the cut-off defined by the cost-minimal threshold function
    pred_empiric = np.where(y_probs >= threshold_empiric, 1 , 0)# Empric cut-off
    
    err_cost_default = test_avg_cost
    err_cost_cost_bayes = calc_custom_cost_score(y_test.values,pred_bayes, list(x_test['item_price']))
    err_cost_empiric = calc_custom_cost_score(y_test.values,pred_empiric, list(x_test['item_price']))
    
    accurracy_default = accuracy_score(y_test, pred_default)
    accuracy_bayes = accuracy_score(y_test, pred_bayes)
    accuracy_empiric = accuracy_score(y_test, pred_empiric)
    
    # save best cutoff
    cutoffs = {0.5 : err_cost_default,
                threshold_bayes : err_cost_cost_bayes,
                threshold_empiric : err_cost_empiric   
                }
    best_cutoff = min(cutoffs, key=cutoffs.get)
    best_err_cost = cutoffs[best_cutoff]
    # Compare Cutoffs
    table_data = [
        ['', 'Default Cutoff', ' cost-minimal Bayes cutoff', 'Empric minimal cutoff'],
        ['Test Cutoff Threshold', 0.5, threshold_bayes, threshold_empiric],
        ['Test Error Cost', err_cost_default, err_cost_cost_bayes, err_cost_empiric],
        ['Test Accuracy', accurracy_default, accuracy_bayes, accuracy_empiric]
    ]
    print(tabulate(table_data, headers = 'firstrow'))

    if verbose:
             
        print("\n================================> COST-SENSITIVE EVALUTATION <=====================================")

        
        
        # Calibration curve
        plt.rcParams["figure.figsize"] = (12,6)

        

        # Plot results
        plt.plot(prob_pred, prob_true, marker = '.', label = clf.__class__.__name__)  
        plt.title(f'Calibration Plot for {clf.__class__.__name__} model')
        plt.ylabel("True Probability per Bin")
        plt.xlabel("Predicted Probability") 
        plt.plot([0, 1], [0, 1], linestyle = '--', label = 'Ideally Calibrated'); 
        plt.legend();
        plt.show()
        
        print("------------------------------------------------------------------------------------------")
    
        
        
        print("\n================================> CONFUSION MATRICES <=====================================")
        #Compare default error cost and accuracy to bayes error cost and accuracy
        cmat_default = metrics.confusion_matrix(y_test, pred_default)
        cmat_bayes = metrics.confusion_matrix(y_test, pred_bayes)
        cmat_empiric = metrics.confusion_matrix(y_test, pred_empiric)
        
        plt.rcParams["figure.figsize"] = (12,4)
        display(confusion_plot([cmat_default, cmat_bayes, cmat_empiric],
                               nrows=1,
                               ncols=3,
                               labels=confusion_labels,
                               titles=['Default', 'Bayes', 'Empiric'] ))
        
        print("\n================================> CALIBRATION CURVE (RELIABILITY PLOT) <=====================================")
        # Calibration curve (reliability plot)
        # Calculate all FPRs and TPRs for the LogitCV model
        fpr, tpr, _ = metrics.roc_curve(y_test, y_probs, pos_label=1)

        # Calculate TPR and FPR for both cutoffs
        tpr_best_cutoff, fpr_best_cutoff = tpr_fpr_calc(best_cutoff, y_probs, y_test)
        tpr_default, fpr_default = tpr_fpr_calc(0.5, y_probs, y_test)
        
        # Plot ROC curve and mark cutoffs on the curve
        plt.rcParams["figure.figsize"] = (12,6)
        plt.plot(fpr, tpr, label = "ROC curve")
        plt.plot(fpr_default, tpr_default, marker="x", markersize=20, label ="0.5 cutoff")
        plt.plot(fpr_best_cutoff, tpr_best_cutoff, marker="x", markersize=20,  label =f"Optimal cutoff")
        plt.xlabel("1-Specificity")
        plt.ylabel("Sensitivity")
        plt.title('Calibration curve (reliability plot)')
        plt.legend();
        plt.show()
        
        print("\n=======================================> FEATURE IMPORTANCE AND ROC <=========================================")

        ## Variable importance plot
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 10))
        roc_axes = axes[0, 0]
        pr_axes = axes[0, 1]
        importances = None

        if importance_plot:
            if not feature_labels:
                raise RuntimeError("'feature_labels' argument not passed "
                                   "when 'importance_plot' is True")

            try:
                importances = pd.Series(clf.feature_importances_,
                                        index=feature_labels) \
                                .sort_values(ascending=False)
            except AttributeError:
                try:
                    importances = pd.Series(clf.coef_.ravel(),
                                            index=feature_labels) \
                                    .sort_values(ascending=False)
                except AttributeError:
                    pass

            if importances is not None:
                # Modifying grid
                grid_spec = axes[0, 0].get_gridspec()
                for ax in axes[:, 0]:
                    ax.remove()   # remove first column axes
                large_axs = fig.add_subplot(grid_spec[0:, 0])

                # Plot importance curve
                feature_importance_plot(importances=importances.values,
                                        feature_labels=importances.index,
                                        ax=large_axs)
                large_axs.axvline(x=0)

                # Axis for ROC and PR curve
                roc_axes = axes[0, 1]
                pr_axes = axes[1, 1]
            else:
                # remove second row axes
                for ax in axes[1, :]:
                    ax.remove()
        else:
            # remove second row axes
            for ax in axes[1, :]:
                ax.remove()


        ## ROC and Precision-Recall curves
        clf_name = clf.__class__.__name__
        roc_plot(y_test, y_probs, clf_name, ax=roc_axes)
        precision_recall_plot(y_test, y_probs, clf_name, ax=pr_axes)

        fig.subplots_adjust(wspace=5)
        fig.tight_layout()
        display(fig)
    ## Dump to report_dict
    dump = dict(clf=clf, accuracy=[train_acc, test_acc],
                train_predictions=train_predictions,
                test_predictions=test_predictions,
                test_probs=y_probs,
                report=clf_rep,
                roc_auc=roc_auc,
                model_memory=model_mem,
                opt_cutoff = best_cutoff,
                total_cost = best_err_cost)
    
    return dump

# Custom Compare models function, builds a table to compare results
def compare_models(y_test=None, clf_reports=[]):
    
    default_names = [rep['clf'].__class__.__name__ for rep in clf_reports]
    
    table = dict()
    index = ['Memory Size', 'Train Accuracy ' , 'Test Accuracy ', 'ROC AUC', 'Optimal Cutoff', 'Total Cost' ]
    
    for i in range(len(clf_reports)):
        model_memory = clf_reports[i]['model_memory']
        train_accuracy = clf_reports[i]['accuracy'][0]
        test_accuracy = clf_reports[i]['accuracy'][1]
        roc_auc = clf_reports[i]['roc_auc']
        opt_cutoff = clf_reports[i]['opt_cutoff']
        total_cost = clf_reports[i]['total_cost']
        
        
        table[default_names[i]] = [
                                    model_memory,
                                    train_accuracy,
                                    test_accuracy,
                                    roc_auc,
                                    opt_cutoff,
                                    total_cost]

    table = pd.DataFrame(data=table, index=index)
    
    return table.T

# ### 2.1 Cost-Sensitive Helper Functions
def calc_custom_cost_score(y_true, y_pred, prices, matrix=False):

    # initiate empty arrays for FP/FN costs
    FP = np.zeros(len(y_true))
    FN = np.zeros(len(y_true))
    
    # Iterate through all rows
    for i in range(len(y_true)):
        if y_true[i] == 0 and y_pred[i] == 1:
            # Predicted return, actual no return --> FP
            FP[i] = 0.5 * prices[i]
        elif y_true[i] == 1 and y_pred[i] == 0:
            #predicted no return, actual return --> FN
            FN[i] = 0.5 * 5 * (3 + 0.1 * prices[i])
        else:
            # No FP or FN --> cost =  0
            FP[i] = 0
            FN[i] = 0
    
    # Calculate Total Costs
    FP_score = FP.sum() 
    FN_score = FN.sum() 
    
    #Alternative return, returns cost-matrix
    if matrix:
        return [
                [0, FN_score],
                [FP_score, 0 ]
                ]
    
    #returns the combined mean score
    return((FP_score + FN_score) / len(y_true))

# Custom Scorer Func used for CV and some other algorithms.
# Calls the regular calculate func above, only passing in slightly different input parameters
# since they have to be in a specific format
def custom_cost_scorer(estimator, X, y):    

    # generate predictions
    y_pred = estimator.predict(X)
    y_true = list(y)
    
    # make prices array to calculate costs
    # if clause since some algorithms pass in 'X' as numpy array and not as pandas df 
    if isinstance(X, pd.DataFrame):
        prices= list(X['item_price'])
    else: 
        prices = X[:, feature_names.index('item_price')]
        
    
    # calculate score
    score = calc_custom_cost_score(y_true, y_pred, prices)
    
    # return -1* score since default for scorer is the higher the better
    return (-1 * score  )

#custom eval metric, used by some algorithms for early stopping to avoid overfitting
def custom_cost_eval_xgb(y_pred, y_true):
    # initiate true y label and prices arrays
    y_true = y_true.get_label()
    y_pred = np.where(y_pred >= 0.5, 1, 0)
    prices = list(x_val['item_price'])
    score = calc_custom_cost_score(y_true, y_pred, prices)
    return ("Cost_Score", score)
    
def custom_cost_eval_lgbm(y_true, y_pred):

    y_pred = np.where(y_pred >= 0.5, 1, 0)
    prices = list(x_val['item_price'])
    
    score = calc_custom_cost_score(y_true, y_pred, prices)
    return ("Cost_Score", score, False)


# Cat Boost requires a custom CostMetric Class to evaluate results
class CostMetric:
    
    @staticmethod
    def get_profit(y_true, y_pred):
        y_pred = np.where(y_pred >= 0.5 ,1, 0)
        y_true = np.array(y_true)
        costs = calc_custom_cost_score(y_true, y_pred, list(x_train['item_price']))
        return costs
    
    def is_max_optimal(self):
        return False # lower is better

    def evaluate(self, approxes, target, weight):            
        assert len(approxes) == 1
        assert len(target) == len(approxes[0])
        y_true = np.array(target).astype(int)
        approx = approxes[0]
        score = self.get_profit(y_true, approx)
        return score, 1

    def get_final_error(self, error, weight):
        return error

if __name__ == '__main__':

    # Constants

    # labels for confustion plot
    confusion_lbs = ['Item Not Returned', 'Item Returned']

    #Random Seed Constant
    random_seed = 420


    #set numpy random seed 
    np.random.seed(random_seed)


    warnings.filterwarnings('ignore')


    # ## 1. Import and Prepare Data 

    # ### Import Data

    # Read processed data
    df = pd.read_pickle('../../data/03_processed/BADS_WS2021_known_processed_all_cols.pkl')
    df_unknown = pd.read_pickle('../../data/03_processed/BADS_WS2021_unknown_processed_all_cols.pkl')

    # Make a copy of the df to prepare data differently for the catboost algorithm
    df_cat_boost = df.copy()
    df_unknown_cat_boost = df_unknown.copy()
    # ### Data Prep and Train / Test Split

    df = drop_cols(df)
    df_unknown = drop_cols(df_unknown)

    X = df.loc[:, df.columns != 'return']
    y = df['return']
    categorical_columns = list(X.select_dtypes(include='category').columns)
    numeric_columns = list(X.select_dtypes(exclude='category').columns)

    # Make Test and Train Sets
    data_splits = train_test_split(X, y, test_size=0.3, random_state=random_seed)
    x_train, x_test, y_train, y_test = data_splits

    # Make valuation Set
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25, random_state=random_seed) # 0.25 x 0.7 = 0.175

    # check shapes of test / val/  train split
    list(map(lambda x: x.shape, [X, y, x_train, x_val, x_test, y_train, y_val, y_test]))

    for cat in categorical_columns:
        x_train = encode_and_bind(x_train, cat)
        x_test = encode_and_bind(x_test, cat)
        x_val = encode_and_bind(x_val, cat)
        df_unknown = encode_and_bind(df_unknown, cat)

    ## Save feature names after one-hot encoding for feature importances plots
    feature_names = list(x_train.columns.values)

    #save eval set
    eval_set = [(x_val, y_val)]


    # #### CatBoost Data Prep
    # catboost labels to drop
    labels=['delivery_date',
            'order_date',
            'item_id',
            'item_size',
            'item_color',
            'brand_id',
            'user_id',
            'user_dob',
            'user_reg_date',
            'is_first_purchase'
        ]

    df_cat_boost = df_cat_boost.drop(labels=labels, axis=1)
    df_unknown_cat_boost = df_unknown_cat_boost.drop(labels=labels, axis=1)

    # For CatBoost 
    X_cat = df_cat_boost.loc[:, df_cat_boost.columns != 'return']
    y_cat = df_cat_boost['return']

    categorical_columns_cat = list(X_cat.select_dtypes(include='category').columns)
    numeric_columns_cat = list(X_cat.select_dtypes(exclude='category').columns)

    data_splits = train_test_split(X_cat, y_cat, test_size=0.3, random_state=random_seed)

    x_train_cat, x_test_cat, y_train_cat, y_test_cat = data_splits

    x_train_cat, x_val_cat, y_train_cat, y_val_cat = train_test_split(x_train_cat, y_train_cat, test_size=0.25, random_state=random_seed) # 0.25 x 0.7 = 0.175
    eval_set_cat = [(x_val_cat, y_val_cat)]

    feature_names_cat = list(x_train_cat.columns.values)

    # ## 2. Modelling Helper Functions

    # # 3.  Model Training 

    # ## 3.1  Logistic Regression
    logit_cv = LogisticRegressionCV(cv=5,
                                    n_jobs=-1,
                                    random_state=random_seed,
                                    refit=True,
                                    scoring=custom_cost_scorer,
                                )

    logit_cv.fit(x_train, y_train)
    logit_report = report(logit_cv, x_train, y_train,
                                    x_test, y_test,
                                    importance_plot=True,
                                    feature_labels=feature_names,
                                    confusion_labels=confusion_lbs,
                                    verbose = False)

    # ## 3.2  Random Forests

    # ### Training

    #Optimal Parameters were extemely overfitting, manually adjusted them to avoid overfitting 
    random_forest = RandomForestClassifier(n_estimators=1800,
                                        max_features=20,
                                        max_depth=10,
                                        min_samples_leaf=2,
                                        bootstrap=True,
                                        n_jobs = -1,
                                        random_state = random_seed,
                                        )

    random_forest.fit(x_train, y_train)

    random_forest_report = report(random_forest, x_train, y_train,
                                                x_test, y_test,
                                                importance_plot=True,
                                                feature_labels=feature_names,
                                                confusion_labels=confusion_lbs,
                                                verbose = False)

    # ## 3.3  XGBoost

    # ### Training
    xgb_clf = XGBClassifier( 
                            colsample_bytree = 0.805259699363398,
                            learning_rate = 0.06774261210446376,
                            max_depth = 6,
                            min_child_weight = 4,
                            n_estimators=972,
                            subsample= 0.9653874758010768,
                            n_jobs=-1,
                            random_state=random_seed,
                            objective = 'binary:logistic', #2 classifications return <--> no return 
                            disable_default_eval_metric = 1 
                        )

    xgb_clf.fit(x_train,
                y_train,
                early_stopping_rounds = 20, # Early Stopping to avoid overfitting
                eval_metric=custom_cost_eval_xgb,
                eval_set=eval_set,
                verbose = False);

    xgb_report = report(xgb_clf, x_train, y_train,
                                x_test, y_test,
                                importance_plot=True,
                                feature_labels=feature_names,
                                confusion_labels=confusion_lbs,
                                verbose = False)


    # ## 3.4  LightGBM

    # ### Training
    lgbm_clf = LGBMClassifier(random_state=random_seed,
                            n_jobs=-1,
                            colsample_bytree =  0.5331507306847113,
                            min_child_samples= 163,
                            min_child_weight = 0.1,
                            num_leaves = 42,
                            reg_alpha = 0.1,
                            reg_lambda=0,
                            subsample =  0.30291861982300505,
                        )
    lgbm_clf.fit(x_train, y_train, eval_set = eval_set, eval_metric = custom_cost_eval_lgbm, verbose=False);

    lgbm_report = report(lgbm_clf, x_train, y_train,
                                x_test, y_test,
                                importance_plot=True,
                                feature_labels=feature_names,
                                confusion_labels=confusion_lbs,
                                verbose = False)

    # ## 3.5  CatBoost
    catboost_clf = CatBoostClassifier(cat_features=categorical_columns_cat,
                                    learning_rate=0.03,
                                    l2_leaf_reg=10,
                                    iterations = 1000,
                                    depth = 9,
                                    border_count= 20,
                                    allow_writing_files=False,
                                    silent=True,
                                    use_best_model=True,
                                    random_state=random_seed,
                                    eval_metric=CostMetric())

    catboost_clf.fit(x_train_cat, y_train, 
                    eval_set=eval_set_cat,
                    logging_level = 'Silent',
                    early_stopping_rounds=20,
                    
                )


    f_labels = feature_names_cat
    catboost_report = report(catboost_clf, x_train_cat, y_train_cat,
                                        x_test_cat, y_test_cat,
                                        importance_plot=True,
                                        feature_labels=f_labels,
                                        confusion_labels=confusion_lbs,
                                        verbose = False)

    # # 4.  Model Comparison and Evaluation
    report_list = [logit_report,                 
                random_forest_report, 
                xgb_report, 
                lgbm_report, 
                catboost_report
                ]

    compare_table = compare_models(y_test, clf_reports=report_list)
    compare_table.sort_values(by=['Total Cost'], inplace = True)
    print(compare_table)

    # save reults
    now = datetime.now()

    compare_table.to_csv(f'../../data/06_reporting/Comparison_Table_{now}.csv')

    # get best cut_off
    best_cutoff = compare_table['Optimal Cutoff'][0]
    best_cutoff

    # predict prbabilities for unlabelled dataset 
    pred_unknown = lgbm_clf.predict_proba(df_unknown)[:,1]

    # generate discrete predictions using the best_cutoff
    discrete_pred_unknown = np.where(pred_unknown >= best_cutoff, 1, 0)

    #save as pandas series in correct format
    predictions = pd.Series(discrete_pred_unknown, index=df_unknown.index, name='return')

    # Sanity checks to see if predictions are discrete
    print(predictions.unique())
    print(len(predictions))
    print(predictions)

    # save predictions to CSV
    predictions.to_csv('../../results/predictions.csv')

    #second sanity check to confirm data can is loaded correctly 
    test_import = pd.read_csv('../../results/predictions.csv', index_col=0)
    print(test_import)

