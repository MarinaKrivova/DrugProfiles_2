import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
from tqdm import tqdm

import os, sys
sys.path.insert(1, os.path.dirname("functions"))  
from sigmoids import *

import warnings
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"


def content_fitting():
    """
    All the functions in the current file
    """

    all_functions = {
        "sigmoid_function": "return sigmoid function for fitting or plotting",
        "fit_curve": "returns values r2 score and fitting parameters for a single sample",
        "fitting_column": "returns r2_scores, fitting_parameters",
        "compute_r2_score": "returns array of r2_scores",
        "compute_fitting_function": "returns df with columns \
                                       [fitting_function+'_r2'] and fitting_function",
        "compare_fitting_functions": "returns df with columns better_fitting, \
                                        displays df_best.loc[fitting_function, min, max, \
                                        r2>0, r2>0.8, r2>0.9, r2>0.99",
    }
    return all_functions


def sigmoid_function(function_name):
    """
    Returns sigmoid function for fitting or plotting
    """
    functions = {
        "fsigmoid": fsigmoid,
        "sigmoid_2_param": sigmoid_2_param,
        "sigmoid_4_param": sigmoid_4_param,
        "sigmoid_3_param": sigmoid_3_param,
        "logistic_4_param": logistic_4_param,
        "ll4_4_param": ll4_4_param,
        "ll4R_4_param": ll4R_4_param,
        "logLogist_3_param": logLogist_3_param,
    }
    return functions[function_name]


def fit_curve(fitting_function, x, y, parameters_guess=None, to_plot=False):
    """
    Returns values r2, parameters for a single sample
    found by curve_fit method from scipy.optimize
    There is also an option to_plot results
    """

    if parameters_guess is not None:
        parameters, p_covariance = curve_fit(fitting_function, x, y, parameters_guess)
    else:
        parameters, p_covariance = curve_fit(fitting_function, x, y)
    x2 = np.linspace(0, 1, 10)
    y_fit = fitting_function(x, *parameters)
    r2 = r2_score(y, y_fit)

    if to_plot:
        print("Fitting parameters:", *parameters)
        plt.scatter(x, y)
        x2 = np.linspace(0, 1, 10)
        y2 = fitting_function(x2, *parameters)
        plt.plot(x2, y2, "blue", label="R^2= %0.5f" % r2)
        plt.title("Least-squares fit")
        plt.legend()
    return r2, parameters


def fitting_column(
    df,
    indexes,
    x_columns,
    y_columns,
    fitting_function,
    parameters_guess=None,
    default_param=None,
):
    """
    Returns arays of r2 scores and fitting parameters
    intial parameter guess [max(y), np.median(x), 1, min(y)]
    potentially they can be different for each data row,
    but as soon as we have scaled and filtered data
    we can use by default [1.0, 0.4, 1.0, .0]
    """

    r2_scores = np.zeros(len(indexes))
    X = df.loc[indexes, x_columns].values.astype(np.float32)
    Y = df.loc[indexes, y_columns].values.astype(np.float32)
    fitting_parameters = [None] * len(indexes)

    # parameters_guess= [np.median(x), 1, max(y), min(y)]
    default_param_model = {
        "sigmoid_2_param": [0.4, 0.1],
        "fsigmoid": [0.4, -10],
        "sigmoid_4_param": [0.4, 1.0, 1.0, 0.0],
        "sigmoid_3_param": [0.4, 1.0, 0.0],
        "logistic_4_param": [1.0, 1.0, 1.0, 0.0],
        "ll4_4_param": [0.4, 1.0, 1.0, 0.0],
        "ll4R_4_param": [0.4, 1.0, 1.0, 0.0],
        "logLogist_3_param": [-1, -0.1, 0.1],
    }

    if default_param is not None:
        parameters_guess = default_param_model[fitting_function]

    else:
        pass

    for i in tqdm(range(len(indexes))):
        x = X[i, :]
        y = Y[i, :]

        try:
            r2_scores[i], fitting_parameters[i] = fit_curve(
                fitting_function_object, x, y, parameters_guess=parameters_guess
            )
        except:
            try:
                fitting_function_object = sigmoid_function(fitting_function)
                r2_scores[i], fitting_parameters[i] = fit_curve(
                    fitting_function_object, x, y, parameters_guess=parameters_guess
                )
            except:
                r2_scores[i] = 0

    print(fitting_function_object)
    return r2_scores, fitting_parameters


def compute_fitting_function(df, fitting_function, x_columns, y_columns, drop_nulls=False):
    """
    Returns df with columns [fitting_function+'_r2'] and fitting_function fit parameters
    """
    shape_1 = df.shape[0]
    r2, fit_param = fitting_column(
        df,
        df.index,
        x_columns,
        y_columns,
        fitting_function=fitting_function,
        default_param=True,
    )
    df[fitting_function + "_r2"] = r2
    df[fitting_function] = fit_param
    if drop_nulls:
        df = df[df[fitting_function].isnull() == False]
        if df.shape[0] != shape_1:
            print("Reduced number of samples:", shape_1 - df.shape[0])
    return df


def compute_r2_score(
    df,
    x_columns,
    y_columns,
    fitting_parameters,
    fitting_function="sigmoid_4_param",
    return_fit_y=False,
):
    """
    Returns an array of r2_scores
    """

    fitting_function_object = sigmoid_function(fitting_function)
    r2_scores = np.zeros(len(df.index))
    y_computed = np.zeros(len(df.index))
    for i in range(len(df.index)):
        x = df.loc[df.index[i], x_columns].astype("float32").values
        y = df.loc[df.index[i], y_columns].astype("float32").values
        if type(fitting_parameters) == str:
            fit_param = df.loc[df.index[i], fitting_parameters]

        else:
            fit_param = df.loc[df.index[i], fitting_parameters].values

        y_fit = fitting_function_object(x, *fit_param)
        r2_scores[i] = r2_score(y, y_fit)
        df.loc[df.index[i], "pred_fit_r2"] = r2_score(y, y_fit)
        if return_fit_y:
            for y in range(len(y_fit)):
                df.loc[df.index[i], "pred_y_" + str(y)] = y_fit[y]

    if return_fit_y:
        return df
    else:
        return r2_scores


def compare_fitting_functions(
    df,
    functions,
    conc_columns,
    response_norm,
    recompute_fitting=True,
    save_file_name=None,
):
    """
    Returns df with columns better_fitting,
    displays df_best.loc[fitting_function, min, max,
    r2>0, r2>0.8, r2>0.9, r2>0.99
    """
    print(df.shape)
    if recompute_fitting:
        for fitting_function in functions:
            print("\n", fitting_function)
            compute_fitting_function(df, fitting_function, conc_columns, response_norm)

    functions_dict = dict(list(enumerate(functions)))
    r2_columns = [fitting_function + "_r2" for fitting_function in functions]

    df["better_fitting"] = np.argmax(df[r2_columns].values, axis=1)
    r2_col_res = r2_columns + ["better_fitting"]
    df["better_fitting"] = df["better_fitting"].map(functions_dict)
    # df[r2_col_res].head()

    print("\n")
    best_functions = df["better_fitting"].unique()

    df_best = pd.DataFrame(index=functions)
    for fitting_function in functions:
        r2_fit = df[fitting_function + "_r2"].values
        try:
            df_best.loc[fitting_function, "best_fitting_count"] = df[
                df["better_fitting"] == fitting_function
            ].shape[0]
        except:
            df_best.loc[fitting_function, "best_fitting_count"] = 0
        df_best.loc[fitting_function, "min"] = min(r2_fit)
        df_best.loc[fitting_function, "max"] = max(r2_fit)
        df_best.loc[fitting_function, "r2>0"] = (r2_fit > 0).sum().astype("int32")
        df_best.loc[fitting_function, "r2>0.8"] = (r2_fit > 0.8).sum().astype("int32")
        df_best.loc[fitting_function, "r2>0.9"] = (r2_fit > 0.9).sum().astype("int32")
        df_best.loc[fitting_function, "r2>0.99"] = (r2_fit > 0.9).sum().astype("int32")
    display(df_best)
    print(
        "\nExamples of bad fitting with sigmoid_4_param (r2<0.61):",
        df[df["sigmoid_4_param_r2"] < 0.61].shape[0],
    )
    display(
        df[df["sigmoid_4_param_r2"] < 0.61][
            ["COSMIC_ID", "DRUG_ID"] + r2_col_res
        ].head()
    )
    if save_file_name is not None:
        df_best.to_csv(save_file_name, index=False)
    return df
