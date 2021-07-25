import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os, sys
sys.path.insert(1, os.path.dirname("functions"))  
from fitting import *


def content_plotting():
    """
    All the functions in the current file
    """
    all_functions = {
        "show_response_curves": "Plots multiple figures (plots_in_row, plots_in_column)in the range of indexes",
        "show_specific": "Plots multiple for samples specified in drug_CCL_list [(drug_name, CCL_name)] or [(drug_id, CCL_id)]",
        "one_fig_no_fitting": "Plots one figure to show or save",
        "show_response_curves_with_fitting": "Shows multiple plots with fitting",
        "show_one_fitting": "plots one plot with fitting",
    }
    return all_functions


def show_response_curves(
    df,
    plots_in_row,
    plots_in_column,
    x_columns,
    y_columns,
    start_index=0,
    indexes=[],
    drug_dict=None,
    CCL_dict=None,
    upper_limit=None,
    lower_limit=None,
):

    """
    Plots several figures (plots_in_row, plots_in_column)in the range of indexes
    Option of red horisontal lines
    """

    fig = plt.figure(figsize=(14, 11))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    n_plots = plots_in_row * plots_in_column

    if len(indexes) == 0:
        indexes = df.index[start_index : start_index + n_plots]

    for i, ind in list(enumerate(indexes)):
        x = df.loc[ind, x_columns].values.astype(np.float32)
        y = df.loc[ind, y_columns].values.astype(np.float32)

        ax = fig.add_subplot(plots_in_row, plots_in_column, i + 1)

        if max(y) > 1:
            max_y = max(y) + 0.1
        else:
            max_y = 1.1
        ax.set_ylim([0, max_y])
        ax.scatter(x, y)

        if drug_dict and CCL_dict:
            ax.set_title(
                "Drug: "
                + drug_dict[df.loc[ind, "DRUG_ID"]]
                + " / CCL: "
                + CCL_dict[df.loc[ind, "COSMIC_ID"]]
            )
        elif drug_dict:
            ax.set_title(
                "Drug: "
                + drug_dict[df.loc[ind, "DRUG_ID"]]
                + "_"
                + str(df.loc[ind, "COSMIC_ID"])
            )

        else:
            ax.set_title(
                str(df.loc[ind, "DRUG_ID"]) + "_" + str(df.loc[ind, "COSMIC_ID"])
            )
        ax.set_xlabel("Scaled dosage")
        ax.set_ylabel("Normalised response")

        if upper_limit:
            ax.axhline(upper_limit, color="red", ls="--")
        if lower_limit:
            ax.axhline(lower_limit, color="black", ls="--")


def show_specific(
    df,
    drug_CCL_list,
    x_columns,
    y_columns,
    drug_col="drug_name",
    CCL_col="CCL_name",
    upper_limit=None,
    lower_limit=None,
):

    """
    Plots multiple for samples specified in drug_CCL_list [(drug_name, CCL_name)] or [(drug_id, CCL_id)]
    Option of red horisontal lines
    """

    n_plots = len(drug_CCL_list)

    if n_plots <= 4:
        fig = plt.figure(figsize=(5 * n_plots, 3))  # 3 is height
    else:
        print("Too many samples")

    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    i = 0
    for (drug, CCL) in drug_CCL_list:
        ind = df[(df[drug_col] == drug) & (df[CCL_col] == CCL)].index

        x = df.loc[ind, x_columns]
        y = df.loc[ind, y_columns].values[0]  # possible problems are here

        ax = fig.add_subplot(1, n_plots, i + 1)

        if max(y) > 1:
            max_y = max(y) + 0.1
        else:
            max_y = 1.1
        ax.set_ylim([0, max_y])

        ax.scatter(x, y)

        ax.set_title("Drug: " + str(drug) + " / CCL: " + str(CCL))
        ax.set_xlabel("Scaled dosage")
        ax.set_ylabel("Normalised response")
        i += 1

        if upper_limit:
            ax.axhline(upper_limit, color="red", ls="--")
        if lower_limit:
            ax.axhline(lower_limit, color="black", ls="--")


def one_fig_no_fitting(
    df,
    drug_id,
    ccl_name,
    x_columns,
    y_columns,
    size=8,
    dpi=300,
    upper_limit=None,
    lower_limit=None,
    save_fig_name=None,
):
    """
    Plots one figure to show or save,
    Options for red horizantal lines
    """

    ind = df[(df["DRUG_ID"] == drug_id) & (df["CELL_LINE_NAME"] == ccl_name)].index
    drug_name = df.loc[ind, "drug_name"].values[0]

    print(
        f"Drug: {drug_name} ({drug_id}) / CCL: {ccl_name}"
    )  # % drug_name +str(drug_id) +" / CCL: "+ str(ccl_name))
    x = df.loc[ind, x_columns]
    y = df.loc[ind, y_columns].values[0]  # possible problems are here

    plt.figure(figsize=(size, size))
    if max(y) > 1:
        max_y = max(y) + 0.1
    else:
        max_y = 1.1
    plt.ylim([0, max_y])
    plt.scatter(x, y)

    plt.xlabel("Scaled dosage")
    plt.ylabel("Normalised response")
    if upper_limit:
        plt.axhline(upper_limit, color="red", ls="--")
    if lower_limit:
        plt.axhline(lower_limit, color="black", ls="--")

    plt.tick_params(labelsize=14)
    plt.xlabel("Scaled dosage", fontsize=14)
    plt.ylabel("Normalised Response", fontsize=14)
    if save_fig_name:

        plt.savefig(save_fig_name, bbox_inches="tight", dpi=dpi)
        plt.show()
    else:
        plt.show()


def show_response_curves_with_fitting(
    df,
    plots_in_row,
    plots_in_column,
    x_columns,
    y_columns,
    start_index=0,
    indexes=[],
    fitting_function=None,
    fitting_parameters=None,
    pred_fitting_param=None,
    drug_dict=None,
    CCL_dict=None,
):
    """
    Shows multiple plots with fitting,
    main parameters: (plots_in_row, plots_in_column,
                    indexes=[], fitting_function)
    """

    print_general_title = False
    fig = plt.figure(figsize=(14, 11))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    n_plots = plots_in_row * plots_in_column

    if len(indexes) == 0:
        indexes = df.index[start_index : start_index + n_plots]

    for i, ind in list(enumerate(indexes)):

        x = df.loc[ind, x_columns].values.astype(np.float32)
        y = df.loc[ind, y_columns].values.astype(np.float32)

        ax = fig.add_subplot(plots_in_row, plots_in_column, i + 1)
        ax.scatter(x, y)
        if (drug_dict is not None) and (CCL_dict is not None):
            ax.set_title(
                "Drug: "
                + drug_dict[df.loc[ind, "DRUG_ID"]]
                + " / CCL: "
                + CCL_dict[df.loc[ind, "COSMIC_ID"]]
            )
        elif drug_dict is not None:
            ax.set_title(
                "Drug: "
                + drug_dict[df.loc[ind, "DRUG_ID"]]
                + "_"
                + str(df.loc[ind, "COSMIC_ID"])
            )
        elif ("drug_name" in df.columns) and ("CCL_name" in df.columns):
            ax.set_title(
                "Drug: "
                + str(df.loc[ind, "drug_name"])
                + " / CCL: "
                + str(df.loc[ind, "CCL_name"])
            )
        else:
            print_general_title = True
            ax.set_title(
                str(df.loc[ind, "DRUG_ID"]) + "_" + str(df.loc[ind, "COSMIC_ID"])
            )

        ax.set_xlabel("Scaled dosage")
        ax.set_ylabel("Normalised response")

        if fitting_function is not None:
            fitting_function_object = sigmoid_function(fitting_function)

            x2 = np.linspace(0, 1, 10)

            if type(fitting_parameters) == str:
                fit_param = df.loc[ind, fitting_parameters]
                if len(fit_param) > 5:
                    fit_param = [
                        float(i) for i in fit_param.strip("[ ").strip(" ]").split()
                    ]
            else:
                fit_param = df.loc[ind, fitting_parameters].values

            y_fit = fitting_function_object(x, *fit_param)
            y2 = fitting_function_object(x2, *fit_param)

            r2 = r2_score(y, y_fit)
            ax.plot(x2, y2, label="R^2 fit = %0.4f" % r2)
            ax.legend()

        if pred_fitting_param is not None:
            x3 = np.linspace(0, 1, 10)
            fit_param = df.loc[ind, pred_fitting_param]
            y_fit3 = fitting_function_object(x, *fit_param)
            y3 = fitting_function_object(x3, *fit_param)
            r2_pred = r2_score(y, y_fit3)
            ax.plot(x3, y3, color="red", label="R^2 pred = %0.4f" % r2_pred)
            ax.legend()

    if print_general_title is not None:
        print("Figures titles: Index_DRUG_ID_COSMIC_ID (COSMIC_ID is a cell line)")


def show_one_fitting(
    df,
    ind,
    conc_columns,
    response_norm,
    fitting_function,
    fitting_parameters,
    predicted_param=None,
    save_fig_name=None,
    fig_size=None,
):
    """
    Plots one plot with fitting,
    main parameters:  ind, fitting_function, fitting_parameters,
                      save_fig_name, fig_size
    """

    fitting_function_object = sigmoid_function(fitting_function)

    if fig_size is not None:
        plt.figure(figsize=fig_size)
    else:
        plt.figure(figsize=(4, 4))
    x = df.loc[ind, conc_columns].astype("float32")
    y = df.loc[ind, response_norm].astype("float32")
    plt.scatter(x, y)
    plt.tick_params(labelsize=14)
    plt.xlabel("Scaled dosage", fontsize=14)
    plt.ylabel("Normalised Response", fontsize=14)
    fit_param = df.loc[ind, fitting_parameters]

    y_fit = fitting_function_object(x, *fit_param)
    r2 = r2_score(y, y_fit)
    plt.plot(x, y_fit, label="fitting: R2= " + str(round(r2, 3)))
    if predicted_param is not None:
        print("Fitting parameters:", fit_param)
        print("Predicted parameters:", predicted_param)
        y_fit = sigmoid_4_param(x, *predicted_param)
        r2 = r2_score(y, y_fit)
        plt.plot(x, y_fit, label="predicted : R2= " + str(round(r2, 3)))

    plt.legend()
    if save_fig_name:
        plt.savefig(save_fig_name, bbox_inches="tight", dpi=300)
