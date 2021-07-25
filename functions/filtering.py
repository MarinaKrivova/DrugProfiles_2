import pandas as pd
import numpy as np
from sklearn.metrics import auc
from scipy.stats import spearmanr
from tqdm import tqdm

import os, sys
sys.path.insert(1, os.path.relpath("functions"))        
from fitting import compute_fitting_function


def content_filtering():
    """
    All the functions in the current file
    """

    all_functions = {
        "find_high_responses": "Returns df with normalised responses >1, df['high_responses'] = df[response_cols].apply(lambda row: sum(row>1), axis=1)",
        "cut_off_outliers": "Returns filtered df without ascending points, middle_points_limit=-0.2",
        "find_ascending_data": "Returns df with acending points, middle_points_limit=-0.2",
        "filtering_sigmoid_curves": "Filtering_scenario I = filtering [1,2,3,4]",
        "auc_filtration": "Filtering scenario II - AUC>0.7, negative Spearman correlation coefficients",
        "filter_good_response": "Returns df without missing values",
        "select_group_limits": "Returns df with min_N_points normalised responses above lower_limit_response",
        "select_group_1": "Returns df with all normalised responses above 1",
        "select_group_1a": "Returns df with all normalised responses above y_lower_limit",
        "select_group_1b": "Retuns df with S-shaped samples based on the fitting with some sigmoid functions - time >25 min",
        "select_group_2": "Returns df with all normalised responses >=1",
        "select_group_2a": "Returns df with last response above y_limit",
        "select_group_2b": "Returns df with S-shape samples based on the fitting with some sigmoid function - time >15 min",
    }
    return all_functions


def select_group_limits(df, response_columns, lower_limit_response=1, min_N_points=1):
    """
    Returns df with min_N_points normalised responses above lower_limit_response
    """
    return df[
        np.sum(df[response_columns] > lower_limit_response, axis=1) >= min_N_points
    ]


def select_group_1(df, response_columns):
    """
    Returns df with all normalised responses above 1
    """
    return select_group_limits(df, response_columns, 1, 1)


def select_group_1a(df, response_columns, y_lower_limit=0.9):
    """
    Returns df with all normalised responses above y_lower_limit,
    input df should be group_1
    """
    return select_group_limits(
        df,
        response_columns,
        y_lower_limit,
        len(response_columns) - df[response_columns].isnull().sum(axis=1),
    )


def select_group_1b(
    df, functions, conc_columns, response_columns, y_limit=0.9, r2_limit=0.9
):
    """
    Retuns df with S-shaped samples based on the fitting with some sigmoid functions - time >25 min
    """

    gr_1 = select_group_1(df, response_columns)
    gr_1a = select_group_1a(df, response_columns, y_limit)
    df2 = gr_1.loc[(list(set(gr_1.index) - set(gr_1a.index)))]

    for fitting_function in functions:
        print("\n", fitting_function)
        df2 = compute_fitting_function(
            df2, fitting_function, conc_columns, response_columns
        )

    r2_columns = [fitting_function + "_r2" for fitting_function in functions]

    return df2[np.sum(df2[r2_columns] > r2_limit, axis=1) > 0]


def select_group_2(df, response_columns):
    """
    Returns df with all normalised responses >=1
    """
    return df[np.sum(df[response_columns] > 1, axis=1) == 0]


def select_group_2a(df, response_columns, y_lower_limit):
    """
    Returns df with last response above y_limit
    """

    gr_2 = select_group_2(df, response_columns)
    gr_2["last_response"] = gr_2[response_columns[-1]]
    ind_nan = gr_2[gr_2["last_response"].isnull()].index
    gr_2.loc[ind_nan, "last_response"] = gr_2.loc[ind_nan, response_columns[5]]

    return gr_2[gr_2["last_response"] > y_lower_limit]


def select_group_2b(
    df, functions, conc_columns, response_columns, y_lower_limit=0.5, r2_limit=0.9
):
    """
    Returns df with S-shape samples based on the fitting with some sigmoid function
    time > 15 min
    """

    gr_2 = select_group_2(df, response_columns)
    gr_2a = select_group_2a(df, response_columns, y_lower_limit)

    df2 = gr_2.loc[(list(set(gr_2.index) - set(gr_2a.index)))]

    for fitting_function in functions:
        print("\n", fitting_function)
        compute_fitting_function(df2, fitting_function, conc_columns, response_columns)

    r2_columns = [fitting_function + "_r2" for fitting_function in functions]

    return df2[np.sum(df2[r2_columns] > r2_limit, axis=1) > 0]


def find_high_responses(df, response_cols):
    """
    Returns df with normalised responses >1
    df['high_responses'] = df[response_cols].apply(lambda row: sum(row>1), axis=1)
    """

    df = df.copy()
    df["high_responses"] = (df[response_cols]> 1).sum(axis=1)
    df_bad = df[df["high_responses"] > 1]
    return df_bad


def cut_off_outliers(df, response_columns, middle_points_limit=-0.2):
    """
    Returns filtered df without ascending points, middle_points_limit=-0.2
    """
    df = df.copy()
    bad_index = []
    for j in range(
        1, len(response_columns) - 1
    ):  # two first and two last are already assessed
        bad_index.extend(
            df[
                (df[response_columns[j]] - df[response_columns[j + 1]])
                <= middle_points_limit
            ].index
        )
    index_to_use = list(set(df.index) - set(bad_index))
    return df.loc[index_to_use, :]


def find_ascending_data(df, response_columns, middle_points_limit=-0.2):
    """
    Returns df with acending points, middle_points_limit=-0.2
    """
    df = df.copy()
    bad_index = []
    for j in range(
        1, len(response_columns) - 1
    ):  # two first and two last are already assessed
        bad_index.extend(
            df[
                (df[response_columns[j]] - df[response_columns[j + 1]])
                < middle_points_limit
            ].index
        )

    index_to_use = list(set(bad_index))
    return df.loc[index_to_use, :]


def filtering_sigmoid_curves(
    df,
    response_columns,
    filtering_scenario=[1, 2, 3],
    first_columns_to_compare=[1, 2],
    last_columns_to_compare=[-1, -2],
    tolerance=0.05,
    first_points_lower_limit=0.8,
    last_points_upper_limit=0.4,
    middle_points_limit=-0.2,
):
    """
    filtering_scenario = [1,2,3,4]
    1. Ensure that all the response are less than 1

    2. Ensure that first and last points form plateus
    the minimal number of points are specified in the function arguments
    by default, two points for both lpateus are considered
    tolerance =0.05 values to ensure the points form a plateu
    first_columns_to_compare = [1, 2]  - first two columns for plateu
    last_columns_to_compare = [-1, -2] - last two columns for plateu

    3. Specify location of the plateus - first_points_lower_limit and last_points_upper_limit

    4. Cutting off ambiqueos data:
    Among all "middle" datapoints a subsequent point should not be higher than antecedent by 0.2
    """
    df = df.copy()
    print("Original dataset:", df.shape)

    for i in filtering_scenario:
        if i == 1:
            # 1st filtering
            index_row_more_than_1 = []
            for col in response_columns:
                if sum(df[col] > 1) > 0:
                    index_row_more_than_1.extend(df[df[col] > 1].index)

            index_row_less_than_1 = set(df.index) - set(index_row_more_than_1)
            df = df.loc[index_row_less_than_1, :].copy()
            print(
                "1st filtration (Ensure that all the response are less than 1): Filtered dataset:",
                df.shape,
            )

        elif i == 2:
            # 2nd filtering
            df["dif_first"] = abs(
                df[response_columns[first_columns_to_compare[0] - 1]]
                - df[response_columns[first_columns_to_compare[1] - 1]]
            )
            df["dif_last"] = abs(
                df[response_columns[last_columns_to_compare[0]]]
                - df[response_columns[last_columns_to_compare[1]]]
            )

            df = df[(df["dif_first"] <= tolerance) & (df["dif_last"] <= tolerance)]

            print(
                "2d filtration (Ensure that first and last points form plateus): Filtered dataset:",
                df.shape,
            )
        elif i == 3:
            # 3d filtering
            df = df[
                (df[response_columns[1]] > first_points_lower_limit)
                & (df[response_columns[-1]] < last_points_upper_limit)
            ]
            print(
                "3d stage filtration (Specified location of the plateus): Filtered dataset:",
                df.shape,
            )

        elif i == 4:
            df = cut_off_outliers(df, response_columns, middle_points_limit)

            print(
                "4th stage filtration (Cut off high ancedent points): Filtered dataset:",
                df.shape,
            )

        else:
            print("Unknown filtration scenario")

    return df


def auc_filtration(
    df,
    conc_columns,
    response_columns,
    auc_limit=0.7,
    final_response_limit=None,
):
    """
    1. Remove all the curves where the normalised response value is greater than one at zero dosage.
    2. Compute the Area Under the Curve (AUC) for all the curves.
    3. Leave only those curves with an AUC>0.7.
    4. Compute the Spearman correlation coefficient between the normalised response and the scaled dosage
    (so the x-axis and the y-axis).
    5. Further remove the curves for which the Spearman correlation coefficient is zero or positive.
    6. Cut off samples with last response above final_response_limit
    """

    df = df[df["norm_cells_0"] <= 1].copy()

    for index in tqdm(df.index):
        row = df.loc[index, :]
        df.loc[index, "auc"] = auc(row[conc_columns], row[response_columns])
        df.loc[index, "spearman_r"] = spearmanr(
            row[conc_columns], row[response_columns]
        )[0]

    df = df[(df["auc"] > auc_limit) & (df["spearman_r"] < 0)].copy()
    if final_response_limit is not None:
        df = filter_good_response(
            df, response_columns, final_response_limit=final_response_limit
        )
    return df


def filter_good_response(df, response_cols, final_response_limit=0.4):
    """
    Returns df without missing values
    """
    df["count_missing"] = df[response_cols].isnull().sum(axis=1)
#     df["count_missing"].value_counts()

    ind_1 = list(
        df[
            (df["count_missing"] == 4) & (df[response_cols[5]] < final_response_limit)
        ].index
    )
    ind_2 = list(
        df[
            (df["count_missing"] == 0) & (df[response_cols[-1]] < final_response_limit)
        ].index
    )

    return df.loc[ind_1 + ind_2, :]
