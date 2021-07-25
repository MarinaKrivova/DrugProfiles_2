import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")


def transpose_cell_features(df, indexes_first_column=True):
    """
    Returns transformed dataframe with indexes CCL_id
    """
    # columns in cell_features correspond to drug_curves["COSMIC_ID"] == cell line id
    if indexes_first_column:
        df_transfromed = pd.DataFrame(
            data=df[df.columns[1:]].values.T,
            index=df.columns[1:],
            columns=df[df.columns[0]].values,
        )
    else:
        print("The logic of this function is not applicable")

    return df_transfromed


def split_fit_param(df, param_col_name):
    """
    Returns dataframe where column fitting_param is splitted into separate columns
    """

    df = df[df[param_col_name].isnull() == False].copy()
    n_params = len(df[param_col_name].values[0])

    for i in range(n_params):
        df["param_" + str(i + 1)] = df[param_col_name].apply(lambda x: x[i])
    return df


def merge_drug_cells(
    df_drug_curves,
    df_cells,
    df_drug_properties,
    splitting_fitting_column_needed=False,
    param_col_name="fitting_param",
    save_CCL_properties=False,
    _FOLDER_to_save=None,
):
    """
    Returns dataframe with merged CCL and drug properties
    """

    cell_features_T = transpose_cell_features(df_cells)
    cell_features_T.index = np.array(cell_features_T.index, dtype="int")

    # Not all the drugs from filtered dataset are present in cell lines features
    common_cells_drug = list(
        set(np.array(df_cells.columns[1:], dtype="int"))
        & set(df_drug_curves["COSMIC_ID"].values)
    )

    # print("Number of drugs in filtered dataset:", df_328["COSMIC_ID"].nunique())
    # print("Number of common drugs in both datasets", len(common_cells_drug328))

    cell_lines = cell_features_T.loc[common_cells_drug, :].reset_index()
    cell_lines.rename(columns={"index": "COSMIC_ID"}, inplace=True)

    df_drug_curves = pd.merge(
        left=df_drug_curves, right=df_drug_properties, on="DRUG_ID"
    )

    if splitting_fitting_column_needed:
        df_drug_curves = split_fit_param(df_drug_curves, param_col_name)

    # merge drug profile data (fitted parameters) and cell line features
    if save_CCL_properties:
        with open(_FOLDER_to_save + "X_features_cancer_cell_lines.txt", "w") as f:
            for s in cell_lines.columns[1:]:
                f.write(str(s) + "\n")

    return pd.merge(left=df_drug_curves, right=cell_lines, on="COSMIC_ID")


def split_train_test_for_10_drugs(df, train_ratio=0.8):
    """Find drugs with more than 10 drug profiles
    for each drugs split data into train and test and then concat train and test"""
    gr = df.groupby("DRUG_ID").size()
    drugs = gr[gr > 10].index

    train = pd.DataFrame()
    test = pd.DataFrame()
    np.random.seed(123)

    if "DRUGID_COSMICID" in df.columns:
        df.set_index("DRUGID_COSMICID", inplace=True)

    for drug_id in drugs:
        df_i = df[df["DRUG_ID"] == drug_id]
        indexes = np.random.permutation(df_i.index)
        train_size = int(df_i.shape[0] * train_ratio)
        indexes_train = indexes[:train_size]
        indexes_test = indexes[train_size:]

        train = pd.concat([train, df_i.loc[indexes_train, :]])
        test = pd.concat([test, df_i.loc[indexes_test, :]])

    print("Number of samples for ML modelling:", df.shape[0])
    print("Number of drugs with more than 10 profiles:", len(drugs))
    print(
        "Number of drug profiles not covered:",
        df.shape[0] - train.shape[0] - test.shape[0],
    )

    test2_index = set(df.index) - set(train.index) - set(test.index)
    test2 = df.loc[test2_index, :]
    return train, test, test2
