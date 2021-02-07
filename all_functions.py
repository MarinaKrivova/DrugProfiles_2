import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from scipy.optimize import curve_fit
import scipy.optimize as opt
from sklearn.metrics import r2_score, mean_absolute_error

from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
from drug_features import GetPubChemId, RunManualCorrections
# from training_testing import TrainTest_KR_Alg1
import pickle

def FilteringCurves(df, response_columns, filtering_scenario = [1,2,3], 
                     first_columns_to_compare = [1, 2], last_columns_to_compare = [-1, -2],
                     tolerance=0.05, 
                     first_points_lower_limit = 0.8, 
                     last_points_upper_limit = 0.4,
                      middle_points_limit = 0.1,
                       print_progress_info= True):
    """
    filtering_scenario = [1,2,3]
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
    if print_progress_info:
        print("Original dataset:", df.shape)
    
    for i in filtering_scenario:
        if i ==1:
            #1st filtering
            index_row_more_than_1 = []
            for col in response_columns:
                if sum(df[col]>1)>0:
                    index_row_more_than_1.extend(df[df[col]>1].index)
        
            index_row_less_than_1 = set(df.index) - set(index_row_more_than_1)
            df = df.loc[index_row_less_than_1, :].copy()
            if print_progress_info:
                print("1st filtration (Ensure that all the response are less than 1): Filtered dataset:", df.shape)
        
        elif i== 2: 
            #2nd filtering
            df["dif_first"]=abs(df[response_columns[first_columns_to_compare[0]-1]]\
                                     - df[response_columns[first_columns_to_compare[1]-1]])
            df["dif_last"]=abs(df[response_columns[last_columns_to_compare[0]]] \
                                        - df[response_columns[last_columns_to_compare[1]]])

            df = df[(df["dif_first"]<= tolerance)
                           &(df["dif_last"]<= tolerance)]
            if print_progress_info:
                print("2d filtration (Ensure that first and last points form plateus): Filtered dataset:", df.shape)
        elif i== 3: 
            #3d filtering
            df = df[(df[response_columns[1]]>first_points_lower_limit) 
                         & (df[response_columns[-1]]<last_points_upper_limit)]
            
            if print_progress_info:
                print("3d stage filtration (Specified location of the plateus): Filtered dataset:", df.shape)
        
        elif i==4:
            if middle_points_limit:
                for j in range(1, len(response_columns)-2): # two first and two last are already assessed
                    df = df[(df[response_columns[j]] - df[response_columns[j+1]])>middle_points_limit]
            if print_progress_info:
                print("4th stage filtration (Cut off high ancedent points): Filtered dataset:", df.shape)
            
        else:
            if print_progress_info:
                print("Unknown filtration scenario")
    
    return df

def ShowResponseCurves(df, plots_in_row, plots_in_column, x_columns, y_columns, start_index=0, indexes=[]):
 
                
    fig = plt.figure(figsize=(14, 11))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    n_plots= plots_in_row*plots_in_column
    
    if len(indexes) ==0:
        indexes =df.index[start_index : start_index+n_plots]

    for i, ind in list(enumerate(indexes)):
        x = df.loc[ind, x_columns]
        y = df.loc[ind, y_columns]
                
        ax = fig.add_subplot(plots_in_row, plots_in_column, i+1)
        ax.scatter(x,y)
        ax.set_title(str(df.loc[ind, "DRUG_ID"])+"_"+str(df.loc[ind, "COSMIC_ID"]))
        ax.set_xlabel("Scaled dosage")
        ax.set_ylabel("Normalised response")
        
def sigmoid_Wang(x, p, s):
    """ Sigmoid function from Dennis Wang's paper:
    x - dosage [0, 1],
    p - position,        default=0.4
    s - shape parameter, default=-1
    """
    return ( 1.0 / (1.0 + np.exp((x-p)/s)) )


def fsigmoid(x, p, k):
    """ Comparing with Dennis Wang's sigmoid:
    x = x  - dosage [0, 1]
    p - position [0,1],           default=0.4
    k = -1/s (s -shape parameter) default=0.4
    """
    return ( 1.0 / (1.0 + np.exp(-k*(x-p))) )


def sigmoid_4_param(x, x0, L, k, d):
    """ Comparing with Dennis Wang's sigmoid:
    x0 -  p - position, correlation with IC50 or EC50
    L = 1 in Dennis Wang's sigmoid, protect from devision by zero if x is too small 
    k = -1/s (s -shape parameter)
    d - determines the vertical position of the sigmoid - shift on y axis - better fitting then Dennis Wang's sigmoid
    """
    return ( 1/ (L + np.exp(-k*(x-x0))) + d)


def sigmoid_3_param(x, x0, k, d):
    """ Comparing with Dennis Wang's sigmoid:
    x0 -  p - position, correlation with IC50 or EC50
    k = -1/s (s -shape parameter)
    d - determines the vertical position of the sigmoid - shift on y axis - better fitting then Dennis Wang's sigmoid
        """
    return ( 1/ (1 + np.exp(-k*(x-x0))) + d )


def ll4(x, e, c, b, d):
    """ https://gist.github.com/yannabraham/5f210fed773785d8b638
    This function is basically a copy of the LL.4 function from the R drc package with
     - b: hill slope
     - d: min response - determines the vertical position of the graph
     - c: max response
     - e: EC50
     c-d - difference between max and min responses
     np.exp( b* (np.log(x)-np.log(e)) -  np.exp((x-p)/s in Dennis Wang's sigmoid
     b- hill slope = 1/s - shape parameter
     np.log(x)-np.log(e) == x-p in Dennis Wang's sigmoid
     """
    return ( (c-d)/(1 + np.exp( b*(np.log(x)-np.log(e) ))) + d)


def ll4_R(x, e, c, b, d):
    """ LL.4 function from R
    https://www.rdocumentation.org/packages/drc/versions/2.5-12/topics/LL.4
   
    c-d - difference between max and min responses
    np.exp( b* np.log(x) - e) -  np.exp((x-p)/s in Dennis Wang's sigmoid
    b - hill slope = 1/s - shape parameter
    np.log(x)- e/b == x-p in Dennis Wang's sigmoid
    """
    return ( (c-d)/(1+np.exp(b*np.log(x)- e)) + d)


def logistic4(x, A, B, C, d):
    """ https://people.duke.edu/~ccc14/pcfb/analysis.html
    4PL logistic equation
    Dennis Wang's sigmoid: 1.0 / (1.0 + np.exp((x-p)/s)
    (A - d) = 1 in Dennis Wang's sigmoid:
    (x/C)**B  - corresponds to np.exp((x-p)/s
    d - determines the vertical position of the graph
    """
    return ( (A-d)/(1.0+((x/C)**B)) + d )


def logLogistR(x, EC50, HS, E_inf):
    """Python analog for PharmacoGx/R/LogLogisticRegression.R
    https://github.com/bhklab/PharmacoGx/blob/master/R/LogLogisticRegression.R
    E = E_inf + (1 - E_inf)/(1 + (x/EC50)^HS)
    Dennis Wang's sigmoid: 1.0 / (1.0 + np.exp((x-p)/s)
    
    (A - d) = 1 in Dennis Wang's sigmoid:
    (np.log10(x)/EC50)**HS  - (in logistic4 (x/C)**B) corresponds to np.exp((x-p)/s 
    
    E_inf - determines the vertical position of the graph /coefficient d, min response in other functions
    """
    return ((1-E_inf)/(1+(np.log10(x)/EC50)**HS) + E_inf)



def FitCurve(fitting_function, x, y, parameters_guess=None, to_plot = False):
#     from scipy.optimize import curve_fit

    if parameters_guess:
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
        plt.plot(x2, y2, "blue", label = "R^2= %0.5f"%r2)   
        plt.title('Least-squares fit')
        plt.legend();
    return r2, parameters


def FittingColumn(df, indexes, x_columns, y_columns, fitting_function, parameters_guess=None, default_param = False):
    """
    intial parameter guess [max(y), np.median(x), 1, min(y)]
    potentially they can be different for each data row, but as soon as we have scaled and filtered data
    we can use by default [1.0, 0.4, 1.0, .0] 
    """
    
    r2_scores = np.zeros(len(indexes))
    X = df.loc[indexes, x_columns].values.astype(np.float32)
    Y = df.loc[indexes, y_columns].values.astype(np.float32)
    fitting_parameters = [None]*len(indexes)
    
    
    # parameters_guess= [np.median(x), 1, max(y), min(y)]
    default_param_model = {"sigmoid_Wang": [0.4, 0.1],
                       "fsigmoid" : [0.4, -10],
                       "sigmoid_4_param": [0.4, 1.0, 1.0, .0],
                       "sigmoid_3_param": [0.4, 1.0, .0],
                       "logistic4": [1.0, 1.0, 1.0, 0.0],
                       "ll4": [0.4, 1.0, 1.0, 0.0],
                       "ll4_R": [0.4, 1.0, 1.0, 0.0],
                       "logLogistR": [-1, -0.1, 0.1]}
    
    if default_param:
        parameters_guess = default_param_model[fitting_function]
       
    else:
        pass
    
    for i in tqdm(range(len(indexes))):
        x = X[i, :]
        y = Y[i, :]
    
        try:
            r2_scores[i], fitting_parameters[i] = FitCurve(fitting_function_object, x, y, parameters_guess = parameters_guess)
            
        except:
            try:
                functions = {"fsigmoid": fsigmoid, 
                 "sigmoid_Wang": sigmoid_Wang, 
                "sigmoid_4_param": sigmoid_4_param,
                 "sigmoid_3_param": sigmoid_3_param, 
                 "logistic4": logistic4,  
                 "ll4": ll4, 
                 "ll4_R":ll4_R,
                 "logLogistR":logLogistR}
                fitting_function_object = functions[fitting_function]
#                 from fitting_curves.py import fitting_function_object
                r2_scores[i], fitting_parameters[i] = FitCurve(fitting_function_object, x, y, parameters_guess = parameters_guess)
            except:
                r2_scores[i] = 0
    return r2_scores, fitting_parameters


def FittedData(df, x_columns, y_columns, fitting_function, parameters_guess=[], default_param = True,
              print_info = False, min_drug_profiles = None):
    
    r2, fit_param = FittingColumn(df, df.index, x_columns, y_columns, fitting_function, default_param = True)
    df["fitting_r2"] = r2
    df["fitting_param"] = fit_param
    df= df[df["fitting_r2"]!=0]
    
    if print_info:
        print("\nResulting DataFrame:", df.shape)
        if min_drug_profiles:
            drugs_10 = df.groupby("DRUG_ID").size()[df.groupby("DRUG_ID").size()>min_drug_profiles].to_dict()
            print("Number of unique drugs:", df["DRUG_ID"].nunique())
            print("Drugs with more than 10 profiles:", len(drugs_10))
    return df

def TransposeCellFeatures(df, indexes_first_column = True):
    # columns in cell_features correspond to drug_curves["COSMIC_ID"] == cell line id
    if indexes_first_column:
        df_transfromed = pd.DataFrame(data= df[df.columns[1:]].values.T,
                          index= df.columns[1:], columns= df[df.columns[0]].values)
    else:
        print("The logic of this function is not applicable")
        
    return df_transfromed 

def SplitFitParam(df, keep_r2_column = False):
    """Column fitting_param is splitted into separate columns """
    conc_columns= ["fd_num_"+str(i) for i in range(10)]
    response_norm = ['norm_cells_'+str(i) for i in range(10)]
    param_columns = ["DRUG_ID", "COSMIC_ID", "MAX_CONC"] + conc_columns + response_norm
    if keep_r2_column:
        param_columns.append("fitting_r2")
    for i in range(len(df['fitting_param'].values[0])):
        param_col = "param_"+str(i+1)
        param_columns.append(param_col)
        df[param_col] = df['fitting_param'].apply(lambda x: x[i])
    return df[param_columns]
    

def MergeDrugCells(df_drugs, df_cells, splitting_needed = False):
    cell_features_T = TransposeCellFeatures(df_cells)
    cell_features_T.index = np.array(cell_features_T.index, dtype = "int")
    
    # Not all the CCL from filtered dataset are present in cell lines features
    common_cells = list(set(np.array(df_cells.columns[1:], dtype="int"))& set(df_drugs["COSMIC_ID"].values))
    
    # print("Number of drugs in filtered dataset:", df_328["COSMIC_ID"].nunique())
    # print("Number of common drugs in both datasets", len(common_cells_drug328))

    cell_lines = cell_features_T.loc[common_cells, :].reset_index()
    cell_lines.rename(columns = {"index": "COSMIC_ID"}, inplace=True)
    if splitting_needed:
        df_drugs = SplitFitParam(df_drugs)
    
    # merge drug profile data (fitted parameters) and cell line features
    return pd.merge(left=df_drugs, right = cell_lines, on = "COSMIC_ID") 


def DataPreprocessing(folder_with_original_data, folder_with_results, filtering_scenario = [1,2,3,4],
                      first_points_lower_limit = 0.8, last_points_upper_limit = 0.4,
                      middle_points_limit =-0.1,
                     fitting_function = "sigmoid_4_param", 
                     print_progress_info = False,
                     keep_r2_column = False):
    """
    middle_points_limit = -0.1, -0.2, None
    fitting_function = "sigmoid_4_param", "logistic4", "logLogistR"
    """
    drug_curves = pd.read_csv(folder_with_original_data+"normalised_dose_response_data.csv")

    conc_columns= ["fd_num_"+str(i) for i in range(10)]
    response_norm = ['norm_cells_'+str(i) for i in range(10)]

    # Data preprocessing

    # filter the data
    df = FilteringCurves(drug_curves, filtering_scenario=filtering_scenario,
                         response_columns = response_norm, 
                         first_points_lower_limit = first_points_lower_limit, 
                         last_points_upper_limit = last_points_upper_limit, 
                         middle_points_limit = middle_points_limit,
                        print_progress_info =print_progress_info)
    # add parameters of fitting function
    df = FittedData(df, x_columns=conc_columns, y_columns= response_norm, 
                fitting_function=fitting_function, default_param = True, 
                print_info = print_progress_info, min_drug_profiles = 10)

    df= SplitFitParam(df, keep_r2_column= keep_r2_column)

    # merge with cell line properties:

    cell_features = pd.read_csv(folder_with_original_data +"Cell_Line_Features_PANCAN_simple_MOBEM.tsv", sep="\t")
    merged_df = MergeDrugCells(df, cell_features)
    if print_progress_info:
        print("\nNot all the CCL from filtered dataset are present in cell lines features!!!")
        print("Number of CCL in CCL features:", len(set(cell_features.columns[1:])))
        print("Number of unique CCL in filtered drug profiles:", df["COSMIC_ID"].nunique())
        common_cells = set(np.array(cell_features.columns[1:], dtype="int")) & set(df["COSMIC_ID"].unique())
        print("Number of common CCL:", len(common_cells))
        # print("Missing CCL - COSMIC_ID:", set(df["COSMIC_ID"].unique())-common_cells)
        print("\nData after merging with CCL properties:", merged_df.shape)

    # merge with drug features:

    with open(folder_with_results+'all_drugs_names.pickle', 'rb') as handle:
        all_drugs_names = pickle.load(handle)


    drug_features = pd.read_csv(folder_with_results + "drug_features_with_pubchem_properties_final.csv")

    df = pd.merge(left = merged_df, right = drug_features, on = "DRUG_ID", how = "left")#.drop("Unnamed: 0", axis=1)
    if print_progress_info:
        print("Data after merging with drug properties:", df.shape)

        display(df[df["molecular_weight"].isnull()]["DRUG_ID"].value_counts())
        df.dropna(axis=0, how='any', inplace=True)
        print("Final data after dropping unknown drugs:", df.shape)
    return df

def TrainTestSplit(df, train_ratio= 0.8, min_number_drug_profiles =10, r2_restriction = None, print_progress_info = False):
                             
    
    # Split into train and test data with more than min_number_drug_profiles record per drug

    train = pd.DataFrame()
    test = pd.DataFrame()
    np.random.seed(123)
                             
    indexes = np.random.permutation(df.index)
    if r2_restriction:
        try:
            df = df[df["fitting_r2"]>r2_restriction].copy()
        except:
            print("Error: No R2 column is found")
                             

    for drug_id in df["DRUG_ID"].unique():
        df_i = df[df["DRUG_ID"]==drug_id]
        indexes = np.random.permutation(df_i.index)
        train_size = int(df_i.shape[0]*train_ratio)
        indexes_train = indexes[:train_size]
        indexes_test = indexes[train_size:]
        train = pd.concat([train, df_i.loc[indexes_train, :]])
        test = pd.concat([test, df_i.loc[indexes_test, :]])
    
    gr = df.groupby("DRUG_ID")["COSMIC_ID"].count()
    drug_ids_limit = list(gr[gr>min_number_drug_profiles].index)


    train_df_limit = train.set_index("DRUG_ID").loc[drug_ids_limit, :].copy()
    test_df_limit = test.set_index("DRUG_ID").loc[drug_ids_limit, :].copy()

    if print_progress_info:
        print("\nAll train/test:", train.shape, test.shape)
        print("Train/test for drugs with 50 profiles:",train_df_50.shape, test_df_50.shape)
    return drug_ids_limit,train_df_limit, test_df_limit


def r2_score_fitting(df, x_columns, y_columns, fitting_function, param_columns = []):
    if "DRUG_ID" not in df.columns:
        df.reset_index(inplace=True)
    functions = {"fsigmoid": fsigmoid, 
                 "sigmoid_Wang": sigmoid_Wang, 
                "sigmoid_4_param": sigmoid_4_param,
                 "sigmoid_3_param": sigmoid_3_param, 
                 "logistic4": logistic4,  
                 "ll4": ll4, 
                 "ll4_R":ll4_R,
                 "logLogistR":logLogistR}
    
    fitting_function_object = functions[fitting_function]
    r2_scores=np.zeros(len(df.index))

    for i in range(len(df.index)):
        x = df.loc[df.index[i], x_columns].values.astype(np.float32)
        y = df.loc[df.index[i], y_columns].values.astype(np.float32)
        fit_param = df.loc[df.index[i], param_columns].values.astype(np.float32)
        y_fit = fitting_function_object(x, *fit_param)
        r2_scores[i] = r2_score(y, y_fit)
    return r2_scores

def mae_score_reconstruct(df, x_columns, y_columns, fitting_function, param_columns = []):
    if "DRUG_ID" not in df.columns:
        df.reset_index(inplace=True)
    functions = {"fsigmoid": fsigmoid, 
                 "sigmoid_Wang": sigmoid_Wang, 
                "sigmoid_4_param": sigmoid_4_param,
                 "sigmoid_3_param": sigmoid_3_param, 
                 "logistic4": logistic4,  
                 "ll4": ll4, 
                 "ll4_R":ll4_R,
                 "logLogistR":logLogistR}
        
    fitting_function_object = functions[fitting_function]

    mae_scores=np.zeros(len(df.index))
    for i in range(len(df.index)):
        x = df.loc[df.index[i], x_columns].values.astype(np.float32)
        y = df.loc[df.index[i], y_columns].values.astype(np.float32)
        fit_param = df.loc[df.index[i], param_columns].values.astype(np.float32)
        y_fit = fitting_function_object(x, *fit_param)
        mae_scores[i] = mean_absolute_error(y, y_fit)
    return mae_scores