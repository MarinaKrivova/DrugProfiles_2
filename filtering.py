import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def ShowResponseCurves(df, plots_in_row, plots_in_column, x_columns, y_columns, start_index=0, 
                       indexes=[], drug_dict = None, CCL_dict=None,
                       upper_limit=None, lower_limit=None):
 
                
    fig = plt.figure(figsize=(14, 11))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    n_plots= plots_in_row*plots_in_column
    
    if len(indexes) ==0:
        indexes =df.index[start_index : start_index+n_plots]

    for i, ind in list(enumerate(indexes)):
        x = df.loc[ind, x_columns].values.astype(np.float32)
        y = df.loc[ind, y_columns].values.astype(np.float32)
                
        ax = fig.add_subplot(plots_in_row, plots_in_column, i+1)
        
        if max(y)>1:
            max_y= max(y)+0.1
        else:
            max_y=1.1
        ax.set_ylim([0, max_y])
        ax.scatter(x,y)
        
        if drug_dict and CCL_dict:
            ax.set_title("Drug: "+ drug_dict[df.loc[ind, "DRUG_ID"]]+" / CCL: "+CCL_dict[df.loc[ind, "COSMIC_ID"]])
        elif drug_dict:
            ax.set_title("Drug: "+drug_dict[df.loc[ind, "DRUG_ID"]] +"_"+ str(df.loc[ind, "COSMIC_ID"]))
        
        else:
            ax.set_title(str(df.loc[ind, "DRUG_ID"])+"_"+str(df.loc[ind, "COSMIC_ID"]))
        ax.set_xlabel("Scaled dosage")
        ax.set_ylabel("Normalised response")
        
        if upper_limit:
            ax.axhline(upper_limit,color='red',ls='--')
        if lower_limit:
            ax.axhline(lower_limit, color='black',ls='--')
        
        
def ShowSpecific(df, drug_CCL_list, x_columns, y_columns, 
                 drug_col = "drug_name", CCL_col= "CCL_name",
                 upper_limit=None, lower_limit=None):
    
    """df should contain columns drug_col and CCL_col corresponding to the values in drug_CCL_list
    Display no more than 4 plots
    drug_CCL_list should be in the form [(drug, CCL)]
    """
    
    n_plots= len(drug_CCL_list)
    
    if n_plots<=4:
        fig = plt.figure(figsize=(5*n_plots, 3)) # 3 is height
    else:
        print("Too many samples")
        
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    i=0
    for (drug, CCL) in drug_CCL_list:
        ind = df[(df[drug_col]==drug)&(df[CCL_col]==CCL)].index

        x = df.loc[ind, x_columns]
        y = df.loc[ind, y_columns].values[0] #possible problems are here

                
        ax = fig.add_subplot(1, n_plots, i+1)

        if max(y)>1:
            max_y= max(y)+0.1
        else:
            max_y = 1.1
        ax.set_ylim([0, max_y])
            
        ax.scatter(x,y)
        
        ax.set_title("Drug: "+ str(drug) +" / CCL: "+ str(CCL))
        ax.set_xlabel("Scaled dosage")
        ax.set_ylabel("Normalised response")
        i+=1
        
        if upper_limit:
            ax.axhline(upper_limit,color='red',ls='--')
        if lower_limit:
            ax.axhline(lower_limit, color='black',ls='--')
            
            
def FindHighResponses(df, response_cols):
    df = df.copy()
    df["high_responses"] = df[response_cols].apply(lambda row: sum(row>1), axis=1)
    df_bad = df[df["high_responses"]>1]
    return df_bad

def CutOffOutliers(df, response_columns, middle_points_limit=-0.2):
    df = df.copy()
    bad_index = []
    for j in range(1, len(response_columns)-1): # two first and two last are already assessed
        bad_index.extend(df[(df[response_columns[j]] - df[response_columns[j+1]])<= middle_points_limit].index)
    index_to_use = list(set(df.index) - set(bad_index))
    return df.loc[index_to_use, :] 

def FindAscendingData(df, response_columns, middle_points_limit=-0.2):
    df = df.copy()
    bad_index = []
    for j in range(1, len(response_columns)-1): # two first and two last are already assessed
        bad_index.extend(df[(df[response_columns[j]] - df[response_columns[j+1]])< middle_points_limit].index)
        
    index_to_use = list(set(bad_index))
    return df.loc[index_to_use, :] 
        
def FilteringSigmoidCurves(df, response_columns, filtering_scenario = [1,2,3], 
                     first_columns_to_compare = [1, 2], last_columns_to_compare = [-1, -2],
                     tolerance=0.05, 
                     first_points_lower_limit = 0.8, 
                     last_points_upper_limit = 0.4,
                      middle_points_limit = -0.2):
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
        if i ==1:
            #1st filtering
            index_row_more_than_1 = []
            for col in response_columns:
                if sum(df[col]>1)>0:
                    index_row_more_than_1.extend(df[df[col]>1].index)
        
            index_row_less_than_1 = set(df.index) - set(index_row_more_than_1)
            df = df.loc[index_row_less_than_1, :].copy()
            print("1st filtration (Ensure that all the response are less than 1): Filtered dataset:", df.shape)
        
        elif i== 2: 
            #2nd filtering
            df["dif_first"]=abs(df[response_columns[first_columns_to_compare[0]-1]]\
                                     - df[response_columns[first_columns_to_compare[1]-1]])
            df["dif_last"]=abs(df[response_columns[last_columns_to_compare[0]]] \
                                        - df[response_columns[last_columns_to_compare[1]]])

            df = df[(df["dif_first"]<= tolerance)
                           &(df["dif_last"]<= tolerance)]
    
            print("2d filtration (Ensure that first and last points form plateus): Filtered dataset:", df.shape)
        elif i== 3: 
                #3d filtering
                df = df[(df[response_columns[1]]>first_points_lower_limit) 
                         & (df[response_columns[-1]]<last_points_upper_limit)]
                print("3d stage filtration (Specified location of the plateus): Filtered dataset:", df.shape)
        
        elif i==4:
            df = CutOffOutliers(df, response_columns, middle_points_limit)
            
            print("4th stage filtration (Cut off high ancedent points): Filtered dataset:", df.shape)
            
        else:
            print("Unknown filtration scenario")
    
    return df



from sklearn.metrics import auc
from scipy.stats import spearmanr
from tqdm import tqdm

def AucFitration(df, conc_columns, response_columns, auc_limit=0.7, save_file_name = None):
    """
    1. Remove all the curves where the normalised response value is greater than one at zero dosage. 
    2. Compute the Area Under the Curve (AUC) for all the curves. 
    3. Leave only those curves with an AUC>0.7.  
    4. Compute the Spearman correlation coefficient between the normalised response and the scaled dosage 
    (so the x-axis and the y-axis).
    5. Further remove the curves for which the Spearman correlation coefficient is zero or positive.
    """
    
    df = df[df["norm_cells_0"]<=1].copy()
    
    for index in tqdm(df.index):
        row = df.loc[index, :]
        df.loc[index, "auc"] = auc(row[conc_columns],row[response_columns])
        df.loc[index, "spearman_r"] = spearmanr(row[conc_columns],row[response_columns])[0]
    
    if save_file_name:
        df.to_csv(save_file_name, index=False)
    
    df = df[(df["auc"]>auc_limit) & (df["spearman_r"]<0)].copy()
    return df

def FilterGoodResponse(df, response_cols, final_response_limit=0.4):
    df["count_missing"] = df[response_cols].isnull().sum(axis=1)
    df["count_missing"].value_counts()
    
    ind_1 = list(df[(df["count_missing"]==4)& (df[response_cols[5]]<final_response_limit)].index)
    ind_2 = list(df[(df["count_missing"]==0)& (df[response_cols[-1]]<final_response_limit)].index)
    
    return df.loc[ind_1+ind_2, :]