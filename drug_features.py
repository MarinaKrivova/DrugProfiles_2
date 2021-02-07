import pandas as pd
import numpy as np
import os
# pip install PubChemPy
import pubchempy as pcp
import re
from pubchempy import Compound
import warnings
warnings.filterwarnings("ignore")
import time
from tqdm import tqdm
import pickle

def GetPubChemId(drug_ids, drug_names_dict):
    pubchem_ids = {}
    
    for drug_id in tqdm(drug_ids):
        deriv = pcp.get_compounds(drug_names_dict[drug_id], 'name')
        pubchem_ids[drug_id] = {}
        pubchem_ids[drug_id]["Drug_Name"] = drug_names_dict[drug_id]
        pubchem_ids[drug_id]["deriv_found"] = len(deriv)
        try:
            pubchem_ids[drug_id]["PubChem_ID"]= re.findall(r'\((.*?)\)', str(deriv))
        except:
            if len(deriv)>1:
                pubchem_ids[drug_id]["PubChem_ID"]= str([np.int(x) for x in re.findall(r'\((.*?)\)', str(deriv))]).strip("[").strip("]")
            else:
                pubchem_ids[drug_id]["PubChem_ID"]= 0
    return pubchem_ids

def RunManualCorrections(pubchem_ids_dict, drop_not_found_drugs = False):
    new_synonyms = {"Y-39983": {"Synonyms": "Y-33075",
                           "reference": ["https://www.medchemexpress.com/Y-33075.html",
                            "https://www.nature.com/articles/s41467-019-13781-3"]}}

    manual_corrections = {
    "Lestaurtinib":{"pubchem_id" : 126565,
               "reference" : "https://www.cancerrxgene.org/compounds"},
    "Lestauritinib":{"pubchem_id" : 126565,
               "reference" : "https://www.cancerrxgene.org/compounds"},
    
    "WZ-1-84": {"pubchem_id" : 49821040,
               "reference" : "http://lincs.hms.harvard.edu/db/datasets/20119/smallmolecules"},
    
    "GW441756": {"pubchem_id" : 9943465 ,
               "reference" : "",
               "note": "no result in drugbank"},
    
    "Parthenolide" : {"pubchem_id" : 6473881,
               "reference" : "https://www.drugbank.ca/drugs/DB13063"},
    
    "Obatoclax Mesylate": {"pubchem_id" : 347828476,
               "reference" : "https://www.drugbank.ca/drugs/DB12191"},
    
    "Bleomycine": {"pubchem_id" : 72467,
               "reference" : "https://www.drugbank.ca/drugs/DB00290"},
    
    "Y-39983": {"pubchem_id" : 20601328,
               "reference" : "https://www.medchemexpress.com/Y-33075.html"},
    
    "JW-7-52-1": {"pubchem_id" : 20822503,
               "reference" : "https://pharmacodb.ca/drugs/392"},
    
    "VNLG/124": { "pubchem_id": 24894414, 
                  "reference": "https://www.cancerrxgene.org/compounds" },
    
    "PDK1 inhibitor 7": { "pubchem_id": 56965967, 
                         "reference": "https://www.cancerrxgene.org/compounds"},
    
    "KIN001-260": {"pubchem_id": 10451420, 
                   "reference": "https://www.cancerrxgene.org/compounds"},
    
    "SB52334": {"pubchem_id": 9967941, 
                "reference": "https://www.cancerrxgene.org/compounds"},
    
    "KIN001-270": { "pubchem_id": 66577006, 
                   "reference": "https://www.cancerrxgene.org/compounds"},
    
    "Cisplatin": {"pubchem_id": 84691, 
                  "reference": "https://www.cancerrxgene.org/compounds"},
    
    "Cetuximab": {"pubchem_id": 85668777, 
                  "reference": "https://www.cancerrxgene.org/compounds"},
    
    "Nutlin-3a (-)": { "pubchem_id": 
                      11433190, "reference": ""},
    
    "681640": { "pubchem_id": 10384072, 
               "reference": ""},
    
    "MPS-1-IN-1": {"pubchem_id": 25195352, 
                   "reference": ""},
    
    "KIN001-266": { "pubchem_id": 44143370, 
                   "reference": ""},
    
    "JW-7-52-1" : {"pubchem_id": 49836027, 
                   "reference": ""},
    
    "Vinorelbine": {"pubchem_id": 44424639, 
                   "reference": "https://www.drugbank.ca/drugs/DB00361"},
    
    "Paclitaxel": {"pubchem_id": 36314, 
                   "reference": "https://www.drugbank.ca/drugs/DB01229"},
    
    "Bleomycin": {"pubchem_id": 5360373, 
                   "reference": "https://www.drugbank.ca/drugs/DB00290"},
    
    "Vinblastine": {"pubchem_id": 13342, 
                   "reference": "https://www.drugbank.ca/drugs/DB00570"},
    
    
    "THZ-2-102-1" : {"pubchem_id": 146011539, 
                   "reference": "Katjusa Koler's suggestion"},
    
    "THZ-2-49" : {"pubchem_id": 78357763 , 
                   "reference": ["https://www.cancerrxgene.org/compounds", 
                                "https://www.medchemexpress.com/THZ2.html",
                                "https://pubchem.ncbi.nlm.nih.gov/compound/78357763"]},
    
    "QL-XII-47": {"pubchem_id": 71748056, 
                   "reference": "https://lincs.hms.harvard.edu/db/sm/10077-101-1/"},
    
    "BMS-345541" : {"pubchem_id": 9813758, 
                   "reference": ""},
    
    "Temsirolimus" : {"pubchem_id": 23724530, 
                   "reference": "https://www.drugbank.ca/drugs/DB06287"},
    
    "SB590885" : {"pubchem_id": 135398506, 
                   "reference": "https://pubchem.ncbi.nlm.nih.gov/#query=SB590885"},
    
    "WZ3105" : {"pubchem_id": 42628507, 
                   "reference": "https://lincs.hms.harvard.edu/db/sm/10084-101/"},
    
    "NPK76-II-72-1" : {"pubchem_id": 46843648, 
                   "reference": "https://lincs.hms.harvard.edu/db/sm/10070-101/"},
    
    "JW-7-24-1" : {"pubchem_id": 69923936, 
                   "reference": "https://lincs.hms.harvard.edu/db/sm/10019-101/"},
    "Bryostatin 1" : {"pubchem_id": 6435419, 
                   "reference": "https://pubchem.ncbi.nlm.nih.gov/#query=Bryostatin%201"},
    "QL-XI-92": {"pubchem_id": 73265214,
                 "reference": "Katjusa Koler's & Dennis Wang's database"},
    
    "SL0101": {"pubchem_id": 10459196,
                 "reference": "https://www.cancerrxgene.org/compounds"}, 
    "Z-LLNle-CHO": {"pubchem_id": 16760646  ,
                 "reference": "https://www.cancerrxgene.org/compounds"}, 
    "JNK-9L": {"pubchem_id": 25222038  ,
                 "reference": "https://www.cancerrxgene.org/compounds"}, 
    "KIN001-244": {"pubchem_id": 56965967  ,
                 "reference": "https://www.cancerrxgene.org/compounds"},
    "RO-3306":  {"pubchem_id": 44450571  ,
                 "reference": "https://www.cancerrxgene.org/compounds"},
    "EHT-1864": {"pubchem_id": 9938202  ,
                 "reference": "https://www.cancerrxgene.org/compounds"},  
    }
    
    corrections_pubchem_id = {
    "Temsirolimus": 6918289,
    "Vinorelbine": 5311497,
    "Y-39983": 9810884,
    "GW441756": 9943465, 
    "Vinblastine": 6710780,
    "Bryostatin 1": 5280757,
    "Parthenolide": 7251185,
    "Obatoclax Mesylate": 11404337,
    "Bleomycin (50 uM)": 5460769,
    "SB590885": 11316960,
    "Paclitaxel" :36314,
    "BMS-345541": 9813758,
    "YM201636" :  9956222, 
    }
    
    not_identified_drugs = {}
    corrected_drugs = {}
    for drug_id in pubchem_ids_dict:
        if pubchem_ids_dict[drug_id]["deriv_found"]!=1:
            drug_name = pubchem_ids_dict[drug_id]["Drug_Name"]
            
            if drug_name in corrections_pubchem_id:
                pubchem_ids_dict[drug_id]["PubChem_ID"] = corrections_pubchem_id[drug_name]
                corrected_drugs[drug_id] = drug_name
                pubchem_ids_dict[drug_id]["deriv_found"] = 1
            elif drug_name in manual_corrections:
                pubchem_ids_dict[drug_id]["PubChem_ID"] = manual_corrections[drug_name]
                corrected_drugs[drug_id] = drug_name
                pubchem_ids_dict[drug_id]["deriv_found"] = 1
            else:
                not_identified_drugs[drug_id] = drug_name
                
    print("Total number of drugs:", len(pubchem_ids_dict))
    print("Total number of drugs for correction:", len(not_identified_drugs) + len(corrected_drugs))
    print("Number of corrected drugs:", len(corrected_drugs))
    print("Number of not found drugs:", len(not_identified_drugs))
    
    if drop_not_found_drugs:
        for drug_id in not_identified_drugs:
            del pubchem_ids_dict[drug_id]
    print("Final number of drugs with PubChem id:", len(pubchem_ids_dict))
    return pubchem_ids_dict, not_identified_drugs