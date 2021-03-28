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

def GetPubChemId(drug_features):
    df = drug_features[(drug_features["pubchem_id"]=="-") | (drug_features["pubchem_id"]=="none")]
    print("Calling PubChem to get pubchem ids")
    for ind in tqdm(df.index):
        drug_name = df.loc[ind, "Drug_Name"]
        deriv = pcp.get_compounds(drug_name, 'name')
        deriv_found = len(deriv)
        try:
            drug_features.loc[ind, "pubchem_id"]= re.findall(r'\((.*?)\)', str(deriv))
        except:
            drug_features.loc[ind, "pubchem_id"]= "-"
    return drug_features

def GetPubChemIdManual(df, gdsc, new_column_name = "pubchem"):
    """Take pubchem id for all the drugs"""
    
    manual_corrections_dict= {
    #Camptothecin == SN38, pubchem id: 104842 
     "Camptothecin" : "104842",
    #from gdsc2: WIKI4	WIKI-4, WIKI 4	WNT signaling	TNKS1, TNKS2	2984337
     "WIKI4": "2984337",
    #from gdsc2: SB505124	SB 505124, SB505124	RTK signaling	TGFBR1, ACVR1B, ACVR1C	9858940
        "SB-505124" : "9858940",
    #from gdsc1: drug_name	synonyms	pathway_name	targets	pubchem
    #Wee1 Inhibitor	681640, Wee1 Inhibitor	Cell cycle	WEE1, CHEK1	10384072
        "681640": "10384072",
     #BX796 https://mpegs-1.com/index.php/2017/12/18/Celecoxib/ 73051434
       #???? "BX796" : "73051434",
    #https://ftp.mcs.anl.gov/pub/candle/public/benchmarks/Pilot1/combo/GDSC_drugs
    # matches BX796 to BX795 with pubchemid 10077147 https://github.com/ECP-CANDLE/Benchmarks
    "BX796" : "10077147",
    "Lestaurtinib":"126565",# "reference" : "https://www.cancerrxgene.org/compounds"},
    "Lestauritinib": "126565", # "reference" : "https://www.cancerrxgene.org/compounds"},
    }
    
    not_found = []
    for ind in df.index:
        name = df.loc[ind, "Drug_Name"]
        drug = gdsc[gdsc["drug_name"]== name]
        if drug.shape[0]==1:
            df.loc[ind, new_column_name]= drug["pubchem"].values[0]
        elif drug.shape[0]==0:
            not_found.append(name)
        else:
            if drug["pubchem"].values[0] != "-":
                df.loc[ind, new_column_name]= drug["pubchem"].values[0]
            else: 
                df.loc[ind, new_column_name]= sorted(drug["pubchem"].values, reverse=True)[0]
    
    not_found2 = []
    for name in not_found:
        if name in manual_corrections_dict:
#             print(name, manual_corrections_dict[name])
            ind = df[df["Drug_Name"]==name].index
            df.loc[ind, new_column_name]= manual_corrections_dict[name]
        else:
            not_found2.append(name)
    
    return df, not_found2



# Manual matching for drugs with missing or mutiple data

def RunManualCorrections(df_drugs, drop_not_found_drugs = False):
    new_synonyms = {"Y-39983": {"Synonyms": "Y-33075",
                           "reference": ["https://www.medchemexpress.com/Y-33075.html",
                            "https://www.nature.com/articles/s41467-019-13781-3"]}}

    manual_corrections = {
  
    "WZ-1-84": {"pubchem_id" : 49821040,
               "reference" : "http://lincs.hms.harvard.edu/db/datasets/20119/smallmolecules"},
    
    "Bleomycine": {"pubchem_id" : 72467,
               "reference" : "https://www.drugbank.ca/drugs/DB00290"},
     "Bleomycin" : {"pubchem_id" : 72467,
               "reference" : "see Bleomycine"},
    
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
     "Cyclopamine":  {"pubchem_id": 442972,
                    "reference" : "https://www.sciencedirect.com/science/article/abs/pii/S0014299915300625"},
     "Midostaurin":  {"pubchem_id": 9829523,
                   "reference" : "https://link.springer.com/article/10.1007/s13205-019-1594-y"}
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
    
    if "DRUG_ID" in df_drugs.columns:
        df_drugs.set_index("DRUG_ID", inplace=True)
    
    not_identified_drugs = {}

    for ind in df_drugs[(df_drugs["pubchem_id"]=="none") | (df_drugs["pubchem_id"]=="several")].index:
            drug_name = df_drugs.loc[ind, "Drug_Name"]
    
            if drug_name in corrections_pubchem_id:
                df_drugs.loc[ind, "pubchem_id"] = corrections_pubchem_id[drug_name]
#                 print("corrections_pubchem_id:", drug_name)
            elif drug_name in manual_corrections:
                df_drugs.loc[ind, "pubchem_id"]  = manual_corrections[drug_name]["pubchem_id"]
#                 print("manual_corrections:", drug_name)
            else:
                not_identified_drugs[ind] = drug_name
                
    print("Total number of drugs:", df_drugs["Drug_Name"].nunique())
    print("Number of not found drugs:", len(not_identified_drugs))
    
    if drop_not_found_drugs:
        df_drugs.drop(list(not_identified_drugs.keys(), axis=1, inplace=True))
        
    return df_drugs, not_identified_drugs

def CallPubChemOne(PubChem_id):
    properties= {}
    try:
        c = Compound.from_cid(PubChem_id)
    except:
        print(PubChem_id, "is not callable")
        
    properties["molecular_weight"] = c.molecular_weight
   
    properties["elements"] = str(set(c.elements)).strip("{").strip("}")
        
    bonds = [int(str(i).split(",")[-1].strip(")")) for i in c.bonds]
    properties["2bonds"] = bonds.count(2)
    properties["3bonds"] = bonds.count(3)

    properties["xlogp"] = c.xlogp
    properties["formal_charge"] = c.charge
    
    properties["surface_area"] = c.tpsa

    properties["complexity"] = c.complexity

    properties["h_bond_donor_count"] = c.h_bond_donor_count

    properties["h_bond_acceptor_count"] = c.h_bond_acceptor_count

    properties["rotatable_bond_count"] = c.rotatable_bond_count

    properties["heavy_atom_count"] = c.heavy_atom_count

    properties["atom_stereo_count"] = c.atom_stereo_count

    properties["defined_atom_stereo_count"] = c.defined_atom_stereo_count

    properties["undefined_atom_stereo_count"] = c.undefined_atom_stereo_count

    properties["bond_stereo_count"] = c.bond_stereo_count

    properties["covalent_unit_count"] = c.covalent_unit_count
    properties["molecular_formula"] = c.molecular_formula

    properties["canonical_smiles"] = c.canonical_smiles

    properties["inchi_string"] = c.inchi

    properties["inchi_key"] = c.inchikey
    return properties




def GetPubchemProperties(df_drug_properties):
    drugs_to_drop = []
    for ind in tqdm(df_drug_properties.index):
        try:
            PubChem_id = int(df_drug_properties.loc[ind, "pubchem_id"])
            c = Compound.from_cid(PubChem_id)
            df_drug_properties.loc[ind, "molecular_weight"] = c.molecular_weight
   
            df_drug_properties.loc[ind, "elements"] = str(set(c.elements)).strip("{").strip("}")
        
            bonds = [int(str(i).split(",")[-1].strip(")")) for i in c.bonds]
            df_drug_properties.loc[ind, "2bonds"] = bonds.count(2)
            df_drug_properties.loc[ind, "3bonds"] = bonds.count(3)

            df_drug_properties.loc[ind, "xlogp"] = c.xlogp
            df_drug_properties.loc[ind, "formal_charge"] = c.charge
    
            df_drug_properties.loc[ind, "surface_area"] = c.tpsa

            df_drug_properties.loc[ind, "complexity"] = c.complexity

            df_drug_properties.loc[ind, "h_bond_donor_count"] = c.h_bond_donor_count

            df_drug_properties.loc[ind, "h_bond_acceptor_count"] = c.h_bond_acceptor_count

            df_drug_properties.loc[ind, "rotatable_bond_count"] = c.rotatable_bond_count

            df_drug_properties.loc[ind, "heavy_atom_count"] = c.heavy_atom_count

            df_drug_properties.loc[ind, "atom_stereo_count"] = c.atom_stereo_count

            df_drug_properties.loc[ind, "defined_atom_stereo_count"] = c.defined_atom_stereo_count

            df_drug_properties.loc[ind, "undefined_atom_stereo_count"] = c.undefined_atom_stereo_count

            df_drug_properties.loc[ind, "bond_stereo_count"] = c.bond_stereo_count

            df_drug_properties.loc[ind, "covalent_unit_count"] = c.covalent_unit_count
            df_drug_properties.loc[ind, "molecular_formula"] = c.molecular_formula

            df_drug_properties.loc[ind, "canonical_smiles"] = c.canonical_smiles

            df_drug_properties.loc[ind, "inchi_string"] = c.inchi

            df_drug_properties.loc[ind, "inchi_key"] = c.inchikey
        except:
            print("Error with drug:", df_drug_properties.loc[ind, "Drug_Name"])
            drugs_to_drop.append(ind)
            pass
    return df_drug_properties, drugs_to_drop


def PreprocessPubChem(drug_features, save_features_names=False, _FOLDER_to_save = None):
    
    int_columns = ['2bonds', '3bonds', 'h_bond_donor_count',
       'h_bond_acceptor_count', 'rotatable_bond_count', 'heavy_atom_count',
       'atom_stereo_count', 'defined_atom_stereo_count',
       'undefined_atom_stereo_count', 'bond_stereo_count',
       'covalent_unit_count']
    
    for col in int_columns:
        drug_features[col] = np.int16(drug_features[col]) 
        
    all_elements = list(set(drug_features["elements"].str.split(",", expand=True).fillna(0).values.flatten())- set([0," 'C'", "'C'", " 'H'"]))
    elements_in_drugs= list(set([atom.strip(" ").strip("'") for atom in all_elements]))
    exceptions =[]
    for drug_index in drug_features.index:
        compound_elements = drug_features.loc[drug_index, "elements"]
        #print(compound_elements)
        try:
            for i, atom in list(enumerate(elements_in_drugs)):
                if atom in compound_elements:
                    drug_features.loc[drug_index, atom] = 1
                #print(atom, "Yes")
                else:
                    drug_features.loc[drug_index, atom] = 0
                #print(atom, "No")
        except:
            exceptions.append(drug_index)
            drug_features.loc[drug_index, atom] = 0
            
    col_elements = ['B', 'I', 'Br', 'Cl', 'O', 'N', 'F', 'P', 'S', 'Pt']  
    
    for col in col_elements:
        drug_features[col]= np.int16(drug_features[col])
        
    if len(exceptions)>0:
        print("Exceptions are found :", drug_features.loc[exceptions, :].shape[0])
    print("Elements in drugs:", len(elements_in_drugs), elements_in_drugs)
    
    
    if save_features_names:
        
        PubChem_features = ["molecular_weight","2bonds", "3bonds", "xlogp", 
                            "formal_charge", "surface_area", "complexity", 
                            "h_bond_donor_count",  "h_bond_acceptor_count", "rotatable_bond_count",
                            "heavy_atom_count", "atom_stereo_count", "defined_atom_stereo_count",
                            "undefined_atom_stereo_count", "bond_stereo_count", "covalent_unit_count",
                            'B', 'I', 'Br', 'Cl', 'O', 'N', 'F', 'P', 'S', 'Pt']
        
        with open(_FOLDER_to_save + "X_PubChem_properties.txt", 'w') as f:
            for s in PubChem_features:
                f.write(str(s) + '\n')
                
    
    return drug_features
    
    
def GetDumniesTargets(drug_features, save_features_names=False, _FOLDER_to_save = None):   
    
    targets = ""
    for x in drug_features["Target"].values:
        targets = targets + ", " + x
    targets = list(set(targets.split(", ")[1:]))
    
    print("Number of targets:", len(targets))
    df_target = pd.DataFrame(data = np.int32(np.zeros([drug_features.shape[0], len(targets)])), 
                         index = drug_features.index, 
                         columns = targets)
    for index in drug_features.index:
        targets_i = drug_features.loc[index, "Target"].split(", ")
        df_target.loc[index, targets_i]=1
        
    print("Number of unique pathways:", drug_features["Target_Pathway"].nunique())
    df_target_target_pathway = pd.concat([df_target, pd.get_dummies(drug_features["Target_Pathway"])], axis=1)
    df_final = pd.concat([drug_features.drop(["Target_Pathway"], axis=1), df_target_target_pathway], axis=1)
    
    if save_features_names:
        with open(_FOLDER_to_save + "X_features_Targets.txt", 'w') as f:
            for s in targets:
                f.write(str(s) + '\n')
        
        with open(_FOLDER_to_save + "X_features_Target_Pathway.txt", 'w') as f:
            for s in drug_features["Target_Pathway"].unique():
                f.write(str(s) + '\n') 
    return df_final


def PreprocessDrugs(drug_features, drugs_gdsc=None, drug_features_wih_pubchem_id =False, save_features_names=False, _FOLDER_to_save = None):
    if "Drug ID" in drug_features.columns:
        drug_features.rename(columns={"Drug ID": "DRUG_ID", 
                                      "Drug Name": "Drug_Name", 
                                      "Target Pathway": "Target_Pathway"},
                            inplace=True)
    if drug_features_wih_pubchem_id ==False:     
        drug_features, not_found = GetPubChemIdManual(drug_features, drugs_gdsc, new_column_name = "pubchem_id")
    
    drug_features, not_identified_drugs = RunManualCorrections(drug_features, drop_not_found_drugs = False)
    
    drug_features = drug_features = drug_features[(drug_features["pubchem_id"]!="none")| (drug_features["pubchem_id"]!="none")]
    drug_features["pubchem_id"] = drug_features["pubchem_id"].astype("object") 
    print("Calling PubChem...")
    drug_features, drugs_to_drop = GetPubchemProperties(drug_features)
    print("Processing drug properties...")
    drug_features = PreprocessPubChem(drug_features, save_features_names=save_features_names, 
                                      _FOLDER_to_save = _FOLDER_to_save)
    
    drug_features = GetDumniesTargets(drug_features, save_features_names=save_features_names, 
                                      _FOLDER_to_save = _FOLDER_to_save)
    return drug_features