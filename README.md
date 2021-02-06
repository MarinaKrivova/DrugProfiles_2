# DrugProfiles_2
Second version of ml predictions of drug response curves


Logics:
1. find out pubchem id for the drugs used in drug profiling
2. remove records with drugs that do not have a reported pubchem id
3. perform filtering of the rest data to leave only valid sigmoid drug response curves
4. divide the rest filtered data into train, validation and test splits according to the number of drug profiles for a certain drug
