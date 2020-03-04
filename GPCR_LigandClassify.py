#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: mmahmed
"""

################################
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import numpy as np
import pandas as pd
import matplotlib
import deepchem
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format
from rdkit.Chem import Descriptors
from deepchem.feat import Featurizer
import pickle
from keras.models import model_from_json
import os
import sys
######
from rdkit import Chem,rdBase
from rdkit.Chem import PandasTools
import keras
import keras.utils
from datetime import date
today = date.today()
import argparse
###################################################################################################
def input_file(path):
    """Check if input file exists."""

    path = os.path.abspath(path)
    if not os.path.exists(path):
        raise IOError('File %s does not exist.' % path)
    return path


def output_file(path):
    """Check if output file can be created."""

    path = os.path.abspath(path)
    dirname = os.path.dirname(path)

    if not os.access(dirname, os.W_OK):
        raise IOError('File %s cannot be created (check your permissions).'
                      % path)
    return path


def string_bool(s):
    s = s.lower()
    if s in ['true', 't', '1', 'yes', 'y','True']:
        return True
    elif s in ['false', 'f', '0', 'no', 'n', 'False']:
        return False
    else:
        raise IOError('%s cannot be interpreted as a boolean' % s)

################################################################################################

os.chdir('.')
##It is important to know which pandas versions you are using in order to avoid syntax mismatch.
##The script should work fine with pandas versions 0.22 & 0.23 & ..
print ('PANDAS version ',pd.__version__)
print('########################')
print(''' The following python librares are required to run the models:

* Python 2.7 (tested with Anaconda distribution, Linux OS (Mint 19.1 for a local PC, and 3.10.0-957.12.2.el7.x86_64 GNU/Linux for ComputeCanada clusters). Running the models on MAC may be cumbersome because of the recent XGBoost updates. We did not test the prediction on Windows.)
* DeepChem 1.x (Require RDKit)
* Pandas (Prediction is tested with Pandas 0.22)
* Tensorflow 1.3
* Keras
* XGBoost
* ScikitLearn
''')
print('########################')
print('''In order for the script to run, and in addition to the input file (see below), the following files should exist in the running directory:

* dl_model_fp.json
* dl_model_fp.h5
* mlp_rdkit_classify_fp.sav
* xgb_rdkit_classify_fp.sav
* rfc_rdkit_classify_fp.sav
* svm_rdkit_classify_fp.sav
* coded_gpcr_list.csv

NB: The rfc_rdkit_classify_fp.sav & svm_rdkit_classify_fp.sav & mlp_rdkit_classify_fp.sav models are required only if the [--ignore_rf_svm argument] option in the script is set to False (True is the default behaviour)\n. The models are not deposited in the github repository because of size limits, to get these two models a direct request should be sent to mmahmed@ualberta.ca & kbarakat@ualberta.ca
''')

print('########################')
print ('Welcome to GPCR_LigandClassify, this is how you can use the program using the models to make novel predictions, we hope you find these predictions useful for your task: \n\
       python GPCR_LigandClassify.py --input_file input.csv --output_file output.csv [--n_rows_to_read <INTEGER>] [--mwt_lower_bound <FLOAT>] [--mwt_upper_bound <FLOAT>] [--logp_lower_bound <FLOAT>] [--logp_upper_bound <FLOAT>] [--ignore_rf_svm <True/False>]')
print('########################')
print('The input & output file names arguments are mandatory arguments, --n_rows_to_read argument determines how many rows you want to read from the input CSV files (default 9999999999 rows)\n, the rest are optional with default same as input dataset used for models training.')
print('########################')
print('The --ignore_rf_svm argument will ignore the RF and the SVM models which are pretty large, suitable in case of limited computational resourcses, particularly memory. Default is True (Ignore Randomforests and SVM models.)')
print('########################')
print('Please note that a today date string will be attached to the output file name.')
print('########################')
print('Please note that the script will only save ligands where all models predictions agree.')
print('########################')
print('For the input file, please keep the same format as the attached sample input file. In case of data coming from different source, you can populate the rest of columns with fake data.\nWith the exception of the SMILES column, other columns may be left blank (not recommended).')
print('########################')
print('For the models and auxiliary files, please visit the following github repository:\n\
      https://github.com/mmagithub/GPCR_LigandClassify')
print('########################')
##############Read Input/output file names#####################################
parser = argparse.ArgumentParser()

parser.add_argument('--input_file', help = "input filename", type = input_file)
parser.add_argument('--output_file', help = "output filename", type = output_file)
parser.add_argument('--n_rows_to_read', help = "Number of rows to read from the input structures file", default=9999999999,type=int)
parser.add_argument('--mwt_lower_bound', help = "Lowest molecular weight to consider", default=100,type=float)
parser.add_argument('--mwt_upper_bound', help = "Highest molecular weight to consider", default=900,type=float)
parser.add_argument('--logp_lower_bound', help = "Lowest LogP to consider", default=-4,type=float)
parser.add_argument('--logp_upper_bound', help = "Highest LogP to consider", default=10,type=float)
parser.add_argument('--ignore_rf_svm', help = "Ignore RF and SVM models, suitable for small computational resources, default is True", default=True,type=string_bool)

args = parser.parse_args()

print('inputfile:', args.input_file)
OUTPUT_FILE = args.output_file.split('.')[0] + "_{0}.csv".format(today)
print('outputfile:', "{0}".format(OUTPUT_FILE))
STRUCTURES_FILE = args.input_file
###############################################################################
#Here add some constants
#STRUCTURES_FILE = r"drug_bank_structures.csv"
ROWS_TO_READ = args.n_rows_to_read
###############################################################################
MWT_LOWER_BOUND = args.mwt_lower_bound
MWT_UPPER_BOUND = args.mwt_upper_bound
LOGP_LOWER_BOUND = args.logp_lower_bound
LOGP_UPPER_BOUND = args.logp_upper_bound
###############################################################################



###############Adding Descriptors And FingerPrints#############################
########[1] Adding FingerPrints################
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw, PyMol, rdFMCS
from rdkit.Chem.Draw import IPythonConsole
from rdkit import rdBase
####Finger Prints Based Predications####################
####Create FingerPrints#################################
from rdkit.Chem import AllChem
from rdkit import DataStructs

class FP:
  def __init__(self, fp):
        self.fp = fp
  def __str__(self):
      return self.fp.__str__()

def computeFP(x):
    #compute depth-2 morgan fingerprint hashed to 1024 bits
    fp = AllChem.GetMorganFingerprintAsBitVect(x,2,nBits=1024)
    res = np.zeros(len(fp),np.int32)
    #convert the fingerprint to a numpy array and wrap it into the dummy container
    DataStructs.ConvertToNumpyArray(fp,res)
    return FP(res)
###############################################################################
def addExplFP(df,molColumn):
    fpCache = []
    for mol in df[molColumn]:
        res = AllChem.GetMorganFingerprintAsBitVect(mol,4,nBits=2048)
        fpCache.append(res)
    arr = np.empty((len(df),), dtype=np.object)
    arr[:]=fpCache
    S =  pd.Series(arr,index=df.index,name='explFP')
    return df.join(pd.DataFrame(S))
###############################################################################
def convertToNumpy(df,fpCol):
    fpCache = []
    for fp in df[fpCol]:
        res = np.zeros(len(fp),np.int32)
        DataStructs.ConvertToNumpyArray(fp,res)
        fpCache.append(res)
    '''
    it is necessary to constructs an empty object array in advance and fill that later,
    because directly initializing an array with the fingerprint would trigger the numpy
    type recognition and result in a array of integers that again would trigger pandas
    to construct a Series object per bit position
    '''
    arr = np.empty((len(df),), dtype=np.object)
    arr[:]=fpCache
    S =  pd.Series(arr,index=df.index,name='npFP')
    return df.join(pd.DataFrame(S))
###############################################################################
########Initialize RDKIT Molecular Descriptors Creation Object#################
rdkit_featurizer = deepchem.feat.RDKitDescriptors()
allowedDescriptors = [name[0] for name in rdkit_featurizer.descList]
###############################################################################
#Read and process CSV inputs
#Columns headers are specific to DrugBank Data, for different data files, you can insert the 
#SMILES to the corresponding column and poulate the rest of files by fake data
###############################################################################
drug_bank_df = pd.read_csv(STRUCTURES_FILE,nrows=ROWS_TO_READ)
drug_bank_df.columns = drug_bank_df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')
drug_bank_df_selected_cols = drug_bank_df[['drugbank_id','name','smiles','pubchem_substance_id','drug_groups']]
drug_bank_df_selected_cols.dropna(subset = ['smiles'],inplace=True)
drug_bank_df_selected_cols_feat = deepchem.data.data_loader.featurize_smiles_df(drug_bank_df_selected_cols,rdkit_featurizer,'smiles')
drug_bank_df_selected_cols_feat_df = pd.DataFrame(drug_bank_df_selected_cols_feat[0],columns=allowedDescriptors)
drug_bank_df_selected_cols_featurized = pd.concat([drug_bank_df_selected_cols.reset_index(), drug_bank_df_selected_cols_feat_df.reset_index()], axis=1, join='inner')
drug_bank_df_selected_cols_featurized.dropna(inplace=True)

#Control the range of drug likeness properties you want, consstants can be specified at the begining of the script
mask1 = drug_bank_df_selected_cols_featurized.MolLogP.between(LOGP_LOWER_BOUND,LOGP_UPPER_BOUND)
mask2 = drug_bank_df_selected_cols_featurized.MolWt.between(MWT_LOWER_BOUND,MWT_UPPER_BOUND)

drug_bank_df_selected_cols_featurized_filtered = drug_bank_df_selected_cols_featurized[(mask1) & (mask2)]

#Compute the ECFP6 fingerprints
PandasTools.AddMoleculeColumnToFrame(drug_bank_df_selected_cols_featurized_filtered,smilesCol='smiles',molCol='molecule',includeFingerprints=True)
drug_bank_df_selected_cols_featurized_filtered.dropna(inplace=True)
###############################################################################
drug_bank_df_selected_cols_featurized_filtered_subset = drug_bank_df_selected_cols_featurized_filtered.iloc[:,0:6]
drug_bank_pred_fp = drug_bank_df_selected_cols_featurized_filtered.iloc[:,[0,1,2,3,118]]
drug_bank_pred_fp = addExplFP(drug_bank_pred_fp,'molecule')
drug_bank_pred_fp = convertToNumpy(drug_bank_pred_fp,'explFP')
X_drug_bank_fp_pred = drug_bank_pred_fp[['npFP']]
####################################################
#Assert that filtered input ligands dataframe has the same shape as the fingerprint generated dataframe
if drug_bank_df_selected_cols_featurized_filtered_subset.shape[0] != X_drug_bank_fp_pred.shape[0]:
    print("################\nError Found:\nThe dataframe is not equal to the fingerprint dataframe, check why !\n################")
    sys.exit()
else:
    pass
    
###############################################################################
###############################################################################
# load json and create model
json_file = open('dl_model_fp.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
dl_model_fp = model_from_json(loaded_model_json)
# load weights into new model
dl_model_fp.load_weights("dl_model_fp.h5")
print("Loaded model from disk")
dl_model_fp_prediction = dl_model_fp.predict_classes(np.vstack(X_drug_bank_fp_pred['npFP']))
dl_model_fp_prediction_proba = dl_model_fp.predict(np.vstack(X_drug_bank_fp_pred['npFP']))
del dl_model_fp
print ("dl_model_fp prediction made")

xgb_fp = pickle.load(open("xgb_rdkit_classify_fp.sav", 'rb'))
xgb_fp_prediction = xgb_fp.predict(np.vstack(X_drug_bank_fp_pred['npFP']))
xgb_fp_prediction_proba = xgb_fp.predict_proba(np.vstack(X_drug_bank_fp_pred['npFP']))
del xgb_fp 
print ("xgb_fp prediction made")

if args.ignore_rf_svm == False:
    rfc_fp = pickle.load(open("rfc_rdkit_classify_fp.sav", 'rb'))
    rfc_fp_prediction = rfc_fp.predict(np.vstack(X_drug_bank_fp_pred['npFP']))
    rfc_fp_prediction_proba = rfc_fp.predict_proba(np.vstack(X_drug_bank_fp_pred['npFP']))
    del rfc_fp 
    print ("rfc_fp prediction made")
    
    mlp_rdkit_classify_fp = pickle.load(open("mlp_rdkit_classify_fp.sav", 'rb'))
    mlp_rdkit_classify_fp_prediction = mlp_rdkit_classify_fp.predict(np.vstack(X_drug_bank_fp_pred['npFP']))
    mlp_rdkit_classify_fp_prediction_proba = mlp_rdkit_classify_fp.predict_proba(np.vstack(X_drug_bank_fp_pred['npFP']))
    del mlp_rdkit_classify_fp
    print ("mlp_rdkit_classify_fp prediction made")

    svm_rdkit_classify_fp = pickle.load(open("svm_rdkit_classify_fp.sav", 'rb'))
    svm_rdkit_classify_fp_prediction = svm_rdkit_classify_fp.predict(np.vstack(X_drug_bank_fp_pred['npFP']))
    svm_rdkit_classify_fp_prediction_proba = svm_rdkit_classify_fp.decision_function(np.vstack(X_drug_bank_fp_pred['npFP']))
    del svm_rdkit_classify_fp
    print ("svm_rdkit_classify_fp made")

else:
    pass
###############################################################################
drug_bank_df_selected_cols_featurized_filtered_subset['prediction_class_dl_model_fp_prediction'] = dl_model_fp_prediction
drug_bank_df_selected_cols_featurized_filtered_subset['prediction_class_dl_model_fp_prediction_proba'] = dl_model_fp_prediction_proba.max()
############3
drug_bank_df_selected_cols_featurized_filtered_subset['prediction_class_xgb_fp_prediction'] = xgb_fp_prediction
drug_bank_df_selected_cols_featurized_filtered_subset['prediction_class_xgb_fp_prediction_proba'] = xgb_fp_prediction_proba.max()


if args.ignore_rf_svm == False:
    drug_bank_df_selected_cols_featurized_filtered_subset['prediction_class_rfc_fp_prediction'] = rfc_fp_prediction
    drug_bank_df_selected_cols_featurized_filtered_subset['prediction_class_rfc_fp_prediction_proba'] = rfc_fp_prediction_proba.max()
    #################
    drug_bank_df_selected_cols_featurized_filtered_subset['prediction_class_mlp_fp_prediction'] = mlp_rdkit_classify_fp_prediction
    drug_bank_df_selected_cols_featurized_filtered_subset['prediction_class_mlp_fp_prediction_proba'] = mlp_rdkit_classify_fp_prediction_proba.max()
    #################
    drug_bank_df_selected_cols_featurized_filtered_subset['prediction_class_svm_fp_prediction'] = svm_rdkit_classify_fp_prediction
    drug_bank_df_selected_cols_featurized_filtered_subset['prediction_class_svm_fp_prediction_proba'] = svm_rdkit_classify_fp_prediction_proba.max()
else:
    pass
#################
#################
######################################################################################
######################################################################################
######################################################################################
coded_labels = pd.read_csv("coded_gpcr_list.csv")
merged_predictions_fullcols = pd.merge(drug_bank_df_selected_cols_featurized_filtered_subset,coded_labels,how="inner", left_on="prediction_class_dl_model_fp_prediction", right_on="gpcr_binding_encoded")
merged_predictions_fullcols.drop_duplicates(inplace=True)
######################################################################################
print(merged_predictions_fullcols.columns)
######################################################################################
merged_predictions_selcols = merged_predictions_fullcols[['drugbank_id', 'name', u'pubchem_substance_id', 'drug_groups', 'prediction_class_dl_model_fp_prediction', 'prediction_class_dl_model_fp_prediction_proba', 'prediction_class_mlp_fp_prediction', 'prediction_class_mlp_fp_prediction_proba', 'prediction_class_xgb_fp_prediction', 'prediction_class_xgb_fp_prediction_proba', 'prediction_class_rfc_fp_prediction', 'prediction_class_rfc_fp_prediction_proba', 'prediction_class_svm_fp_prediction', 'prediction_class_svm_fp_prediction_proba', 'first_seg','gpcr_binding_encoded']]


if args.ignore_rf_svm == False:
    cols = ['prediction_class_dl_model_fp_prediction','prediction_class_mlp_fp_prediction','prediction_class_xgb_fp_prediction', 'prediction_class_rfc_fp_prediction','prediction_class_svm_fp_prediction']
    consensus_prediction_df = merged_predictions_selcols[merged_predictions_selcols[cols].eq(merged_predictions_selcols[cols[0]], axis=0).all(axis=1)]
    print('ignore is false')
else:
    cols = ['prediction_class_dl_model_fp_prediction','prediction_class_xgb_fp_prediction']
    consensus_prediction_df = merged_predictions_selcols[merged_predictions_selcols[cols].eq(merged_predictions_selcols[cols[0]], axis=0).all(axis=1)]
    print('ignore is true')
###############################################################################
consensus_prediction_df.drop_duplicates().to_csv("{0}".format(OUTPUT_FILE))
print('########################')
print ("Mission Accomplished and prediction made")

print("If you find these predictions useful, please cite the following article:\n\
      M. Ahmed, H. Jalily, S. Kalayaanoorty and K. Barakat, (2020) XX,XXX,XXX ....")









