# GPCR_LigandClassify

===================================

** The following python libraries are required to run the models: **

* Python 2.7 (tested with Anaconda distribution, Linux OS (Mint 19.1 for a local PC, and 3.10.0-957.12.2.el7.x86_64 GNU/Linux for ComputeCanada clusters). Running the models on MAC may be cumbersome because of the recent XGBoost updates. We did not test the prediction on Windows.)
* DeepChem 1.x (Requires RDKit)
* Pandas (Prediction is tested with Pandas 0.22)
* Tensorflow 1.3
* Keras
* XGBoost
* ScikitLearn

** In order for the script to run, and in addition to the input file (see below), the following files should exist in the running directory: **
 
* dl_model_fp.json
* dl_model_fp.h5
* mlp_rdkit_classify_fp.sav
* xgb_rdkit_classify_fp.sav
* rfc_rdkit_classify_fp.sav
* svm_rdkit_classify_fp.sav
* coded_gpcr_list.csv

NB: The rfc_rdkit_classify_fp.sav & svm_rdkit_classify_fp.sav & mlp_rdkit_classify_fp.sav models are required only if the [--ignore_rf_svm argument] option in the script is set to False (True is the default behaviour)
. The models are not deposited in the github repository because of size limits, to get these two models a direct request should be sent to mmahmed@ualberta.ca & kbarakat@ualberta.ca

################################

** This is how you can use the program using the models to make novel predictions: **
```
python GPCR_LigandClassify.py --input_file input.csv --output_file output.csv [--n_rows_to_read <INTEGER>] [--mwt_lower_bound <FLOAT>] [--mwt_upper_bound <FLOAT>] [--logp_lower_bound <FLOAT>] [--logp_upper_bound <FLOAT>] [--ignore_rf_svm <True/False>]
```
    
## Important:   
*The input & output file names arguments are mandatory arguments, --n_rows_to_read argument determines how many rows you want to read from the input CSV files (default 9999999999 rows)
, the rest are optional with default same as input dataset used for models training.

*The --ignore_rf_svm argument will ignore the RF, the SVM and the MLP models which are pretty large, suitable in case of limited computational resourcses, particularly memory. Default is True (Ignore Randomforests and SVM models.). These models can be requested from the authors: Khaled Barakat (kbarakat@ualberta.ca) and Marawan Ahmed (mmahmed@ualberta.ca).

*Please note that a today date string will be attached to the output file name.

*Please note that the script will only save ligands where all predictions agree.

For the input file, please keep the same format as the attached sample input file (drug bank data file). In case of data coming from different source, with the exception of the SMILES column, other columns may be left blank (not recommended). You can populate the rest of columns with fake data.

*For the models and auxiliary files, please visit the following github repository:
      https://github.com/mmagithub/GPCR_LigandClassify

## Credits

* Main contributor: Marawan Ahmed (mmahmed@ualberta.ca).
* Models were dveloped using Computecanada clusters.
* Code/Models/Data are distributed under the standard MIT license for non-commercial users. Commercial users should contact the authors.
* If you find these predictions useful, please cite the following article:
```
Marawan Ahmed, H. Jalily, S. Kalyaanamoorthy and K. Barakat, “GPCR_LigandClassify.py; a rigorous machine learning classifier for GPCR targeting compounds”, Scientific reports 11.1 (2021): 1-17.
```
