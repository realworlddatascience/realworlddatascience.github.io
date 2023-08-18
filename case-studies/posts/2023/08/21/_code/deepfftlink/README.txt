Hello!


Below you can find a outline of how to reproduce our solution for the first interim challenge.
If you run into any trouble with the setup/code or have any questions please contact me at ywu19@wpi.edu

#ARCHIVE CONTENTS   The detailed instructions for how to use them is in "directory_structure.txt" file.
The scripts are in ./code directory    
The data are in ./final_data directory

Please put Bert and roberta model in separate folders under this directory.

Please use the below link to download bert and roberta model folder in this directory.
git lfs clone https://huggingface.co/bert-base-uncased
git lfs clone https://huggingface.co/roberta-base

nltk_data folder can be download here: https://www.nltk.org/nltk_data/
We will only use chunkers and corpora sub folder in nltk_data folder.


Please download raw data from databse into "final_data/" folder. Here is the necessary raw data list:

2015-2016:
"addfooddesc1516.csv"
"fnddsingred1516.csv"
"mainfooddesc1516.csv"
"pd_pos_all1516.csv"
"ppc20152016.csv"


2017-2018:
"ec.csv"
"ingredient.csv"
"pd_pos_all1718_public.csv"
"ppc20172018_publictest.csv"



#HARDWARE: 
workspace

#SOFTWARE (python packages are detailed separately in `requirements.txt`):
Python 3.8.12
transformers  4.18.0
You can also   "conda activate deep-fft-link"


#DATA SETUP 
All datasets are installed from competition database.

#DATA PROCESSING

upc:  please check ./code/1718upc.ipynb     #When save data, please remove "#" in this ipynb file to save
ec: please check ./code/1718ec.ipynb        #When save data, please remove "#" in this ipynb file to save                                   


#MODEL BUILD: There are three options to produce the solution.
The most important functions are in ./code/model.py file
Model to train and predict:       ./code/final_model.ipynb    # as we suggest, you can split the upc set as 32 subsets (each has 1200 upc). They are in ./final_data/subsets/  .
                                  ./code/(1).ipynb      Then, You can use 5 ipynb files (1.ipynb, 2.ipynb...) The workspace GPU allows to run at most 5 at the same time. The total expected time is 6 hours for 32 subsets. (run a single one cost near 1 hour)


#shell command to run (is not suggested, That use ipynb file split upc to subsets will save much time!)
In final_model.ipynb file, change three datasets with new test datasets:
public_upc = pd.read_csv(data_path + "pd_pos_all1718_public.csv") #Original upc file
upc = pd.read_csv(data_path + "1718upc_cleaned.csv") #cleaned upc
ec = pd.read_csv(data_path + "1718ec_cleaned.csv") #cleaned ec


please make sure your data is well cleaned and files should located on ./final_data/
For security reason, the data is removed.


# The detailed information of how to use each code is in directory_structure.txt file.
Thank you!