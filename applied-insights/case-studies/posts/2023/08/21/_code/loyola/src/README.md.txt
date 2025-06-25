<!-- ABOUT THE PROJECT -->
## About The Project

This project is the Loyola Marymount University submission for the 2023 Food for Thought challenge, in which we receieved third place with our approach using pre-trained BERT for binary sequence classification.

<!-- GETTING STARTED -->
## Getting Started

Please note that all data and data pre-processing scripts have been removed for security purposes. This works runs on either a CPU or a GPU.

### Directory Structure
The source code is all in the `src` directory. Input and output data files should go inside a `data` directory. Model checkpoints should go inside a `models` directory.

### Installation

1. Be sure you have Python 3 installed.
2. Install packages with pip
   ```sh
   pip install torch time datetime random pandas numpy heapq tqdm sklearn keras transformers
   ```

<!-- USAGE EXAMPLES -->
## Usage

Training: Fine-tunes BERT for binary sequence classification.
  ```
  python 01_train_bert.py
  ```
  Input: `data/nn_train_data.csv`
  Output: `models/model.ckpt`

Note: To change how much data is used to train the BERT model, modify line 56.
  
Predicting: Outputs predictions using a fine-tuned BERT model.
  ```
  python 02_predict_with_bert_batches.py
  ````
  Input:
    data/cleaned1.csv
    data/cleaned2.csv
    data/ppc20172018_publictest.csv
  Output: `data/bert_predicted.csv` OR `data/bert_predicted_slow.csv`

Evaluation: Outputs the Success@5 and NDCG@5 metrics.
  `python 03_evaluation_script.py data/bert_predicted.csv data/ground_truth.csv` [for fast, complete evaluation]
  `python 03_evaluation_script.py data/bert_predicted_slow.csv data/ground_truth_slow.csv` [for slow evaluation on a subset of test data]
  Input:
    data/bert_predicted.csv
    data/ground_truth.csv

<!-- CONTACT -->
## Contact

Mandy Korpusik - Mandy.Korpusik@lmu.edu
Yifan (Rosetta) Hu - rosettahu@gmail.com