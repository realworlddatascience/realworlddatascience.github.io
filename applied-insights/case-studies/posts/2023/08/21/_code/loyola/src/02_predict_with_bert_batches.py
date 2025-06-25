"""
Author: Mandy Korpusik
Date: September 12, 2022
Description: Uses fine-tuned BERT to select top-5 matching codes.
"""

import torch
import heapq
import time
import numpy as np
import pandas as pd

from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from keras.preprocessing.sequence import pad_sequences
from transformers import BertTokenizer, BertForSequenceClassification


RUN_FAST = False # When set to True, optimizes for speed instead of accuracy.
PATH = '/'


def encode_seqs(desc1, matches):
    """Encode food descriptions as tokenized IDs and attentions masks."""
    inputs = []
    for desc2 in matches:
        encoding = tokenizer.encode(desc1, desc2, add_special_tokens = True)
        inputs.append(encoding)
            
    inputs = pad_sequences(inputs, maxlen=240, dtype='long', 
                           value=tokenizer.pad_token_id, truncating='post', padding='post')
            
    # Create attention masks for all sequences.
    masks = []
    for sent in inputs:
        att_mask = [int(token_id > 0) for token_id in sent]
        masks.append(att_mask)
    return torch.tensor(inputs), torch.tensor(masks)
    

def get_predictions(inputs, mask):
    """Feed encoded input seqs into pre-trained BERT and output logits."""
    with torch.no_grad():     
        outputs = model(inputs, 
                        token_type_ids=None, 
                        attention_mask=mask)
        
    logits = outputs[0]
    logits = logits.detach().cpu().numpy()
    return logits.flatten()


def closest_description(data2, data1, start_index=0, end_index=1, similarity=0.5):
    """Get top-5 matches for each input food, using BERT."""
    closest_code = []
    closest_desc = []
    highest_conf = []
    code_list = []
    desc_list = []
    
    # Iterate through test data1 table and compare desc with each data2 description.
    for index, row in tqdm(data1[start_index:end_index].iterrows()):
        desc1 = row.description1
        code1 = row.code1
        
        # Use blocking (i.e., matching words from first 6 tokens) to narrow down potential matches. Select only 50 of those.
        if RUN_FAST:
            matches = data2[data2.description2.str.contains('|'.join(desc1.split()[:6]))].head(10)
        else:
            matches = data2[data2.description2.str.contains('|'.join(desc1.split()[:6]))]
        
        if matches.shape[0] == 0:
            highest_conf.append(1.0)
            desc_list.append(desc1)
            code_list.append(code1)
            closest_code.append(-99)
            closest_desc.append('No Match')
            continue
        
        # Prepare sequences to feed into BERT and get output confidence scores.
        inputs, masks = encode_seqs(desc1, matches.description2)
        predictions = get_predictions(inputs, masks)
        
        # Determine top-5 matching descriptions.
        index = min(5, len(predictions))
        top5_indices = np.argpartition(predictions, -index)[-index:]
        top5_indices = top5_indices[np.argsort(predictions[top5_indices])][::-1] # sort indices (descending)
        
        # Extend closest_code, closest_desc, closest_distance
        highest_conf.extend(predictions[top5_indices])
        desc_list.extend([desc1] * index)
        code_list.extend([code1] * index)
        closest_code.extend(matches.iloc[list(top5_indices)].code2)
        closest_desc.extend(matches.iloc[list(top5_indices)].description2)
    
    closest_df = pd.DataFrame({'code1':code_list, 'desc1':desc_list, 
                               'code2':closest_code,'desc2':closest_desc, 'confidence':highest_conf})
    return closest_df
            

data1 = pd.read_csv(PATH + 'data/cleaned1.csv', dtype=str)
data1 = data1[data1['description'].notna()] # Filter out nan.
data2 = pd.read_csv(PATH + 'data/cleaned2.csv', dtype=str)
ppc = pd.read_csv(PATH + 'data/ppc20172018_publictest.csv', dtype=str)

# Set up GPU.
if torch.cuda.is_available():
    device = torch.device('cuda')
    print('Using GPU:', torch.cuda.get_device_name(0))
else:
    device = torch.device('cpu')
    print('No GPU available, using the CPU instead.')

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

# Load trained BERT model.
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased', # Use the 12-layer BERT model, with an uncased vocab.
    num_labels = 1, # The number of output labels--2 for binary classification.
    output_attentions = False, # Whether the model returns attentions weights.
    output_hidden_states = False, # Whether the model returns all hidden-states.
)
model.to(device)
model.eval()
checkpt = torch.load(PATH + '/models/model.ckpt')
model.load_state_dict(checkpt['model_state_dict'])

# Load previously saved csv into a dataframe.
filename = PATH + 'data/bert_predicted_slow.csv'
if RUN_FAST:
    filename = PATH + 'data/bert_predicted.csv'
match_df =  pd.read_csv(filename, dtype=str)
n = int(match_df.shape[0] / 5)

start = time.time()

# Save to csv file every 5 codes.
for start_index in range(n, data1.shape[0], 5):
    new_match_df = closest_description(data2, data1, start_index=start_index, end_index=start_index+5)
    match_df = pd.concat([match_df, new_match_df])
    print(match_df.tail())
    print(match_df.shape)
    
    # Output new predictions (combined with previous) and run evaluation script.
    match_df[['code1', 'code2', 'confidence']].to_csv(filename, index=False)

end = time.time()
print('total time:', end - start)
