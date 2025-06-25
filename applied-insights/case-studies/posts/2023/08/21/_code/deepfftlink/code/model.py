import pandas as pd
import numpy as np
import random
import torch
import heapq
from tqdm import tqdm
from transformers.file_utils import is_tf_available, is_torch_available
from transformers import BertTokenizer, BertModel
from transformers import RobertaTokenizer, RobertaModel
import warnings 
from sklearn.metrics.pairwise import cosine_similarity
import re
import heapq
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import logging as lo
lo.set_verbosity_error()
warnings.filterwarnings('ignore')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# function set
def set_seed(seed: int):
    """
    Helper function for reproducible behavior to set the seed in ``random``, ``numpy``, ``torch`` and/or ``tf`` (if
    installed).

    Args:
        seed (:obj:`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    if is_torch_available():
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # ^^ safe to call this function even if cuda is not available
    if is_tf_available():
        import tensorflow as tf

        tf.random.set_seed(seed)

set_seed(1)
def load_model(model_name):
    if model_name == "bert-base-uncased":
        tokenizer = BertTokenizer.from_pretrained('../bert-base-uncased', do_lower_case=True)
        model = BertModel.from_pretrained("../bert-base-uncased")
    elif model_name == "roberta-base":
        tokenizer = RobertaTokenizer.from_pretrained('../roberta-base',do_lower_case = True)
        model = RobertaModel.from_pretrained('../roberta-base')
    return tokenizer, model
def ngrams_analyzer(string):
    string = re.sub(r'[,-./]', r'', string)
    ngrams = zip(*[string[i:] for i in range(3)])  # N-Gram length is 3
    return [''.join(ngram) for ngram in ngrams]

def closest_description(ec, upc, similarity=0):
    vectorizer = TfidfVectorizer(analyzer=ngrams_analyzer)
    
    #Apply the defined vectorizer
    ec_desc = ec.ec_description
    tfidf_matrix = vectorizer.fit_transform(ec_desc)
    #Calculate the closest distance for each word
    closest_desc=[]
    closest_distance=[]
    upc_desc_list=[]
    # Iterate through the UPC table and compare the descriptions with each EC description
    for index, row in upc.iterrows():
        upc_desc = [row.upc_description]
        upc_desc = vectorizer.transform(upc_desc)
        cos_sim = cosine_similarity(upc_desc, tfidf_matrix)
        max_ = heapq.nlargest(len(ec_desc), cos_sim[0])
        closest_index = [(i, j) for i, j in enumerate(cos_sim[0]) if j in max_]
        if len(closest_index) > len(ec_desc):
            closest_index = closest_index[:len(ec_desc)]
        closest_desc.extend([ec_desc[x[0]] for x in closest_index])
        closest_distance.extend([x[1] for x in closest_index])
        upc_desc_list.extend([row.upc_description] * len(ec_desc))
    
    closest_df=pd.DataFrame({'upc_desc':upc_desc_list,'closest_desc':closest_desc,'closest_distance':closest_distance})
    # The record will be labelled no match if the similarity is below certain threshold
    closest_df['closest_desc']=[row.closest_desc if row.closest_distance>=similarity else 'No Match' for index, row in closest_df.iterrows() ]
    return closest_df


def with_code(match_df, ec, upc):
    # This function filters out unmatched pairs and connects the description to the corresponding codes
    with_code = match_df[~(match_df.closest_desc == 'No Match')]
    with_code = with_code.merge(ec, left_on='closest_desc', right_on='ec_description', how='left')
    with_code = with_code.merge(upc, left_on='upc_desc', right_on='upc_description', how='left')
    with_code = with_code.drop_duplicates()
   
    return with_code
def get_top_5(table):
    a = table.sort_values(['upc',table.columns[2]],ascending=False).groupby('upc').head(5)
    a = a.reset_index()
    a.drop('index', axis=1, inplace=True)
    return a

#evaluations
def dcg_at_k(r, k=5):
    """
    Args:
        r: Relevance scores list (binary value) in rank order
        k: Number of results to consider

    Returns:
        Discounted Cumulative Gain
    """
    r = np.asfarray(r)[:k]
    if r.size:
        return np.sum(r / np.log2(np.arange(2, r.size + 2))) 
    return 0.


def ndcg_at_k(r, k=5):
    """
    Args:
        r: Relevance scores list (binary value) in rank order
        k: Number of results to consider
    Returns:
        Normalized Discounted Cumulative Gain
    """
    dcg_max = dcg_at_k(sorted(r, reverse=True), k)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k) / dcg_max


def success_at_k(r, k=5):
    """
    Args:
        r: Correct match list (binary value) in rank order
        k: Number of results to consider
    Returns:
    """
    return np.sum(r[:k])


def padding(r, k=5):
    while len(r) < k:
        r.append(0)
    return r
def get_result(submission, ground_truth):
    submission = submission.drop_duplicates()
    df = pd.merge(submission, ground_truth, how='outer', on='upc')
    df = df[df['ec_y'].notna()]
    df['correct'] = df['ec_x'] == df['ec_y']
    df['correct'] = df['correct'].astype(int)
    df = df.groupby('upc').agg(lambda x: x.tolist()).reset_index()
    
    df['correct'] = df['correct'].apply(padding)
    df['ndcg@5'] = df['correct'].apply(ndcg_at_k)
    df['s@5'] = df['correct'].apply(success_at_k)
    
    ndcg_score = df['ndcg@5'].mean()
    success_score = df['s@5'].mean()
    return success_score,ndcg_score

def select_model(model_name,upc,ec):
    upc = upc.reset_index(drop=True)
    ec = ec.reset_index(drop=True)
    if model_name == "bert":
        tokenizer, model = load_model("bert-base-uncased")
        sent =upc["upc_description"].tolist()+ec["ec_description"].tolist()
        # initialize dictionary: stores tokenized sentences
        token = {'input_ids': [], 'attention_mask': []}
        for sentence in sent:
            # encode each sentence, append to dictionary
            with torch.no_grad():
                new_token = tokenizer.encode_plus(sentence, max_length=10,truncation=True, padding='max_length',return_tensors='pt').to(device)
            token['input_ids'].append(new_token['input_ids'][0])
            token['attention_mask'].append(new_token['attention_mask'][0])
        # reformat list of tensors to single tensor
        token['input_ids'] = torch.stack(token['input_ids'])
        token['attention_mask'] = torch.stack(token['attention_mask'])
        model = model.to(device)
        with torch.no_grad():
            output = model(**token)
        embeddings = output.last_hidden_state
        # To perform this operation, we first resize our attention_mask tensor:
        att_mask = token['attention_mask']
        mask = att_mask.unsqueeze(-1).expand(embeddings.size()).float()
        mask_embeddings = embeddings * mask
        #Then we sum the remained of the embeddings along axis 1:
        summed = torch.sum(mask_embeddings, 1)
        #Then sum the number of values that must be given attention in each position of the tensor:
        summed_mask = torch.clamp(mask.sum(1), min=1e-9)
        mean_pooled = summed / summed_mask
        # convert from PyTorch tensor to numpy array
        mean_pooled = mean_pooled.detach().cpu().numpy()
        # calculate cosine similarity between upc and each ec
        result = pd.DataFrame(columns = ["upc", "ec","bert_confidence"])
        for i in range(upc.shape[0]):
            list1 = cosine_similarity([mean_pooled[i]],mean_pooled[upc.shape[0]:]).tolist()[0]
            max_5 = heapq.nlargest(ec.shape[0],list1)
            closest_index = [(a, b) for a, b in enumerate(list1) if b in max_5]
            closest_index = sorted(closest_index,key = lambda x: x[1],reverse = True)
            table = pd.DataFrame(columns = ["upc", "ec","bert_confidence"], index = range(ec.shape[0]))
            for j in range(ec.shape[0]):
                table.iloc[j,0] = upc.iloc[i,0]
                table.iloc[j,1] = ec["ec_code"][closest_index[j][0]]
                table.iloc[j,2] = closest_index[j][1]
            result = pd.concat([result,table])
        return result
    elif model_name == "string":
        # Reset index so it's incremental
        upc = upc.reset_index(drop=True)
        ec = ec.reset_index(drop=True)
        # Fill missing value with empty string
        upc = upc.fillna("")
        match_df = closest_description(ec, upc)
        result = with_code(match_df, ec, upc)
        result = result[['upc_code', 'upc_description', 'ec_code', 'ec_description', 'closest_distance']]
        # Clean the table in the format of PPC table
        result = result.rename(columns={'upc_code': 'upc', 'ec_code': 'ec','closest_distance':"string_confidence"})
        clean_result = result[['upc', 'ec', 'string_confidence']]
        return clean_result
def test_other_data(upc,ec,a,b):
    result1 = select_model("bert",upc,ec)
    result2 = select_model("string",upc,ec)
    result3=pd.merge(result2,result1)
    result3["bert_confidence"] = result3["bert_confidence"].astype(float)
    result3["confidence"] = a * result3["string_confidence"]+b * result3["bert_confidence"]
    table1 = result3[['upc', 'ec', 'confidence']]
    table2 = get_top_5(table1)
    return table2


## Experiment 1: We set a + b =1
def exp1(table,ppc_clipped,start=0,lr=0.001, condition = "None"):
    a = start
    res = pd.DataFrame(columns = ['a','b','success_score','ndcg_score'])
    pbar = tqdm(desc = "while loop", total = 100)
    while a<=1:
        b= 1-a
        if condition == "None":
            table["confidence"] = a * table["string_confidence"]+b * table["bert_confidence"]
        elif condition == "log2":
            table["confidence"] = a * np.log2(table["string_confidence"])+b * np.log2(table["bert_confidence"])
        elif condition == "log10":
            table["confidence"] = a * np.log10(table["string_confidence"])+b * np.log10(table["bert_confidence"])
        table1 = table[['upc', 'ec', 'confidence']]
        table2 = get_top_5(table1)
        success_score,ndcg_score = get_result(table2, ppc_clipped)
        s = pd.Series([a,b,success_score,ndcg_score],index=['a','b','success_score','ndcg_score'])
        res = res.append(s,ignore_index=True)
        a+=lr
        pbar.update(100/((1-start)/lr+1))
    pbar.close()
    res = res.sort_values(["success_score","ndcg_score"],ascending=False)
    res = res.reset_index()
    res.drop('index', axis=1, inplace=True)
    return res

## Experiment 2: We set a and b are random numbers in range[0,1]. This experiment may include exp1 but have more a,b combinations.
def exp2(table,ppc_clipped,a_start=0,b_start=0,a_lr =0.001,b_lr=0.001,condition = "None"):
    a = a_start
    res = pd.DataFrame(columns = ['a','b','success_score','ndcg_score'])
    pbar = tqdm(desc = "while loop", total = 100)
    while a<=1:
        b = b_start
        while b<=1:
            if condition == "None":
                table["confidence"] = a * table["string_confidence"]+b * table["bert_confidence"]
            elif condition == "log2":
                table["confidence"] = a * np.log2(table["string_confidence"])+b * np.log2(table["bert_confidence"])
            elif condition == "log10":
                table["confidence"] = a * np.log10(table["string_confidence"])+b * np.log10(table["bert_confidence"])
            table1 = table[['upc', 'ec', 'confidence']]
            table2 = get_top_5(table1)
            success_score,ndcg_score = get_result(table2, ppc_clipped)
            s = pd.Series([a,b,success_score,ndcg_score],index=['a','b','success_score','ndcg_score'])
            res = res.append(s,ignore_index=True)
            b += b_lr
        a += a_lr
        pbar.update(100/((1-a_start)/a_lr+1))
    pbar.close()
    res = res.sort_values(["success_score", "ndcg_score"],ascending=False)
    res = res.reset_index()
    res.drop('index', axis=1, inplace=True)
    return res

