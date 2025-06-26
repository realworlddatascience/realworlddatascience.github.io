import pandas as pd
import numpy as np
import sys


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


if __name__ == "__main__":
    submission_file = sys.argv[1]
    ground_truth_file = sys.argv[2]
    
    submission = pd.read_csv(submission_file, dtype=str)
    ground_truth = pd.read_csv(ground_truth_file, dtype=str)
    
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
    
    print("The NDCG@5 score is: {}".format(round(ndcg_score, 3)))
    print("The Success@5 score is: {}".format(round(success_score, 3)))