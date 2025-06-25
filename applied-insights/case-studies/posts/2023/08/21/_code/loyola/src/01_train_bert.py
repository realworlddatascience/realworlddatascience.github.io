"""
Author: Mandy Korpusik
Date: June 30, 2022
Description: Trains BERT for binary sequence classification on pairs of food descriptions.
Reference: https://colab.research.google.com/drive/1pfGM3xPeDwxRbmW_bMso9Z45a7Kzy8X2#scrollTo=CvIwHugRXbnp
"""

import torch
import time
import datetime
import random
import pandas as pd
import numpy as np

from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from transformers import WarmupLinearSchedule as get_linear_schedule_with_warmup
from keras.preprocessing.sequence import pad_sequences
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split


PATH = 'models/'
load_model = False


def flat_accuracy(preds, labels):
    """Calculates the accuracy of our predictions vs labels."""
    # pred_flat = np.argmax(preds, axis=1).flatten()
    pred_flat = (preds.flatten() >= 0.5).astype(int)
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def format_time(elapsed):
    """Takes a time in seconds and returns a string hh:mm:ss"""
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

# Set up GPU.
if torch.cuda.is_available():
    device = torch.device('cuda')
    print('Using GPU:', torch.cuda.get_device_name(0))
else:
    device = torch.device('cpu')
    print('No GPU available, using the CPU instead.')


# Add 5 randomly sampled negative examples per positive example, using preprocessed data.
df = pd.read_csv('P:/pr-westat-fft-loyola-marymount/baseline_model/nn_train_data.csv', dtype=str)
df = df.dropna()

# TODO: Remove clipping and save prepared data for faster processing in the future.
df = df.sample(48000, random_state=2498).iloc[32000:48000]

data1_sents = df.description1.values
data2_sents = df.description2.values

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

# Encode each pair of food descriptions with BertTokenizer.
inputs = []
labels = []
for sent1, sent2, label in zip(data1_sents, data2_sents, df.dictionary.values):
    encoding = tokenizer.encode(sent1, sent2, add_special_tokens = True)
    inputs.append(encoding)
    
    if label == 'POS':
        labels.append(1)
    else:
        labels.append(0)
      
# Pad all sequences to the maximum length.      
MAX_LEN = max([len(sen) for sen in inputs])
print('Max sentence length: ', MAX_LEN)
inputs = pad_sequences(inputs, maxlen=MAX_LEN, dtype='long', 
                       value=tokenizer.pad_token_id, truncating='post', padding='post')
                          
# Create attention masks for all sequences.
masks = []
for sent in inputs:
    att_mask = [int(token_id > 0) for token_id in sent]
    masks.append(att_mask)
    
# Split input token IDs and attention masks into train/validation sets.
train_x, val_x, train_y, val_y = train_test_split(inputs, labels, random_state=2498, test_size=0.1)
train_mask, val_mask, _, _ = train_test_split(masks, labels, random_state=2498, test_size=0.1)

# Convert data to Torch tensors.
train_x = torch.tensor(train_x)
val_x = torch.tensor(val_x)
train_y = torch.tensor(train_y).to(torch.float32)
val_y = torch.tensor(val_y).to(torch.float32)
train_mask = torch.tensor(train_mask)
val_mask = torch.tensor(val_mask)

# Load training and validation data in batches.
batch_size = 16 # TODO: try 32
train_data = TensorDataset(train_x, train_mask, train_y)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
val_data = TensorDataset(val_x, val_mask, val_y)
val_sampler = RandomSampler(val_data)
val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size)
print(train_x.shape, train_mask.shape, train_y)

model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased', # Use the 12-layer BERT model, with an uncased vocab.
    num_labels = 1,
    output_attentions = False, # Whether the model returns attentions weights.
    output_hidden_states = False, # Whether the model returns all hidden-states.
)
model.to(device)

# Set the optimizer and learning rate.
optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
epochs = 2 # TODO: try 3 and 4
total_steps = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps=0, t_total=total_steps)

# Training loop!
seed_val = 42
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)
loss_values = []
elapsed_time = 0
start_epoch = 0
start_step = 0

# Load model
if load_model:
    print('Loading model')
    checkpt = torch.load(PATH + 'model.ckpt')
    model.load_state_dict(checkpt['model_state_dict'])
    optimizer.load_state_dict(checkpt['optimizer'])
    scheduler.load_state_dict(checkpt['scheduler'])
    start_epoch = checkpt['epoch']
    elapsed_time = checkpt['time']
    loss_values = checkpt['loss_values']
    loss = checkpt['loss']
    start_step = checkpt['step']
    batch_size = checkpt['batch_size']
    if start_step == 895:
        start_step = 0
    print(f'Batch size: {batch_size}')
    print(f'Epoch: {start_epoch}')
    print(f'Step: {start_step}')
    print(f'Elapsed: {elapsed_time}')

for epoch_i in range(start_epoch, epochs):
    # ========================================
    #               Training
    # ========================================
    print('\n======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    t0 = time.time()
    total_loss = 0
    if load_model:
        total_loss = checkpt['loss']
    model.train()
    for step, batch in enumerate(train_dataloader):
        if step < start_step:
            continue
        if (step % 5 == 0) and (not step == 0):
            elapsed = format_time(time.time() - t0)
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))
            print(f'\n Most recent loss: {loss}')
            
            # Save model checkpoint.
            torch.save({'epoch': epoch_i, 'model_state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(), 'loss': total_loss, 'step': step,
                        'scheduler': scheduler.state_dict(), 'time': time.time() - t0 + elapsed_time, 
                        'loss_values': loss_values, 'batch_size': batch_size},  PATH + 'model.ckpt')

        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        
        model.zero_grad()  
        outputs = model(b_input_ids, 
                        token_type_ids=None, 
                        attention_mask=b_input_mask, 
                        labels=b_labels)

        loss = outputs[0]
        total_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
    avg_train_loss = total_loss / len(train_dataloader)
    loss_values.append(avg_train_loss)
    print(outputs)
    print(b_labels)
    print('\n  Average training loss: {0:.2f}'.format(avg_train_loss))
    print('  Training epoch took: {:}'.format(format_time(time.time() - t0 + elapsed_time)))
    
    # ========================================
    #               Validation
    # ========================================
    print('\nRunning Validation...')
    t0 = time.time()
    model.eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    
    # Evaluate data for one epoch.
    for batch in val_dataloader:
        
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        
        with torch.no_grad():     
            outputs = model(b_input_ids, 
                            token_type_ids=None, 
                            attention_mask=b_input_mask)
        
        logits = outputs[0]
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        
        # Calculate the accuracy for this batch of test sentences.
        tmp_eval_accuracy = flat_accuracy(logits, label_ids)
        
        # Accumulate the total accuracy.
        eval_accuracy += tmp_eval_accuracy
        nb_eval_steps += 1

    print('  Accuracy: {0:.2f}'.format(eval_accuracy/nb_eval_steps))
    print('  Validation took: {:}'.format(format_time(time.time() - t0)))
    start_step = 0
