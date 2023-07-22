

import imp
from tkinter.ttk import LabeledScale
import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np
#from Constants import *

import transformers 

class TextDataset(Dataset):
    def __init__(self, 
            dataset_file_path, 
            tokenizer,
            max_text_len = 128,
            vocab_file_path=None,
            
            ):
        # Read JSON file and assign to headlines variable (list of strings)
        data = pd.read_csv( dataset_file_path,  encoding = "ISO-8859-1") #sep="\t")
        
        ## for disaster data 
        #data = data[["Title", "Abstract", "disaster_related"]]
        if "disaster_related" in data.columns: 
            data['label'] = data['disaster_related'] 
            print("use disaster label")
            #import pdb; pdb.set_trace()
        if "label" in data.columns:
            data = data[["Title", "Abstract", "label"]]
        else:
            data = data[["Title", "Abstract", ] ] 
        
        data = data.dropna()
        self.title = data["Title"].values.tolist()
        self.abstract = data["Abstract"].values.tolist()
        if "label" in data.columns:
            self.labels = data["label"].values.tolist() 
        else:
            self.labels = [-1] * len(self.abstract) 
        if vocab_file_path is not None: 
            self.label2id = {l.strip():i for i, l in enumerate( open(vocab_file_path, "r").readlines() ) }
        self.tokenizer = tokenizer
        self.max_text_len = max_text_len 

    def __len__(self):
        return len(self.title)

    def tokenize(self, title, abstract):
        CLS_TOKEN = "[CLS]"
        SEP_TOKEN = "[SEP]" 

        title = "Title: " + title.strip()
        abstract = "Abstract: " + abstract.strip() 
        template = "The work is [MASK] related to natural disaster and health."
        #template = "Is the work mainly about natural disaster? Answer: [MASK]." # and health.

        #abstract = abstract.strip().split(" ")[:500]
        title = self.tokenizer.tokenize(title) 
        abstract = self.tokenizer.tokenize(abstract, max_length=self.max_text_len - 2, truncation=True) 
        template = self.tokenizer.tokenize(template) 

        #tokens = [CLS_TOKEN] + title + [SEP_TOKEN] + abstract + [SEP_TOKEN] + template +  [SEP_TOKEN] 
        #tokens = [CLS_TOKEN] + title + [SEP_TOKEN] + abstract + [SEP_TOKEN] 
        #tokens = [CLS_TOKEN] + title + [SEP_TOKEN] 
        tokens = [CLS_TOKEN] + title + abstract + [SEP_TOKEN] 
        
        #tokens = [CLS_TOKEN] + abstract + [SEP_TOKEN] 
        if len(tokens) >= self.max_text_len:
            cut =  len(tokens) -  self.max_text_len + 1
            abstract = abstract[:-cut] 

        #tokens = [CLS_TOKEN] + title + [SEP_TOKEN] 
        #tokens = [CLS_TOKEN] + title + [SEP_TOKEN] + abstract + [SEP_TOKEN] 
        #tokens = [CLS_TOKEN] + title + [SEP_TOKEN] + abstract + [SEP_TOKEN] + template +  [SEP_TOKEN] 
        tokens = [CLS_TOKEN] + title + abstract + [SEP_TOKEN] 
        #tokens = [CLS_TOKEN] + abstract + [SEP_TOKEN] 

        if len(tokens) > self.max_text_len:
            print(len(tokens))
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        segment_ids = [0] * len(input_ids)
        input_mask = [1] * len(input_ids)
        
        return input_ids, segment_ids, input_mask 

    def __getitem__(self, index):
        title = self.title[index]
        abstract = self.abstract[index]
        label = self.labels[index]
        
        input_ids, segment_ids, input_mask = self.tokenize(title, abstract)
        #label_ids = self.label2id[label]
        label_ids = int(label)
        return input_ids, segment_ids, input_mask, label_ids   


def func_pad(lst, seq_len, PAD=0):
    #out = np.pad(array_index,(0, max(seq_len-len(array_index),0) ),'constant',constant_values=(0,PAD))
    out = lst + [PAD] * (seq_len - len(lst) )
    return out 


class MyCollator(object):
    """ dynamic padding """
    def __init__( self, max_text_len = 128, PAD=0, ):
        self.max_text_len = max_text_len
        self.PAD = PAD

    def __call__(self, batch ):
        ## [input_ids,  segment_ids, input_mask, label_ids ] 
        max_len = min( max( [len(x[0]) for x in batch] ), self.max_text_len )
        input_ids = [ func_pad( data_piece[0], max_len ) for data_piece in batch ]
        segment_ids = [ func_pad( data_piece[1], max_len ) for data_piece in batch ]
        input_mask = [ func_pad( data_piece[2], max_len ) for data_piece in batch ]
        labels = [ data_piece[3] for data_piece in batch ]
        data = { 'inputs': 
                    { 
                    'input_ids': torch.tensor(input_ids,dtype=torch.long),
                    'segment_ids': torch.tensor(segment_ids,dtype=torch.long),
                    'input_mask': torch.tensor(input_mask,dtype=torch.long),
                    },
                }
        if labels[0] >=0 :
            data["labels"] =  torch.tensor( np.array(labels).reshape((-1)) )
        return data