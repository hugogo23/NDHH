

import imp
import os
from models import BertClassifier
from data_loader import TextDataset, MyCollator
from trainer import Trainer

import torch
from torch.utils.data import DataLoader
#from transformers import BertModel， WarmupLinearSchedule
from transformers import AutoTokenizer
import numpy as np

def get_data_reader(
                    dataset_file_path, 
                    tokenizer, 
                    args, 
                    shuffle=True,
                    trainset=True 
                    ): 
    vocab_path = args.vocab_path
    max_len = args.max_len 
    if trainset: 
        batch_size = args.batch_size
    else:
        batch_size = 1  
    
    dataset = TextDataset( dataset_file_path, tokenizer,  max_text_len = max_len, vocab_file_path=vocab_path,)
    collate = MyCollator( max_text_len=max_len, PAD=0  )
    dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=True, num_workers=4, shuffle=shuffle , collate_fn=collate, drop_last=False )
    return dataloader

def main(args, do_train=True): 
    if not os.path.exists(args.model_dir):
        os.mkdir(args.model_dir)
    
    ## define the model
    model = BertClassifier(args) 
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    
    if args.init_model and os.path.exists( args.init_model):
        model = init_model(model, args.init_model )
        print("init model by args.init_model")
    
    model = model.to(device)

    ## define the optimizer and scheduler
    optimizer = torch.optim.Adam([
        {'params': model.bert.parameters(), 'lr': 1e-5},        # lr': 1e-5}
        {'params': model.classifier.parameters(), 'lr': 1e-5},  # 'lr': 2e-5 , 3e-5
    ])  
    scheduler = None
    """
    # Learning rate scheduler
    scheduler = WarmupLinearSchedule(
        optimizer, 
        warmup_steps=WARMUP_STEPS,
        t_total=len(train_loader) // GRADIENT_ACCUMULATION_STEPS * NUM_EPOCHS
        )
    """

    ## load data
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_dataset = get_data_reader( args.train_file_path, tokenizer, args, shuffle=True, trainset=True )
    dev_dataset = get_data_reader( args.dev_file_path, tokenizer, args, shuffle=False, trainset=False )
    test_dataset = get_data_reader( args.test_file_path, tokenizer, args, shuffle=False,  trainset=False )

    trainer = Trainer( 
                model, 
                device,
                optimizer, 
                scheduler, 
                train_dataset,
                dev_dataset,
                test_dataset,
                model_dir = args.model_dir,
                grad_clipping = 1.0,
                cold_start_epoch = 0, # args.num_epochs, # 10, #0,
                num_epochs = args.num_epochs,
                num_eval_epoch  = 1,
                #num_save_epoch : int = -1,
                batch_size = args.batch_size,
                accumulated_batch_count = args.accumulated_batch_count ,
                ) 
    if do_train:
        trainer.train()
        eval_f1, infer_f1 =  trainer.eval()
        return eval_f1, infer_f1
    else:
        eval_f1, infer_f1 =  trainer.eval(re_init=False) 
        print(eval_f1, infer_f1)
        return eval_f1, infer_f1
    return 

def run_for_eval(args, num_exp=5):
    eval_metrics, infer_metrics = [], []
    for i in range(num_exp):
        eval_f1, infer_f1 = main(args, do_train=True)
        eval_metrics.append( eval_f1 )
        infer_metrics.append( infer_f1 ) 
    #import pdb; pdb.set_trace()

    print("for eval metric: [mean %.3f std %.3f]"% (np.mean(eval_metrics), np.std(eval_metrics) ) )
    print("for infer metric: [mean %.3f std %.3f]"%(np.mean(infer_metrics),  np.std(infer_metrics) ) )


def init_model(model, model_path):
    pt_params = torch.load(model_path, map_location=torch.device('cpu') )
    model_dict = model.state_dict()
    # only load the parameters that in both pt_params and model_params
    state_dict = { k:v for k,v in pt_params.items() if k in model_dict.keys()}
    #import pdb; pdb.set_trace()
    #state_dict = pt_params 
    model_dict.update(state_dict)
    model.load_state_dict(model_dict)
    print(model_path)
    return model 

if __name__ == "__main__":

    model_dir = "default data output folder"
    data_folder = "default data input folder"
 
    train_file_path = data_folder + "train.csv"
    dev_file_path = data_folder + "dev.csv" 
    test_file_path = data_folder + "test.csv"  
    vocab_file_path = None
    
    model_name = "allenai/scibert_scivocab_uncased"  
    model_name = 'bert-base-uncased'
    model_name = 'bert-base-cased'
    #model_name = "google/electra-base-generator"
    model_name = "microsoft/deberta-base"       ## best till now
    ### default and best 
    model_name = 'bert-base-uncased'

    init_bin_fp = "initialize the model in training or inference, example like" 
    init_bin_fp = "folder/health/best_model.bin"

    import argparse 
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', help='Path to the store models', default=model_dir)
    parser.add_argument('--train_file_path', default= train_file_path )
    parser.add_argument('--dev_file_path', default= dev_file_path )
    parser.add_argument('--test_file_path', default= test_file_path )
    parser.add_argument('--vocab_path', default = vocab_file_path )
    parser.add_argument('--model_name', default = model_name )
    parser.add_argument('--num_labels', default = 2 )
    
    ## training details 
    parser.add_argument('--batch_size', default = 4 )   ## . 4
    parser.add_argument('--accumulated_batch_count', default = 4 )  # 4 # 8
    parser.add_argument('--max_len', default = 512 )
    parser.add_argument('--num_epochs', default = 10 )  #30
    parser.add_argument('--init_model', default = init_bin_fp )

    args = parser.parse_args()
    
    #run_for_eval(args,num_exp=5)

    eval_f1, infer_f1 = main(args, do_train=False)
    #print(eval_f1, infer_f1)