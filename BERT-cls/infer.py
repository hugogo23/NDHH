

import imp
import os
#from typing_extensions import dataclass_transform
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

def infer(args, out_fp, key): 
    if not os.path.exists(args.model_dir):
        os.mkdir(args.model_dir)
    
    ## define the model
    model = BertClassifier(args) 
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    if args.init_model:
        model = init_model(model, args.init_model )
        print("init model by args.init_model")
    model = model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    test_dataset = get_data_reader( args.test_file_path, tokenizer, args, shuffle=False,  trainset=False )
    
    optimizer = scheduler = None
    train_dataset = dev_dataset = [] 
    trainer = Trainer( 
                model, 
                device,
                optimizer, 
                scheduler, 
                train_dataset,
                dev_dataset,

                test_dataset,
                model_dir = args.model_dir,
                ) 

     #
    
    trainer.infer(args.test_file_path, out_fp, key ) 
    return 

def run_for_eval(args, num_exp=5):
    eval_metrics, infer_metrics = [], []
    for i in range(num_exp):
        eval_f1, infer_f1 = main(args)
        eval_metrics.append( eval_f1 )
        infer_metrics.append( infer_f1 ) 
    print("for eval metric: [mean %.3f std %.3f]"% (np.mean(eval_metrics), np.std(eval_metrics) ) )
    print("for infer metric: [mean %.3f std %.3f]"%(np.mean(infer_metrics),  np.std(infer_metrics) ) )


def init_model(model, model_path):
    pt_params = torch.load(model_path, map_location=torch.device('cpu') )
    model_dict = model.state_dict()
    # only load the parameters that in both pt_params and model_params
    #state_dict = { k:v for k,v in pt_params.items() if k in model_dict.keys()}
    state_dict = pt_params 
    model_dict.update(state_dict)
    model.load_state_dict(model_dict)
    return model 

def main(model_dir, test_file_path, out_fp, key ):
    import argparse 
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', help='Path to the store models', default=model_dir)
    parser.add_argument('--train_file_path', default= "" )
    parser.add_argument('--dev_file_path', default= "" )
    parser.add_argument('--test_file_path', default= test_file_path )
    parser.add_argument('--vocab_path', default = None )
    parser.add_argument('--model_name', default = 'bert-base-uncased' )
    parser.add_argument('--num_labels', default = 2 )
    ## training details 
    parser.add_argument('--batch_size', default = 8 )
    parser.add_argument('--accumulated_batch_count', default = 8 ) 
    parser.add_argument('--max_len', default = 512 )
    parser.add_argument('--num_epochs', default = 30 )
    parser.add_argument('--init_model', default = "" )
    args = parser.parse_args()
    infer(args, out_fp, key)

def run_infer(data_folder, test_in_file ):
    ### model params
    disaster_folder="/data00/home/labspeech/linyu/datasets/health/params_used/disaster/"
    heath_folder="/data00/home/labspeech/linyu/datasets/health/params_used//health/"
    """
    out_fp = data_folder + "doc_disaster_out.csv" 
    key = "disaster_proba"
    main(disaster_folder, test_in_file, out_fp, key )    
    """
    out_fp = data_folder + "doc_health_out.csv"   
    key = "health_proba" 
    main(heath_folder, test_in_file, out_fp, key )     

if __name__ == "__main__":

    """
    #best_model.bin"
    data_folder="/data00/home/labspeech/linyu/datasets/health/unsup_cls/infer_data/"
    test_file_path = data_folder + "doc_1w_input.csv"  
    ### default and best 

    model_dir="/data00/home/labspeech/linyu/datasets/health/params_used/disaster/"
    out_fp = data_folder + "doc_1w_output.csv"  
    key = "disaster_proba"

    model_dir="/data00/home/labspeech/linyu/datasets/health/params_used//health/"
    out_fp = data_folder + "doc_1w_health.csv" 
    key = "health_proba" 
    main(model_dir, test_file_path, out_fp, key )    
    """
 
    ## infer on 10w data : data from 3 datasets
    data_folder="/data00/home/labspeech/linyu/datasets/health/unsup_cls/infer_data/doc_13w_0405/"
    test_file_path = data_folder + "/doc_13w.csv"
    #run_infer(data_folder, test_file_path )

    ## infer on pubmed 4.3w data 
    data_folder="/data00/home/labspeech/linyu/datasets/health/unsup_cls/infer_data/doc_pub_med/"
    test_file_path = data_folder + "/doc_pubmec.csv" 
    #run_infer(data_folder, test_file_path )

    data_folder="/data00/home/labspeech/linyu/datasets/health.2/pre4label/" 
    test_file_path = data_folder + "/all_data_100.csv" 
    run_infer(data_folder, test_file_path )


    """
    data_folder="/data00/home/labspeech/linyu/datasets/health/unsup_cls/infer_data/doc_14w_all/"
    test_file_path = data_folder + "/doc_pt_all"
    run_infer(data_folder, test_file_path ) 
    """