
#from itertools import accumulate

import imp
import os
import torch 
import logging
import datetime
import numpy as np
from tqdm import tqdm 

from typing import Dict, Optional, List, Tuple, Union, Iterable, Any 
logger = logging.getLogger(__name__)
import time 
import pandas as pd 

class Trainer():
    def __init__( 
                self,
                model, 
                device,
                optimizer, 
                scheduler, 
                train_dataset,
                valid_dataset,
                test_dataset,
                model_dir,
                grad_clipping = 5.0,
                cold_start_epoch = 0,
                num_epochs  = 10,
                num_eval_epoch = 1,
                num_save_epoch = -1,
                batch_size = 16,
                accumulated_batch_count = 2 ,
                max_iter_per_batch = -1,
                 ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self._train_data = train_dataset 
        self._validation_data = valid_dataset 
        self._test_data = test_dataset
        self._model_dir = model_dir

        self._cold_start_epoch = cold_start_epoch
        self._num_epochs = num_epochs 
        self._eval_epochs = num_eval_epoch
        self._save_epochs = num_save_epoch 
        self._batch_size = batch_size
        self._accumulated_batch_count = accumulated_batch_count
        self._max_iter_per_batch = max_iter_per_batch 
        self._grad_clipping = grad_clipping
        self._model_path = self._model_dir + "/best_model.bin"

    def process_start_setting(self, epoch=-1 ):
        ## -1 means not start yet
        if self._cold_start_epoch > 0 and epoch < self._cold_start_epoch:
            #self.optimizer.set_cold_start()
            ### freeze the main bert model
            for name, param in self.model.named_parameters():
                if "bert" in name:
                    param.requires_grad = False
        elif epoch == self._cold_start_epoch: 
            for name, param in self.model.named_parameters():
               param.requires_grad = True
        else:
            pass
        """
        ### freeze lower bert
        free_layer = 10
        for name, param in self.model.named_parameters():
            if "bert.embeddings" in name:
                param.requires_grad = False
            elif "bert.pooler." in name or "classifier" in name:
                param.requires_grad = True 
            elif name.startswith("bert.encoder.layer"):
                layer_i = int( name.strip().split(".")[3] )
                if layer_i >= free_layer:
                    param.requires_grad = True 
                else:
                    param.requires_grad = False 
        """
        return 

    def _train_epoch(self, epoch: int) -> Dict[str, float]: 
        # Set the model to "train" mode.
        self.model.train()
        self.model.get_metrics(reset=True)
        #self.optimizer.zero_grad()

        train_loss = 0.0
        real_data_size = 0
        torch.cuda.empty_cache()  
        for batches_this_epoch, batch_group in enumerate(self._train_data ):            
            self.optimizer.zero_grad()
            loss, logits, _ = self.batch_loss(batch_group ) 
            if self._accumulated_batch_count > 0: 
                loss /= self._accumulated_batch_count
            batch_size = logits.size()[0] 
            real_data_size += batch_size 
            loss.backward()
            train_loss += loss.detach().item() * batch_size
            if (batches_this_epoch + 1) % self._accumulated_batch_count == 0: 
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self._grad_clipping )  
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()
        train_loss /= real_data_size
        metric = self.model.get_metrics(reset=False)
        #logger.info("Epoch %d: Loss %.3f accurancy %.3f" %( epoch, train_loss, metric["acc"]  ) ) 
        if epoch == 0:
            print("num_train_step each epoch ", batches_this_epoch)
        return train_loss, metric  

    def batch_loss(self, batch_input ):
        inputs = {k:v.to(self.device) for k,v in batch_input['inputs'].items() }
        if "labels" in batch_input: 
            labels = batch_input['labels'].to(self.device)
        else:
            labels = None
        loss, logits, probas = self.model( inputs, labels )
        return loss, logits, probas 

    def train(self) -> Dict[str, Any]:
       
        """
        train_metrics: Dict[str, float] = {}
        val_metrics: Dict[str, float] = {}
        this_epoch_val_metric: float = None
        metrics: Dict[str, Any] = {}
        epochs_trained = 0
        metrics["best_epoch"] = self._metric_tracker.best_epoch
        for key, value in self._metric_tracker.best_epoch_metrics.items():
            metrics["best_validation_" + key] = value
        """

        metrics_best_epoch = {"epoch": -1,  "loss": 1e10 }
        training_start_time = time.time()
        if os.path.exists(self._model_path):
            os.remove( self._model_path )

        for epoch in tqdm( range( self._num_epochs ) ):
            self.process_start_setting(epoch)
            epoch_start_time = time.time()
            train_loss, train_metric = self._train_epoch(epoch)
            #print("Epoch %d: Loss %.3f accurancy %.3f" %( epoch, train_loss, train_metric["acc"]  ) )

            torch.cuda.empty_cache()
            if self._validation_data is not None and \
                    ( (epoch+1) % self._eval_epochs == 0 or epoch == self._num_epochs - 1 ) :
                with torch.no_grad():
                    # We have a validation set, so compute all the metrics on it.
                    val_loss, eval_metric = self._validation_loss()
                #print("\tEval: Loss %.3f accurancy %.3f" %( val_loss, eval_metric['acc'] ) )
                if val_loss < metrics_best_epoch["loss"]:
                    metrics_best_epoch['loss'] = val_loss 
                    metrics_best_epoch['epoch'] = epoch 
                    for k in eval_metric: 
                        metrics_best_epoch[k] = eval_metric[k]
                    self._save_checkpoint()
        
        print("\n======best epoch: ")
        print(metrics_best_epoch)
        return 

    def eval(self, re_init=True ):
        if re_init:
            self._restore_checkpoint()
        val_loss, eval_metric =  self._validation_loss( in_test=False ) 
        print("\tEval: Loss %.3f accurancy %.3f=%d/%d || [p %.2f][r: %.2f][f1 %.2f]" %( val_loss, eval_metric['acc'] , eval_metric[ "correct"], eval_metric["total"], \
            eval_metric['prec'], eval_metric['rec'], eval_metric['f1'] ) )
        #print("eval metric: ", eval_metric )
        #import pdb; pdb.set_trace()
        target_eval_metric = eval_metric['f1'] #.numpy()

        infer_loss, infer_metric = self._validation_loss( in_test=True )
        print("\tInfer: Loss %.3f accurancy %.3f=%d/%d || [p %.2f][r: %.2f][f1 %.2f]" %( infer_loss, infer_metric['acc'],  infer_metric[ "correct"], infer_metric["total"], \
            infer_metric['prec'], infer_metric['rec'], infer_metric['f1'] ) )   
        #print("infer metric: ",infer_metric)
        target_infer_metric = infer_metric['f1'] #.numpy()
        print()
        try:
            return target_eval_metric.cpu().numpy(), target_infer_metric.cpu().numpy()
        except:
            return target_eval_metric, target_infer_metric 

    def _validation_loss(self, in_test = False ):
        self.model.eval()
        self.model.get_metrics(reset=True)  
        val_loss = 0
        batches_this_epoch = 0   #+ 1e-10
        real_data_size = 0
        target_data = self._validation_data if not in_test else self._test_data 
        #target_data = self._train_data 
        for batch_group in target_data: 
            loss, logits, probas = self.batch_loss(batch_group)
            batch_size = logits.size()[0] 
            real_data_size += batch_size 
            if loss is not None:
                val_loss += loss.detach().item() * batch_size
        val_loss /=  real_data_size
        metric = self.model.get_metrics(reset=False)  
        #logger.info("Epoch %d: Loss %.3f accurancy %.3f" %( epoch, val_loss, acc ) )
        #import pdb; pdb.set_trace()
        return val_loss, metric
    
    def infer(self, in_fp, out_fp, key = "health_proba", model_path=None, ):
        self._restore_checkpoint(model_path)
        self.model.eval()
        
        all_probas = []
        #data = pd.read_csv( in_fp, sep="\t")
        data = pd.read_csv( in_fp,  encoding = "ISO-8859-1") #sep="\t")
        
        for batch_group in tqdm(self._test_data ):            
            loss, logits, probas = self.batch_loss(batch_group ) 
            probas = probas.cpu().detach().numpy()[:,1].tolist()
            all_probas.extend(probas)  
        data[key] = all_probas         
        data.to_csv( out_fp,  sep="\t", index=False  )
        print(data[key].describe())
        print("pos predict count: ", len(data[data[key] >= 0.5]) )
        #import pdb; pdb.set_trace()
        return 

    def _save_checkpoint(self,  ): 
        path = self._model_dir + "/best_model.bin"
        if os.path.exists(path):
            os.remove(path)
        torch.save( self.model.state_dict(),path)
        return 
    
    def _restore_checkpoint(self, path=None ):
        if path is None:
            path = self._model_path
            #self._model_dir + "/best_model.bin" 
        #path="/data00/home/labspeech/linyu/datasets/health/params_used/health//best_model.bin"
        if not os.path.exists(path):
            assert False, "ckpt not exists"
        self.model.load_state_dict(torch.load(path, map_location=torch.device('cpu') ) ) 
        self.model.to( self.device )
        return 
