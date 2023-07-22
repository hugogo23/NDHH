
import imp
import torch
import torch.nn as nn
from transformers import BertModel, AutoModel, AutoModelForMaskedLM, AutoTokenizer

from utils import AccMetric, F1Metric

class BertClassifier(nn.Module):
    def __init__(self, 
                config,
                dropout = 0.5
                ):
        super(BertClassifier, self).__init__()
        # Binary classification problem (num_labels = 2)
        self.num_labels =  config.num_labels
        # Pre-trained BERT model
        self.bert = AutoModel.from_pretrained(config.model_name, output_hidden_states=True)
        # Dropout to avoid overfitting 
        #self.bert = AutoModelForMaskedLM.from_pretrained(config.model_name )     

        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout( dropout ) 
        hidden_size = self.bert.config.hidden_size 
        self.classifier = nn.Linear(hidden_size, self.num_labels )
        torch.nn.init.xavier_normal_(self.classifier.weight)
        self.loss_layer = nn.CrossEntropyLoss(reduction="mean")

        threshold=0.5 #0.7
        self.acc_metric = AccMetric(threshold=threshold) 
        self.f1_metric = F1Metric(threshold=threshold) 

    def forward(self, inputs, labels = None):  
        output = self.bert( 
            inputs.get('input_ids'), 
            token_type_ids = inputs.get('segment_ids'), 
            attention_mask = inputs.get('input_mask'),
            )
        """
        words = torch.argmax( output['logits'], dim=2)
        model_name="allenai/scibert_scivocab_uncased"
        tokenizer = AutoTokenizer.from_pretrained( model_name )
        texts = tokenizer.batch_decode( words)
        import pdb; pdb.set_trace()
        """
        sequence_embs = output['last_hidden_state']
        
        model_type = "mean" 
        if model_type == "cls":
            """
            if 'pooler_output' in output: 
                pooled_output = output['pooler_output']
            else:
                pooled_output = sequence_embs[:, 0, :] 
            """
            pooled_output = sequence_embs[:, 0, :]  
            output = pooled_output 
        elif model_type == "mean":  
            input_mask = torch.unsqueeze(inputs.get('input_mask') , 2)
            #input_mask = input_mask.expand(-1, -1, sequence_embs.size()[2])
            sent_emb = torch.sum( sequence_embs * input_mask, axis=1)
            mask = torch.sum(inputs.get('input_mask'), axis=1, keepdim=True) 
            output = sent_emb / mask

        elif model_type == "template":
            mask_id = 104   ## cls=102
            input_mask_pos = ( inputs.get('input_ids') == mask_id )
            input_mask_pos = torch.unsqueeze( input_mask_pos , 2)
            input_mask_emb = torch.sum( sequence_embs* input_mask_pos, axis=1) 
            output = input_mask_emb 
        
        output = self.dropout(output)
        logits = self.classifier(output)
        probs = self.softmax( logits )
        if labels is not None:
            loss = self.loss_layer(input=logits, target=labels)
            self.acc_metric( probs, labels)
            self.f1_metric( probs, labels )
        else:
            loss = None
        return loss, logits, probs
    
    def get_metrics(self, reset = False ):
        acc, correct, total = self.acc_metric.get_metric(reset)
        prec, rec, f1 = self.f1_metric.get_metric(reset)
        return {"acc":acc, "correct": correct, "total": total,
                "prec": prec, "rec": rec, "f1": f1 }