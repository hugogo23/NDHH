
import torch 

class F1Metric():
    ## for multi-classification
    def __init__(self, threshold=0.5):
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.eps = 1e-10
        self.threshold = threshold
    
    def get_metric(self, reset = False ):
        tp, fp, fn = self.tp, self.fp, self.fn
        prec = tp / ( tp + fp + self.eps)
        rec = tp / ( tp + fn + self.eps )
        f1 = 2 * prec * rec / ( prec + rec + self.eps)
        
        if reset:
            self.reset()
        return prec, rec, f1
    
    def reset(self, ):
        self.tp = 0
        self.fp = 0
        self.fn = 0

    def __call__(self, probs, labels ):
        #preds = torch.argmax(probs, dim=1)

        ## for binary cls
        preds = torch.tensor( probs.clone().detach()[:,1] >= self.threshold, dtype=torch.int32)
        assert preds.shape == labels.shape
        tp = torch.sum( (preds ==1) & (labels == 1) )
        
        fp = torch.sum(preds == 1) - tp
        fn = torch.sum(labels == 1) - tp 
        self.tp += tp
        self.fp += fp
        self.fn += fn

        return

class AccMetric():
    ## for multi-classification
    def __init__(self, threshold=0.5):
        self.correct = 0
        self.total = 0
        self.acc = 0
        self.epsion = 1e-10
        self.threshold = threshold
    
    def get_metric(self, reset = False ):
        acc, correct, total = self.acc, self.correct, self.total
        if reset:
            self.reset()
        return acc,  correct, total 
    
    def reset(self, ):
        self.correct = 0
        self.total = 0
        self.acc = 0

    def __call__(self, probs: torch.Tensor, labels:torch.Tensor ):
        ## logits: [batch_size, label_size]
        ## labels: [batch_size, 1]
        
        #preds = torch.argmax(logits, dim=1)
        ## for binary cls

        preds = torch.tensor( probs.clone().detach()[:,1] >= self.threshold, dtype=torch.int32)

        assert preds.shape == labels.shape
        self.correct += torch.sum(preds == labels) 
        self.total += labels.numel()
        self.acc = self.correct / ( self.total + self.epsion )
        return 
