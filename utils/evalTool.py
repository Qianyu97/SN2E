import torch

def calcF1score(prediction:torch.Tensor, groundTruth:torch.Tensor, askmore = False):
        tp = (   groundTruth *   prediction).sum()
        fp = (   groundTruth * ~ prediction).sum()
        fn = ( ~ groundTruth *   prediction).sum()
        precision   = (tp + 1) / ( tp + fp + 1)
        recall      = (tp + 1) / ( tp + fn + 1) 
        F1score = (2 * precision * recall / (precision + recall)).max().item()
        if not askmore:        
            return F1score, precision, recall
        else:
            return F1score, precision, recall, tp, fp, fn