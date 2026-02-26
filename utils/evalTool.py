import torch

def calcF1score(prediction:torch.Tensor, groundTruth:torch.Tensor, askmore = False):
        groundTruth = groundTruth.to(prediction.device)
        tp = (   groundTruth *   prediction).sum(-1)
        fp = (   groundTruth * ~ prediction).sum(-1)
        fn = ( ~ groundTruth *   prediction).sum(-1)
        precision   = (tp + 1) / ( tp + fp + 1)
        recall      = (tp + 1) / ( tp + fn + 1) 
        F1score = (2 * precision * recall / (precision + recall))
        if not askmore:        
            return F1score, precision, recall
        else:
            return F1score, precision, recall, tp, fp, fn