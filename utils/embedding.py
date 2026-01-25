import torch
type Embedding = tuple[torch.Tensor, torch.Tensor]

class EmbeddingOperator():
    def cat(embeddingList:list[Embedding]) -> Embedding:
        meanList, invaList = [], []
        for e in embeddingList:
            mean, invariance = e
            meanList.append(mean)
            invaList.append(invariance)
        newMean = torch.vstack(meanList)
        newVar  = torch.vstack(invaList)
        return newMean, newVar
        