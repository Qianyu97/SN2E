import os
import torch
class Module(torch.nn.Module):
    def __init__(self):
        super(Module, self).__init__()
    
    def loadCheckpoint(self, path, device):
        self.load_state_dict(torch.load(os.path.join(path), device))

    def saveCheckpoint(self, path):
        torch.save(self.state_dict(), path)
    
    def evaluate(self, a, b) -> torch.Tensor: 
        output:torch.Tensor = None  # type: ignore
        return output
    
    def lookupEmbedding(self, index, detach = False):
        return None

    def tailingworks(self):
        pass