import os
from typing import overload
import torch
class Module(torch.nn.Module):
    def __init__(self):
        super(Module, self).__init__()
    
    def loadCheckpoint(self, path):
        self.load_state_dict(torch.load(os.path.join(path)))

    def saveCheckpoint(self, path):
        torch.save(self.state_dict(), path)
    
    def evaluate(self):
        return None
    
    def lookupEmbedding(self):
        return None