import torch
class NodeUnit(object):
    name = ' '
    def __init__(self, name = ' '):
        self.name:str               = name
        self.fathers:set[NodeUnit]  = set()
        self.sons:set[NodeUnit] = set()
        self.attributes:set[NodeUnit]    = set()
        self.depth = 0
    
    def addFather(self, father):
        self.fathers.add(father)
    
    def addSon(self, son):
        self.sons.add(son)
    
    def addAttribute(self, attribute):
        self.attributes.add(attribute)
    
    def updateAttribute(self, attributes):
        self.attributes.update(attributes)
    
    def __str__(self):
        return self.name
    
    def __repr__(self):
        return self.name
    
    def __hash__(self):
        return hash(self.name)
    
    def __eq__(self, __o: object) -> bool:
        if type(__o) == str:
            return __o == self.name
        elif type(__o) == NodeUnit :
            return __o.name == self.name # type: ignore
        else:
            return False

class Embedding():
    def __init__(self, mean, variance_inv) -> None:
        self.m:torch.Tensor = mean
        self.v:torch.Tensor = variance_inv
    
    def detach(self):
        self.m = self.m.detach()
        self.v = self.v.detach()
        return self
    
    def cat(self, t):
        self.m = torch.cat([self.m, t.m], dim=-2)
        self.v = torch.cat([self.v, t.v], dim=-2)
        return self
    
    def inv(self):
        self.v = self.v.reciprocal()
        return self

    def unsqueeze(self, dim):
        self.m = self.m.unsqueeze(dim)
        self.v = self.v.unsqueeze(dim)
        return self
        
