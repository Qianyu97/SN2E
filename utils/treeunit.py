import torch
class Unit(object):
    def __init__(self, name:str = ''):
        self.essence = name
        self.name = name if name is not None else 'None'
        self.index = 0
    
    def setIndex(self, index:int):
        self.index = index
    
    def __str__(self):
        return self.name
    
    def __repr__(self):
        return self.name
    
    def __hash__(self):
        return hash(self.essence)
    
    def __eq__(self, __o: object) -> bool:
        if __o is None:
            return __o == self.essence
        elif type(__o) == Unit:
            return __o.essence == self.essence # type: ignore
        else:
            return __o == self.essence

class NodeUnit(Unit):
    def __init__(self, 
                 name:str = '', 
                 father:'NodeUnit' = None, 
                 attributes:list['AttributeUnit'] = None, 
                 children:list['NodeUnit'] = None,
                 depth:int = None):
        super().__init__(name)
        self.father = father
        self.attributes:set[AttributeUnit] = set(attributes) if attributes is not None else set()
        self.children:set[NodeUnit] = set(children) if children is not None else set()
        self.depth = depth
        self.index = 0
    
    def setFather(self, father:'NodeUnit'):
        self.father = father
    
    def setChildren(self, children:list['NodeUnit']):
        self.children = set(children)
    
    def setAttributes(self, attributes:list['AttributeUnit']):
        self.attributes = set(attributes)
    
    def setDepth(self, depth:int):
        self.depth = depth
    
    def addChild(self, child:'NodeUnit'):
        self.children.add(child)

    def addAttribute(self, attribute:'AttributeUnit'):
        self.attributes.add(attribute)

class AttributeUnit(Unit):
    def __init__(self, name = '', children:list['NodeUnit'] = None):
        super().__init__(name)
        self.children = set(children) if children is not None else set()
        self.index = 0
    
    def addChild(self, child:NodeUnit):
        self.children.add(child)
    
    def setIndex(self, index:int):
        self.index = index
    


    
        
