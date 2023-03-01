class NodeUnit(object):
    name = ' '
    def __init__(self, name = ' '):
        self.name:str               = name
        self.fathers:set[NodeUnit]  = set() 
        self.sons:set[NodeUnit]     = set()
        self.attributes:set[str]    = set()
    
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
