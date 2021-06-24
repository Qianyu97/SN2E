class attrDict(object):
    def __init__(self, attrstring):
        super(attrDict, self).__init__()
        attrstring_list = attrstring.split()
        if len(attrstring_list) == 1:
            attrstring_list = ['IsA'] + attrstring_list 
            attrstring = 'IsA ' + attrstring
        [self.relation, self.entity] = attrstring_list
        self.attrstring = attrstring
        self.isaFlag = (self.relation == 'IsA')

    def __hash__(self):
        if self.isaFlag:
            return hash(self.entity)
        else:
            return hash(self.attrstring)
    
    def __str__(self):
        return self.attrstring

    def __repr__(self):
        return self.attrstring
    
    def __eq__(self, value):
        if type(value) is attrDict:    
            return self.attrstring == value.attrstring
        elif type(value) is str:
            if self.isaFlag :
                return (self.entity == value) or (self.attrstring == value)
            else:
                return self.attrstring == value
        else:
            return False