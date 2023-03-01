
from refine import *
from config import DatapathArg
class FinalData():
    def __init__(self, path_rawdata):
        finedata = FineData(path_rawdata)
        dictionary = self.creat_index(finedata.fulllist)
        indexdata = self.indexconvert(finedata, dictionary) 
        self.finedata   = finedata
        self.indexdata  = indexdata
        
    def creat_index(self, fulllist):
        num2strList = fulllist
        str2numdict = {item: idx for idx, item in enumerate(num2strList)}
        return {'str' : str2numdict, 'num': num2strList}
    
    def indexconvert(self, data, dictionary, arrow = 'str2num'):
        def translate(input):
            inputype = type(input)
            if inputype == FineData:
                newversion = FineData()
                for a in dir(input):
                    ga = input.__getattribute__(a)
                    if not callable(ga) and not a.startswith('__'):
                        newversion.__setattr__(a, translate(ga))
                return newversion
            elif inputype == dict:
                return  {translate(key) : translate(value) for key, value in input.items()}
            elif inputype == list:
                return [translate(i) for i in input]
            elif inputype == set:
                return {translate(i) for i in input}
            elif inputype == str:
                return dictionary[input]
            else:
                raise Exception('input type not accessable')
        assert arrow == 'str2num' or arrow == 'num2str'
        dictionary = dictionary['str'] if arrow == 'str2num' else dictionary['num']
        output:data = translate(data)
        return output

if __name__ == '__main__':
    finaldata = FinalData(DatapathArg.path_rawdata)
    a = 0
    
            
    
        
        