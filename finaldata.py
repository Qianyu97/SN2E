from refine import FineData, RawData, BaseData, pd
from config import DatapathArg

class FinalData(BaseData):
    def __init__(self, ifloadDictionary = False):
        finedata = FineData(DatapathArg.path_rawdata)
        if not ifloadDictionary:
            self.dictionary = self.creat_index(finedata.fulllist) 
        else: 
            self.dictionary:dict = self.load(DatapathArg.path_indexdict)   # type: ignore
        indexdata:FineData = self.indexconvert(finedata) 
        indexdata.negtDF.replace(0, 1, inplace=True)       
        self.finedata   = finedata
        self.indexdata  = indexdata
    
    def creat_index(self, fulllist):
        int2strList = [None] + ['negtpad'] + fulllist + ['godfather']
        str2intdict = {item: idx for idx, item in enumerate(int2strList)}
        return {'str' : str2intdict, 'int': int2strList}
    
    def indexconvert(self, data, arrow = 'str2int'):
        def translate(source):
            sourcetype = type(source)
            if sourcetype == FineData:
                newsource = FineData()
                for a in dir(source):
                    ga = source.__getattribute__(a)
                    if not callable(ga) and not a.startswith('__'):
                        newsource.__setattr__(a, translate(ga))
                return newsource
            elif sourcetype == pd.DataFrame:
                newDataFrame:pd.DataFrame = source.applymap(lambda x: dictionary[x])
                newDataFrame.index = source.index.map(lambda x: dictionary[x])
                return newDataFrame
            elif sourcetype == dict:
                return  {dictionary[key] : translate(value) for key, value in source.items()}
            elif sourcetype == list:
                return [dictionary[i] for i in source]
            elif sourcetype == set:
                return {dictionary[i] for i in source}
            elif (sourcetype == str and arrow == 'str2int') or (sourcetype == int and arrow == 'int2str'):
                return dictionary[source]
            else:
                return source
        assert arrow == 'str2int' or arrow == 'int2str'
        dictionary = self.dictionary['str'] if arrow == 'str2int' else self.dictionary['int'] 
        output:data = translate(data)
        return output

if __name__ == '__main__':
    finaldata = FinalData()
    a = 0