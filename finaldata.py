from refine import *
class FinalData(BaseData):
    def __init__(self, path_rawdata):
        finedata = FineData(path_rawdata)
        dictionary = self.creat_index(finedata.fulllist)
        indexdata:FineData = self.indexconvert(finedata, dictionary) 
        self.finedata   = finedata
        self.indexdata  = indexdata
    
    
    def creat_index(self, fulllist):
        num2strList = [None] + fulllist
        str2numdict = {item: idx for idx, item in enumerate(num2strList)}
        return {'str' : str2numdict, 'num': num2strList}
    
    def indexconvert(self, data, dictionary, arrow = 'str2num'):
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
                newDataFrame = source.applymap(lambda x: dictionary[x])
                newDataFrame.columns = source.columns.map(lambda x: dictionary[x])
                return newDataFrame
            elif sourcetype == dict:
                return  {translate(key) : translate(value) for key, value in source.items()}
            elif sourcetype == list:
                return [translate(i) for i in source]
            elif sourcetype == set:
                return {translate(i) for i in source}
            elif sourcetype == str or sourcetype == int:
                return dictionary[source]
            else:
                return source
        assert arrow == 'str2num' or arrow == 'num2str'
        dictionary = dictionary['str'] if arrow == 'str2num' else dictionary['num']
        output:data = translate(data)
        return output

if __name__ == '__main__':
    finaldata = FinalData(DatapathArg.path_rawdata)
    a = 0
    
            
    
        
        