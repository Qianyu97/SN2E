from tensorboardX import SummaryWriter
from line_profiler import LineProfiler
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import DatapathArg, DataloaderArg, TrainArg, ModelArg
from finaldata import FinalData
from mycode.utils import prepare
from mycode.utils.evaluater import Evaluater
from mycode.datasets.attrDataset import attrDataset
from mycode.datasets.tripleDataset import tripleDataset



class Trainer():
    def __init__(self, dataLoader:DataLoader, evaluater:Evaluater) -> None:
        self.dataLoader = dataLoader
        self.evaluater  = evaluater
        self.model      = prepare.prepareModel()
        self.optimizer  = prepare.prepareOptimizer(self.model)
    
    def train_one_batch(self, batchData, trainMode):
        self.optimizer.zero_grad()
        self.model.tempProcess(self.evaluater.dataset.numData.homoDF)
        loss = self.model(batchData, trainMode)
        loss.backward()
        self.optimizer.step()
        self.model.tailingWorks()
        return loss.item()

    def train_one_batch_neg(self, batchData, trainMode):
        self.optimizer.zero_grad()
        self.model.tempProcess(self.evaluater.dataset.numData.homoDF)
        loss = self.model(batchData, trainMode)
        loss.backward()
        self.optimizer.step()
        self.model.tailingWorks()       
        return loss.item()
    
    def train_one_batch_pos(self, batchData, trainMode):
        self.optimizer.zero_grad()
        self.model.tempProcess(self.evaluater.dataset.numData.homoDF)
        loss = self.model(batchData, trainMode)
        loss.backward()
        self.optimizer.step()
        self.model.tailingWorks()
        return loss.item() 
    
    def run(self):
        EPOCHS = TrainArg.epochs
        EPOCHS_ITER = tqdm(range(EPOCHS))
        posMinLoss, negMinLoss = float("inf"), float("inf") 
        bestHR = float("inf")
        for epoch in EPOCHS_ITER:
            posSumLoss, negSumLoss = 0, 0
            posCheckLoss, negCheckLoss = 0, 0
            count_i, count_j = 0, 0
            #self.model.tempProcess(self.evaluater.dataset.numData.homoDF)
            for primBatchData in self.dataLoader:
                negLoss = self.train_one_batch_neg(primBatchData, 'negMode')
                negCheckLoss += negLoss
                negSumLoss += negLoss
                count_j += 1
            negMinLoss = min(negMinLoss, negSumLoss / count_j)
            
            self.dataLoader.dataset.transformLoader()
            #self.model.tempProcess(self.evaluater.dataset.numData.homoDF)
            for defiBatchData in self.dataLoader:
                posLoss = self.train_one_batch_pos(defiBatchData, 'posMode')
                posCheckLoss += posLoss
                posSumLoss += posLoss
                count_i += 1 
            if posMinLoss > posSumLoss / count_i:
                posMinLoss = posSumLoss / count_i
            
            self.dataLoader.dataset.transformLoader()

            EPOCHS_ITER.set_description("Epoch %d | postive loss : %f, negtive loss : %f, min positive loss: %f, min negtive loss %f" \
                        % (epoch, posSumLoss/count_i, negSumLoss/count_j/self.configs.model.Alpha, posMinLoss, negMinLoss))
            '''
            if epoch % self.configs.evalepoch == 0:
                HR = self.evaluater.HREvaluate(self.model)
                #CR = self.evaluater.CREvaluate(self.model)
                EPOCHS_ITER.set_description("Epoch %d | Loss : %d, HR: %d, CR: %d " \
                        % (epoch, minLoss, HR))#, CR))
                if bestHR < HR:
                    bestHR = HR
                    self.model.saveCheckpoint(self.configs.modelPath)
            '''
        self.model.setDefiConceptEmbedding(self.dataLoader.dataset.numData.homoDF)
        self.model.saveCheckpoint(self.configs.modelPath)
        a = 0

if __name__ == "__main__":
    finedata = FinalData(DatapathArg.path_rawdata)
    dataset = attrDataset(finedata.indexdata) 
    mdataloader = DataLoader(dataset    = dataset,
                             batch_size =DataloaderArg.batchsize,
                             shuffle    =DataloaderArg.shuffle,
                             num_workers=DataloaderArg.numworkers,
                             drop_last  =DataloaderArg.droplast)
    mevaluater = Evaluater(finedata)
    mtrainer = Trainer(mdataloader, mevaluater)
    #lprofiler = LineProfiler(Trainer.run)
    #lprofiler.run('mtrainer.run()')
    #lprofiler.print_stats()
    mtrainer.run()


                

        
