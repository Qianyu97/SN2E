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
    
    def train_one_batch(self, batchdata):
        self.optimizer.zero_grad()
        loss = self.model(batchdata)
        loss.backward()
        self.optimizer.step()
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
            for batchdata in self.dataLoader:
                loss = self.train_one_batch(batchdata)
            
            '''
            EPOCHS_ITER.set_description("Epoch %d | postive loss : %f, negtive loss : %f, min positive loss: %f, min negtive loss %f" \
                        % (epoch, posSumLoss/count_i, negSumLoss/count_j/self.configs.model.Alpha, posMinLoss, negMinLoss))
            
            if epoch % self.configs.evalepoch == 0:
                HR = self.evaluater.HREvaluate(self.model)
                #CR = self.evaluater.CREvaluate(self.model)
                EPOCHS_ITER.set_description("Epoch %d | Loss : %d, HR: %d, CR: %d " \
                        % (epoch, minLoss, HR))#, CR))
                if bestHR < HR:
                    bestHR = HR
                    self.model.saveCheckpoint(self.configs.modelPath)
            '''
        #self.model.setDefiConceptEmbedding(self.dataLoader.dataset.numData.homoDF)
        #self.model.saveCheckpoint(self.configs.modelPath)
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


                

        
