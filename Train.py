from tensorboardX import SummaryWriter
from line_profiler import LineProfiler
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config
from mycode.utils import prepare
from mycode.utils.evaluater import Evaluater



class Trainer():
    def __init__(self, configs:Config, dataLoader:DataLoader, evaluater:Evaluater, ifLoadModel = False) -> None:
        self.dataLoader = dataLoader
        self.evaluater  = evaluater
        self.configs    = configs
        self.model      = prepare.prepareModel(configs, ifLoadModel)
        self.optimizer  = prepare.prepareOptimizer(configs, self.model)
    
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
        EPOCHS = self.configs.model.epochs
        EPOCHS_ITER = tqdm(range(EPOCHS))
        posMinLoss, negMinLoss = float("inf"), float("inf") 
        bestHR = float("inf")
        for epoch in EPOCHS_ITER:
            posSumLoss, negSumLoss = 0, 0
            posCheckLoss, negCheckLoss = 0, 0
            i, j = 0, 0
            #self.model.tempProcess(self.evaluater.dataset.numData.homoDF)
            for primBatchData in self.dataLoader:
                negLoss = self.train_one_batch_neg(primBatchData, 'negMode')
                negCheckLoss += negLoss
                negSumLoss += negLoss
                j += 1
            if negMinLoss > negSumLoss / j /self.configs.model.Alpha:
                negMinLoss = negSumLoss / j /self.configs.model.Alpha
            
            self.dataLoader.dataset.transformLoader()
            #self.model.tempProcess(self.evaluater.dataset.numData.homoDF)
            for defiBatchData in self.dataLoader:
                posLoss = self.train_one_batch_pos(defiBatchData, 'posMode')
                posCheckLoss += posLoss
                posSumLoss += posLoss
                i += 1 
            if posMinLoss > posSumLoss / i:
                posMinLoss = posSumLoss / i
            
            self.dataLoader.dataset.transformLoader()

            EPOCHS_ITER.set_description("Epoch %d | postive loss : %f, negtive loss : %f, min positive loss: %f, min negtive loss %f" \
                        % (epoch, posSumLoss/i, negSumLoss/j/self.configs.model.Alpha, posMinLoss, negMinLoss))
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
    dataset = attr
    mdataloader = DataLoader(dataset,
                             batch_size=configs.batchsize,
                             shuffle=configs.shuffle,
                             num_workers=configs.numworkers,
                             drop_last=configs.droplast)
    mevaluater = Evaluater(configs, dataset)
    mtrainer = Trainer(configs, mdataloader, mevaluater)
    #lprofiler = LineProfiler(Trainer.run)
    #lprofiler.run('mtrainer.run()')
    #lprofiler.print_stats()
    mtrainer.run()


                

        
