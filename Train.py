#from tensorboardX import SummaryWriter
from line_profiler import LineProfiler
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import DatapathArg, DataloaderArg, TrainArg, ModelArg, validateArg
from finaldata import FinalData, RawData
from mycode.utils import prepare
from mycode.utils.evaluate import Evaluater
from mycode.datasets.attrDataset import attrDataset
from mycode.datasets.tripleDataset import tripleDataset
from mycode.models.SN2E import SN2E

class Trainer():
    def __init__(self, dataLoader:DataLoader, model:SN2E, evaluater:Evaluater) -> None:
        self.dataLoader = dataLoader
        self.evaluater  = evaluater
        self.model      = model
        self.optimizer, self.scheduler  = prepare.prepareOptimizer(self.model)
    
    def train_one_batch(self, batchdata):
        loss, posloss, negloss = self.model(batchdata)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.model.tailingWorks()
        return loss.item(), posloss.item(), negloss.item()
    
    def run(self):
        print('Info : start model training')
        EPOCHS = TrainArg.epochs
        EPOCHS_ITER = tqdm(range(EPOCHS))
        posMinLoss, negMinLoss = float("inf"), float("inf") 
        bestHR = float("inf")
        for epoch in EPOCHS_ITER:
            sumloss, posSumLoss, negSumLoss = 0, 0, 0
            for batchdata in self.dataLoader:
                loss, posloss, negloss = self.train_one_batch(batchdata)
                sumloss     += loss
                posSumLoss  += posloss
                negSumLoss  += negloss
            EPOCHS_ITER.set_description("Epoch %d | loss : %.2f, postive loss : %.2f, negtive loss : %.2f" \
                        % (epoch, sumloss, posSumLoss, negSumLoss))
            self.scheduler.step()
        print('Info : finish model training')
        self.model.saveCheckpoint(ModelArg.path_model)
        print('Info : save model sucessfully')
        a = 0

def main():
    dataloader = DataLoader(
            dataset     = dataset,                
            batch_size  = DataloaderArg.batchsize,
            shuffle     = DataloaderArg.shuffle,
            num_workers = DataloaderArg.numworkers,
            drop_last   = DataloaderArg.droplast,
            )
    model      = prepare.prepareModel(finaldata.indexdata.homoDF)
    evaluater = Evaluater(finaldata, model)
    trainer = Trainer(dataloader, model, evaluater)
    trainer.run()
    finaldata.save(DatapathArg.path_indexdict, 'dictionary')

if __name__ == "__main__":
    VALIDATE = True
    finaldata = FinalData(DatapathArg.path_rawdata)
    dataset = attrDataset(finaldata.indexdata) 
    if VALIDATE:
        for i, paramter in enumerate(validateArg.candidate):
            print('the %dth validation begin. the '%(i) \
                + validateArg.name + ' is ' + str(paramter) \
                + '  -----------------------------------------------------------')
            setattr(validateArg.field, validateArg.name, paramter) # type: ignore
            main()
    else:
        if TrainArg.measuretime:
            lprofiler = LineProfiler(Trainer.run)
            lprofiler.run('main')
            lprofiler.print_stats()
            lprofiler.dump_stats(DatapathArg.path_profiler)
        else:
            main()
    
        
    
    


                

        
