
import torch
import numpy as np
class Model(torch.nn.Module):
    def __init__(self) -> None:
        super(Model, self).__init__()
        self.a = torch.nn.Embedding(4, 8)
        torch.nn.init.constant_(self.a.weight, 1)
        
        a = 0
        
        
    
    def forward(self):
        self.a.weight.data[3] = 3 * self.a.weight[0]
        b = self.a(torch.LongTensor([3]))
        return b.sum()
    

model = Model()
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.1
    )

optimizer.zero_grad()
loss = model()
loss.backward()
optimizer.step()
a = 0
