
import torch
import numpy as np
class Model(torch.nn.Module):
    def __init__(self) -> None:
        super(Model, self).__init__()
        self.b = torch.nn.parameter.Parameter(torch.Tensor(np.ones([3,5])))
        #self.c = torch.Tensor(np.zeros([4,5]))
        self.c = 3*self.b
        self.d = torch.cat((self.b, self.c), 0)
        #self.e[0:2] = self.b
        
        
    
    def forward(self):
        return self.c.sum()
    
    def sum(self):
        self.d.data[3:6] = 3*self.d[0:3]

model = Model()
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.1
    )

optimizer.zero_grad()
#model.sum()
loss = model()
loss.backward()
optimizer.step()
a = 0
