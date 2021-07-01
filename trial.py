
import torch
import numpy as np
class Model(torch.nn.Module):
    def __init__(self) -> None:
        super(Model, self).__init__()
        self.b = torch.nn.Parameter(torch.Tensor(np.ones([3,5])))
        self.c = torch.nn.Parameter(torch.Tensor(np.zeros([4,5])), requires_grad = False)
        self.c[0:2] = 3*self.b[0:2]
        self.d = torch.cat((self.b, self.c), 0)
        #self.e[0:2] = self.b
        
        
    
    def forward(self):
        return self.d[2:6].sum()
    
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
