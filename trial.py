import torch
class Model(torch.nn.Module):
    def __init__(self) -> None:
        super(Model, self).__init__()
        self.a = torch.nn.Embedding(num_embeddings=4,embedding_dim=5)
        torch.nn.init.constant_(self.a.weight.data, 1)
    
    def forward(self):
        return self.a(torch.tensor(1)).sum()
    
    def sum(self):
        self.a.weight.data[1] = self.a.weight.data[2] + self.a.weight.data[3]

model = Model()
optimizer = torch.optim.SGD(
            model.parameters(),
            lr=0.1
        )

optimizer.zero_grad()
model.sum()
loss = model()
loss.backward()
optimizer.step()
a = 0
