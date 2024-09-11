import torch
from torch import nn

'''def SGD(model,lr,momentum,X,Y,v_list,loss_fn):
    idx=torch.randint(len(X))
    x_train=X[idx]
    y_train=Y[idx]
    loss=loss_fn(y_train,model.parameters())
    loss.backward()
    for name,param in model.named_parameters():
        v_list[name]=momentum*v_list[name]-lr*param.grad
        param.data+=v_list[name]'''
class SGD(nn.Module):
    def __init__(self,model,lr=0.01,momentum=0,weight_decay=0,dampening=0):
        super(SGD,self).__init__()
        self.model=model
        self.lr=lr
        self.momentum=momentum
        self.weight_decay=weight_decay
        self.dampening=dampening
        self.v_list={name:torch.zeros_like(param) for name,param in model.named_parameters()}
    def step(self):
        for name, param in self.model.named_parameters():
            self.v_list[name] = self.momentum * self.v_list[name] - (1-self.dampening)*self.lr * param.grad
            param.data = (1 - self.weight_decay) * param.data + self.v_list[name]
    def zero_grad(self):
        for name,param in self.model.named_parameters():
            param.grad=None


'''        for name,param in self.model.named_parameters():
            self.v_list[name]=self.momentum*self.v_list[name]-self.lr*param.grad
            param.data=(1-self.weight_decay)*param.data+self.v_list[name]'''