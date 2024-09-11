import torch
from torch import nn


class SGD(nn.Module):
    def __init__(self,model,lr=0.01,momentum=0,weight_decay=0,dampening=0,nesterov=False,maximaze=False):
        super(SGD,self).__init__()
        self.model=model
        self.lr=lr
        self.momentum=momentum
        self.weight_decay=weight_decay
        self.dampening=dampening
        self.nesterov=nesterov
        self.maximaze=maximaze
        self.t=1
        self.b={name:torch.zeros_like(param) for name,param in model.named_parameters()}
    def step(self):
        for name,param in self.model.named_parameters():
            if self.weight_decay!=0:
                g=param.grad+self.weight_decay*param.data
            else:
                g=param.grad
            if self.momentum!=0:
                if self.t>1:
                    self.b[name]=self.momentum*self.b[name]+(1-self.dampening)*g
                else:
                    self.b[name]=g
                if self.nesterov:
                    g=g+self.momentum*self.b[name]
                else:
                    g=self.b[name]
            if self.maximaze:
                param.data+=self.lr*g
            else:
                param.data-=self.lr*g
    def zero_grad(self):
        for name,param in self.model.named_parameters():
            param.grad=None


'''        for name,param in self.model.named_parameters():
            self.v_list[name]=self.momentum*self.v_list[name]-self.lr*param.grad
            param.data=(1-self.weight_decay)*param.data+self.v_list[name]'''