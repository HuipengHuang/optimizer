import torch
class AdaGrad(torch.nn.Module):
    def __init__(self,model,lr,weight_decay=0,constant=1e-7):
        super(AdaGrad, self).__init__()
        self.lr = lr
        self.weight_decay = weight_decay
        self.model=model
        self.constant = constant
        self.r_dict={name:torch.zeros_like(param.data) for name, param in model.named_parameters()}
    def step(self):
        for name,param in self.model.named_parameters():
            g=param.grad+self.weight_decay*param.data
            self.r_dict[name]+=g**2
            param.data-=self.lr*g/(self.r_dict[name]**0.5+self.constant)
    def zero_grad(self):
        for param in self.model.parameters():
            param.grad=None