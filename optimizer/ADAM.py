import torch
import torch.nn as nn


class ADAM(nn.Module):
    def __init__(self, model, lr, weight_decay=0, e1=0.9, e2=0.999, constant=1e-8):
        super(ADAM, self).__init__()
        self.lr = lr
        self.weight_decay = weight_decay
        self.e1 = e1
        self.e2 = e2
        self.constant = constant
        self.t = 0
        self.model = model
        self.s_dict = {name: torch.zeros_like(param) for name, param in model.named_parameters()}
        self.r_dict = {name: torch.zeros_like(param) for name, param in model.named_parameters()}

    def step(self):
        self.t += 1
        s_hat_dict = {}
        r_hat_dict = {}

        # Update moving averages
        for name, param in self.model.named_parameters():
            if param.grad is None:
                continue
            g = param.grad + self.weight_decay * param.data
            self.s_dict[name] = self.e1 * self.s_dict[name] + (1 - self.e1) * g
            self.r_dict[name] = self.e2 * self.r_dict[name] + (1 - self.e2) * g ** 2

            # Bias correction
            s_hat = self.s_dict[name] / (1 - self.e1 ** self.t)
            r_hat = self.r_dict[name] / (1 - self.e2 ** self.t)

            # Parameter update
            param.data -= self.lr * s_hat / (r_hat.sqrt() + self.constant)

    def zero_grad(self):
        for param in self.model.parameters():
            param.grad=None