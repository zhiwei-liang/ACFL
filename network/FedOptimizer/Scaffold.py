import torch
from torch.optim.optimizer import Optimizer, required


class Scaffold(Optimizer):
    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, variance=0):
        
        self.itr = 0
        self.a_sum = 0
        

        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov, variance=variance)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(Scaffold, self).__init__(params, defaults)


    def __setstate__(self, state):
        super(Scaffold, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)
        
    
    def step(self, c_ci):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
                
        group['params'] is a list
        """

        loss = None
        group = self.param_groups[0]
        weight_decay = group['weight_decay']
        momentum = group['momentum']
        dampening = group['dampening']
        nesterov = group['nesterov']
        for p, control_c_ci in zip(group['params'], c_ci):
            if p.grad is None:
                continue
            d_p = p.grad.data
            
            d_p.add_(1.0, control_c_ci.data)
            
            if weight_decay != 0:
                d_p.add_(weight_decay, p.data)
            
            param_state = self.state[p]
            if momentum != 0:
                if 'momentum_buffer' not in param_state:
                    buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                else:
                    buf = param_state['momentum_buffer']
                    buf.mul_(momentum).add_(1 - dampening, d_p)
                if nesterov:
                    d_p = d_p.add(momentum, buf)
                else:
                    d_p = buf
            
            
            p.data.add_(-group['lr'], d_p)

        return loss


def GenParamList(model):
    raw_list = list(model.parameters())
    new_list = []
    for param in raw_list:
        new_list.append(param.clone())
    return new_list

def GenZeroParamList(model):
    c = GenParamList(model)
    for i in c:
        i.data.zero_()
    return c


def ListMinus(a,b):
    c = []
    for ai, bi in zip(a,b):
        c.append(ai - bi)
    return c

def UpdateLocalControl(c, ci, global_model, local_model, K):
    '''
    Use Option II in the paper as default
    ci = ci - c + 1/(K*lr) *(global_model - local_model)
    K means the update times of local model
    '''
    param_num = len(c)
    x = GenParamList(global_model)
    yi = GenParamList(local_model)
    
    weight = 1./K
    
    for i in range(param_num):
        ci[i] = ci[i] - c[i] + weight * (x[i] - yi[i])
        
    return ci
        

def UpdateServerControl(c, ci_dict, weight_dict):
    param_nums = len(c)
    for i in range(param_nums):
        new_c_value = 0
        for local_client, value in weight_dict.items():
            new_c_value += value * ci_dict[local_client][i]
        c[i] = new_c_value
    return c
