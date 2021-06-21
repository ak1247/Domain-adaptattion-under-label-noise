from torchvision import models
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Function, Variable
#from LinearAverage import *
from .resnetmodel import *


class GradReverse(Function):
    @staticmethod
    def forward(ctx, x,lambd):
        ctx.save_for_backward(lambd)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        lambd=ctx.saved_tensors[0]
        return grad_output.neg()*lambd, None
def grad_reverse(x,lambd=1.0):
    return GradReverse.apply(x, Variable(torch.ones(1)*lambd).cuda())


class ResBase(nn.Module):
    def __init__(self, option='resnet50', pret=True, unit_size=100):
        super(ResBase, self).__init__()
        self.dim = 2048
        if option == 'resnet18':
            model_ft = models.resnet18(pretrained=pret)
            self.dim = 512
        if option == 'resnet50':
            model_ft = models.resnet50(pretrained=pret)
            # model_ft=resnet50(pretrained=False)
            # state_dict=torch.load('../DANCE_CoTuning/resnet50_l2_eps0.01.ckpt')
            # dict0=state_dict['model']
            # dict1={k:v for k,v in dict0.items()}
            # dict2={}
            
            # model_dict = model_ft.state_dict()
            
            # for key,value in model_dict.items():
            #     dict2[key]=dict1["module.attacker.model."+key]
            # model_ft.load_state_dict(dict2)
            
        if option == 'resnet101':
            model_ft = models.resnet101(pretrained=pret)
        if option == 'resnet152':
            model_ft = models.resnet152(pretrained=pret)
        mod = list(model_ft.children())
        self.fc=model_ft.fc
        mod.pop()
        #print(mod)
        self.features = nn.Sequential(*mod)


    def forward(self, x):
        x = self.features(x)
        #print(x.shape)
        x = x.view(x.size(0), self.dim)
        x1=self.fc(x)
        #print(x.shape)
        return x,x1


class ResClassifier_MME(nn.Module):
    def __init__(self, num_classes=12, input_size=2048, temp=0.05):
        super(ResClassifier_MME, self).__init__()
        self.fc = nn.Linear(input_size, num_classes, bias=False)
        self.tmp = temp

    def set_lambda(self, lambd):
        self.lambd = lambd

    def forward(self, x, dropout=False, return_feat=False, reverse=False):
        if return_feat:
            return x
        x = F.normalize(x)
        x = self.fc(x)/self.tmp
        return x

    def weight_norm(self):
        w = self.fc.weight.data
        norm = w.norm(p=2, dim=1, keepdim=True)
        self.fc.weight.data = w.div(norm.expand_as(w))

    def weights_init(self, m):
        m.weight.data.normal_(0.0, 0.1)
