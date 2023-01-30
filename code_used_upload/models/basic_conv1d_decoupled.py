import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from fastai.layers import *
from fastai.core import *

##############################################################################################################################################
# utility functions


def _conv1d(in_planes,out_planes,kernel_size=3, stride=1, dilation=1, act="relu", bn=True, drop_p=0):
    lst=[]
    if(drop_p>0):
        lst.append(nn.Dropout(drop_p))
    lst.append(nn.Conv1d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, dilation=dilation, bias=not(bn)))
    if(bn):
        lst.append(nn.BatchNorm1d(out_planes))
    if(act=="relu"):
        lst.append(nn.ReLU(True))
    if(act=="elu"):
        lst.append(nn.ELU(True))
    if(act=="prelu"):
        lst.append(nn.PReLU(True))
    return nn.Sequential(*lst)

def _fc(in_planes,out_planes, act="relu", bn=True):
    lst = [nn.Linear(in_planes, out_planes, bias=not(bn))]
    if(bn):
        lst.append(nn.BatchNorm1d(out_planes))
    if(act=="relu"):
        lst.append(nn.ReLU(True))
    if(act=="elu"):
        lst.append(nn.ELU(True))
    if(act=="prelu"):
        lst.append(nn.PReLU(True))
    return nn.Sequential(*lst)

def cd_adaptiveconcatpool(relevant, irrelevant, module):
    mpr, mpi = module.mp.attrib(relevant,irrelevant)
    apr, api = module.ap.attrib(relevant,irrelevant)
    return torch.cat([mpr, apr], 1), torch.cat([mpi, api], 1)
def attrib_adaptiveconcatpool(self,relevant,irrelevant):
    return cd_adaptiveconcatpool(relevant,irrelevant,self)

class AdaptiveConcatPool1d(nn.Module):
    "Layer that concats `AdaptiveAvgPool1d` and `AdaptiveMaxPool1d`."
    def __init__(self, sz:Optional[int]=None):
        "Output will be 2*sz or 2 if sz is None"
        super().__init__()
        sz = sz or 1
        self.ap,self.mp = nn.AdaptiveAvgPool1d(sz), nn.AdaptiveMaxPool1d(sz)
    def forward(self, x): return torch.cat([self.mp(x), self.ap(x)], 1)
    def attrib(self,relevant,irrelevant):
        return attrib_adaptiveconcatpool(self,relevant,irrelevant)
    
class SqueezeExcite1d(nn.Module):
    '''squeeze excite block as used for example in LSTM FCN'''
    def __init__(self,channels,reduction=16):
        super().__init__()
        channels_reduced = channels//reduction
        self.w1 = torch.nn.Parameter(torch.randn(channels_reduced,channels).unsqueeze(0))
        self.w2 = torch.nn.Parameter(torch.randn(channels, channels_reduced).unsqueeze(0))

    def forward(self, x):
        #input is bs,ch,seq
        z=torch.mean(x,dim=2,keepdim=True)#bs,ch
        intermed = F.relu(torch.matmul(self.w1,z))#(1,ch_red,ch * bs,ch,1) = (bs, ch_red, 1)
        s=F.sigmoid(torch.matmul(self.w2,intermed))#(1,ch,ch_red * bs, ch_red, 1=bs, ch, 1
        return s*x #bs,ch,seq * bs, ch,1 = bs,ch,seq

def weight_init(m):
    '''call weight initialization for model n via n.appy(weight_init)'''
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    if isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight,1)
        nn.init.constant_(m.bias,0)
    if isinstance(m,SqueezeExcite1d):
        stdv1=math.sqrt(2./m.w1.size[0])
        nn.init.normal_(m.w1,0.,stdv1)
        stdv2=math.sqrt(1./m.w2.size[1])
        nn.init.normal_(m.w2,0.,stdv2)

class catAdaptiveConcatPool1d(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()
        self.op = AdaptiveConcatPool1d()

    def forward(self, x):
        cnn, gru = x
        pool = self.op(cnn)
        return [pool, gru]


class catMaxPool1d(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()
        self.op = nn.MaxPool1d(2)

    def forward(self, x):
        cnn, gru = x
        pool = self.op(cnn)
        return [pool, gru]

class catFlatten(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()
        self.op = Flatten()

    def forward(self, x):
        cnn, gru = x
        flatten = self.op(cnn)
        dense = torch.cat([flatten, gru], dim=-1)
        return dense

class rSE(nn.Module):
    def __init__(self, nin, reduce=16):
        super(rSE, self).__init__()
        self.nin = nin
        self.se = nn.Sequential(nn.Linear(self.nin, self.nin // reduce),
                                nn.ReLU(inplace=True),
                                nn.Linear(self.nin // reduce, self.nin),
                                nn.Sigmoid())

        self.rse = nn.Sequential(nn.Linear(self.nin, self.nin),
                                nn.Sigmoid())
    def forward(self, x):

        em = x[:, :, 0]
        div = x[:, :, 1]

        #diff = torch.abs(em - div)

        prob_em = self.rse(torch.ones_like(em))

        #prob_em = self.se(diff)

        prob_div = 1 - prob_em

        out = em * prob_em + div * prob_div

        return out

class DivOutLayer(nn.Module):

    def __init__(self, em_structure, div_structure, bn, drop_rate, em_actns, div_actns, cls_num, metric_out_dim, if_train, **kwargs):
        super().__init__()
        self.em_stru = em_structure
        self.div_stru = div_structure
        self.bn = bn
        self.drop_rate = drop_rate
        self.em_actns = em_actns
        self.div_actns = div_actns
        self.cls_num = cls_num
        self.metric_out_dim = metric_out_dim
        self.if_train = if_train
        self.baskets = nn.ModuleList()
        self.em_basket = nn.ModuleList()
        self.aggre = rSE(nin=cls_num, reduce=cls_num // 2)

        for ni, no, p, actn in zip(self.em_stru[:-1], self.em_stru[1:], self.drop_rate, self.em_actns):
            bag = []
            if self.bn:
                bag.append(nn.BatchNorm1d(ni).cuda())
            if p != 0:
                bag.append(nn.Dropout(p).cuda())

            bag.append(nn.Linear(ni, no).cuda())

            if actn != None:
                bag.append(actn)
            bag = nn.Sequential(*bag)
            self.em_basket.append(bag)


        for div_num in range(self.cls_num):
            sub_basket = nn.ModuleList()
            for ni, no, p, actn in zip(self.div_stru[:-1], self.div_stru[1:], self.drop_rate, self.div_actns):
                bag = []

                if self.bn:
                    bag.append(nn.BatchNorm1d(ni).cuda())
                if p != 0:
                    bag.append(nn.Dropout(p).cuda())

                bag.append(nn.Linear(ni, no).cuda())

                if actn != None:
                    bag.append(actn)
                bag = nn.Sequential(*bag)
                sub_basket.append(bag)

            self.baskets.append(sub_basket)




    def forward(self, x):
        cat_out = []
        feats = []
        count = 0

        x_em_deal = x

        for layer in self.em_basket:
            x_em_deal = layer(x_em_deal)

        x_em_deal = torch.unsqueeze(x_em_deal, dim=-1)

        for layers in self.baskets:
            count += 1
            x_deal = x
            for layer in layers:
                x_deal = layer(x_deal)
                if x_deal.shape[-1] == self.metric_out_dim:
                    x_deal_feat = F.normalize(x_deal, p=2, dim=-1)
                    feats.append(x_deal_feat)

                if x_deal.shape[-1] == 1 and count == 1:
                    cat_out = x_deal
                if x_deal.shape[-1] == 1 and count != 1:
                    cat_out = torch.cat((cat_out, x_deal), dim=-1)

        cat_out = torch.unsqueeze(cat_out, dim=-1)

        out = self.aggre(torch.cat((x_em_deal, cat_out), dim=-1))
        if self.if_train == True:
            return [out, feats]
        else:
            return out

def create_head1d(nf:int, nc:int, lin_ftrs:Optional[Collection[int]]=None, ps:Floats=0.5, bn_final:bool=False, bn:bool=True, act="relu", concat_pooling=True):
    "Model head that takes `nf` features, runs through `lin_ftrs`, and about `nc` classes; added bn and act here"
    lin_ftrs = [2*nf if concat_pooling else nf, nc] if lin_ftrs is None else [2*nf if concat_pooling else nf] + lin_ftrs + [nc] #was [nf, 512,nc]
    ps = listify(ps)
    if len(ps)==1: ps = [ps[0]/2] * (len(lin_ftrs)-2) + ps
    actns = [nn.ReLU(inplace=True) if act=="relu" else nn.ELU(inplace=True)] * (len(lin_ftrs)-2) + [None]
    layers = [AdaptiveConcatPool1d() if concat_pooling else nn.MaxPool1d(2), Flatten()]
    for ni,no,p,actn in zip(lin_ftrs[:-1],lin_ftrs[1:],ps,actns):
        layers += bn_drop_lin(ni,no,bn,p,actn)
    if bn_final: layers.append(nn.BatchNorm1d(lin_ftrs[-1], momentum=0.01))
    return nn.Sequential(*layers)

def create_head1d_decoupled(nf:int, nc:int, lin_ftrs:Optional[Collection[int]]=None, div_lin_ftrs:Optional[Collection[int]]=None, ps:Floats=0.5, bn_final:bool=False, bn:bool=True, act="relu", concat_pooling=True, if_train=True):
    "Model head that takes `nf` features, runs through `lin_ftrs`, and about `nc` classes; added bn and act here"
    lin_ftrs = [2*nf if concat_pooling else nf, nc] if lin_ftrs is None else [2*nf if concat_pooling else nf] + lin_ftrs + [nc]  # was [nf, 512,nc]
    div_lin_ftrs = [2*nf if concat_pooling else nf, nc] if div_lin_ftrs is None else [2*nf if concat_pooling else nf] + div_lin_ftrs + [1] #was [nf, 512,nc]
    ps = listify(ps)
    if len(ps)==1: ps = [ps[0]/2] * (len(lin_ftrs)-2) + ps
    em_actns = [nn.ReLU(inplace=True) if act == "relu" else nn.ELU(inplace=True)] * (len(lin_ftrs) - 2) + [None]
    div_actns = [nn.ReLU(inplace=True) if act=="relu" else nn.ELU(inplace=True)] * (len(div_lin_ftrs)-3) + [None, None]
    layers = [AdaptiveConcatPool1d() if concat_pooling else nn.MaxPool1d(2), Flatten(),
              DivOutLayer(em_structure=lin_ftrs, div_structure=div_lin_ftrs, bn=bn, drop_rate=ps, em_actns=em_actns, div_actns=div_actns, cls_num=nc, metric_out_dim=div_lin_ftrs[-2], if_train=if_train)]

    if bn_final: layers.append(nn.BatchNorm1d(lin_ftrs[-1], momentum=0.01))
    return nn.Sequential(*layers)
##############################################################################################################################################
# basic convolutional architecture

class basic_conv1d(nn.Sequential):
    '''basic conv1d'''
    def __init__(self, filters=[128,128,128,128],kernel_size=3, stride=2, dilation=1, pool=0, pool_stride=1, squeeze_excite_reduction=0, num_classes=2, input_channels=8, act="relu", bn=True, headless=False,split_first_layer=False,drop_p=0.,lin_ftrs_head=None, ps_head=0.5, bn_final_head=False, bn_head=True, act_head="relu", concat_pooling=True):
        layers = []
        if(isinstance(kernel_size,int)):
            kernel_size = [kernel_size]*len(filters)
        for i in range(len(filters)):
            layers_tmp = []
            
            layers_tmp.append(_conv1d(input_channels if i==0 else filters[i-1],filters[i],kernel_size=kernel_size[i],stride=(1 if (split_first_layer is True and i==0) else stride),dilation=dilation,act="none" if ((headless is True and i==len(filters)-1) or (split_first_layer is True and i==0)) else act, bn=False if (headless is True and i==len(filters)-1) else bn,drop_p=(0. if i==0 else drop_p)))
            if((split_first_layer is True and i==0)):
                layers_tmp.append(_conv1d(filters[0],filters[0],kernel_size=1,stride=1,act=act, bn=bn,drop_p=0.))
                #layers_tmp.append(nn.Linear(filters[0],filters[0],bias=not(bn)))
                #layers_tmp.append(_fc(filters[0],filters[0],act=act,bn=bn))
            if(pool>0 and i<len(filters)-1):
                layers_tmp.append(nn.MaxPool1d(pool,stride=pool_stride,padding=(pool-1)//2))
            if(squeeze_excite_reduction>0):
                layers_tmp.append(SqueezeExcite1d(filters[i],squeeze_excite_reduction))
            layers.append(nn.Sequential(*layers_tmp))

        #head
        #layers.append(nn.AdaptiveAvgPool1d(1))    
        #layers.append(nn.Linear(filters[-1],num_classes))
        #head #inplace=True leads to a runtime error see ReLU+ dropout https://discuss.pytorch.org/t/relu-dropout-inplace/13467/5
        self.headless = headless
        if(headless is True):
            head = nn.Sequential(nn.AdaptiveAvgPool1d(1),Flatten())
        else:
            head=create_head1d(filters[-1], nc=num_classes, lin_ftrs=lin_ftrs_head, ps=ps_head, bn_final=bn_final_head, bn=bn_head, act=act_head, concat_pooling=concat_pooling)
        layers.append(head)
        
        super().__init__(*layers)
    
    def get_layer_groups(self):
        return (self[2],self[-1])

    def get_output_layer(self):
        if self.headless is False:
            return self[-1][-1]
        else:
            return None
    
    def set_output_layer(self,x):
        if self.headless is False:
            self[-1][-1] = x


class basic_conv1d_decoupled(nn.Sequential):
    '''basic conv1d'''

    def __init__(self, filters=[128, 128, 128, 128], kernel_size=3, stride=2, dilation=1, pool=0, pool_stride=1,
                 squeeze_excite_reduction=0, num_classes=2, input_channels=8, act="relu", bn=True, headless=False,
                 split_first_layer=False, drop_p=0., lin_ftrs_head=None, div_lin_ftrs_head=None, ps_head=0.5, bn_final_head=False, bn_head=True,
                 act_head="relu", concat_pooling=True, if_train=True):
        layers = []
        if (isinstance(kernel_size, int)):
            kernel_size = [kernel_size] * len(filters)
        for i in range(len(filters)):
            layers_tmp = []

            layers_tmp.append(
                _conv1d(input_channels if i == 0 else filters[i - 1], filters[i], kernel_size=kernel_size[i],
                        stride=(1 if (split_first_layer is True and i == 0) else stride), dilation=dilation,
                        act="none" if ((headless is True and i == len(filters) - 1) or (
                                    split_first_layer is True and i == 0)) else act,
                        bn=False if (headless is True and i == len(filters) - 1) else bn,
                        drop_p=(0. if i == 0 else drop_p)))
            if ((split_first_layer is True and i == 0)):
                layers_tmp.append(_conv1d(filters[0], filters[0], kernel_size=1, stride=1, act=act, bn=bn, drop_p=0.))
                # layers_tmp.append(nn.Linear(filters[0],filters[0],bias=not(bn)))
                # layers_tmp.append(_fc(filters[0],filters[0],act=act,bn=bn))
            if (pool > 0 and i < len(filters) - 1):
                layers_tmp.append(nn.MaxPool1d(pool, stride=pool_stride, padding=(pool - 1) // 2))
            if (squeeze_excite_reduction > 0):
                layers_tmp.append(SqueezeExcite1d(filters[i], squeeze_excite_reduction))
            layers.append(nn.Sequential(*layers_tmp))

        # head
        # layers.append(nn.AdaptiveAvgPool1d(1))
        # layers.append(nn.Linear(filters[-1],num_classes))
        # head #inplace=True leads to a runtime error see ReLU+ dropout https://discuss.pytorch.org/t/relu-dropout-inplace/13467/5
        self.headless = headless
        if (headless is True):
            head = nn.Sequential(nn.AdaptiveAvgPool1d(1), Flatten())
        else:
            head = create_head1d_decoupled(filters[-1], nc=num_classes, lin_ftrs=lin_ftrs_head, div_lin_ftrs=div_lin_ftrs_head, ps=ps_head,
                                 bn_final=bn_final_head, bn=bn_head, act=act_head, concat_pooling=concat_pooling, if_train=if_train)
        layers.append(head)

        super().__init__(*layers)

    def get_layer_groups(self):
        return (self[2], self[-1])

    def get_output_layer(self):
        if self.headless is False:
            return self[-1][-1]
        else:
            return None

    def set_output_layer(self, x):
        if self.headless is False:
            self[-1][-1] = x
 
############################################################################################
# convenience functions for basic convolutional architectures

def fcn(filters=[128]*5,num_classes=2,input_channels=8):
    filters_in = filters + [num_classes]
    return basic_conv1d(filters=filters_in,kernel_size=3,stride=1,pool=2,pool_stride=2,input_channels=input_channels,act="relu",bn=True,headless=True)

def fcn_wang(num_classes=2,input_channels=8,lin_ftrs_head=None, ps_head=0.5, bn_final_head=False, bn_head=True, act_head="relu", concat_pooling=True):
    return basic_conv1d(filters=[128,256,128],kernel_size=[8,5,3],stride=1,pool=0,pool_stride=2, num_classes=num_classes,input_channels=input_channels,act="relu",bn=True,lin_ftrs_head=lin_ftrs_head, ps_head=ps_head, bn_final_head=bn_final_head, bn_head=bn_head, act_head=act_head, concat_pooling=concat_pooling)

def fcn_wang_decoupled(num_classes=2,input_channels=8,lin_ftrs_head=None, div_lin_ftrs_head=None, ps_head=0.5, bn_final_head=False, bn_head=True, act_head="relu", concat_pooling=True, if_train=True):
    return basic_conv1d_decoupled(filters=[128,256,128],kernel_size=[8,5,3],stride=1,pool=0,pool_stride=2, num_classes=num_classes,input_channels=input_channels,act="relu",bn=True,lin_ftrs_head=lin_ftrs_head, div_lin_ftrs_head=div_lin_ftrs_head, ps_head=ps_head, bn_final_head=bn_final_head, bn_head=bn_head, act_head=act_head, concat_pooling=concat_pooling, if_train=if_train)

def schirrmeister(num_classes=2,input_channels=8,lin_ftrs_head=None, ps_head=0.5, bn_final_head=False, bn_head=True, act_head="relu", concat_pooling=True):
    return basic_conv1d(filters=[25,50,100,200],kernel_size=10, stride=3, pool=3, pool_stride=1, num_classes=num_classes, input_channels=input_channels, act="relu", bn=True, headless=False,split_first_layer=True,drop_p=0.5,lin_ftrs_head=lin_ftrs_head, ps_head=ps_head, bn_final_head=bn_final_head, bn_head=bn_head, act_head=act_head, concat_pooling=concat_pooling)

def sen(filters=[128]*5,num_classes=2,input_channels=8,squeeze_excite_reduction=16,drop_p=0.,lin_ftrs_head=None, ps_head=0.5, bn_final_head=False, bn_head=True, act_head="relu", concat_pooling=True):
    return basic_conv1d(filters=filters,kernel_size=3,stride=2,pool=0,pool_stride=0,input_channels=input_channels,act="relu",bn=True,num_classes=num_classes,squeeze_excite_reduction=squeeze_excite_reduction,drop_p=drop_p,lin_ftrs_head=lin_ftrs_head, ps_head=ps_head, bn_final_head=bn_final_head, bn_head=bn_head, act_head=act_head, concat_pooling=concat_pooling)

def basic1d(filters=[128]*5,kernel_size=3, stride=2, dilation=1, pool=0, pool_stride=1, squeeze_excite_reduction=0, num_classes=2, input_channels=8, act="relu", bn=True, headless=False,drop_p=0.,lin_ftrs_head=None, ps_head=0.5, bn_final_head=False, bn_head=True, act_head="relu", concat_pooling=True):
    return basic_conv1d(filters=filters,kernel_size=kernel_size, stride=stride, dilation=dilation, pool=pool, pool_stride=pool_stride, squeeze_excite_reduction=squeeze_excite_reduction, num_classes=num_classes, input_channels=input_channels, act=act, bn=bn, headless=headless,drop_p=drop_p,lin_ftrs_head=lin_ftrs_head, ps_head=ps_head, bn_final_head=bn_final_head, bn_head=bn_head, act_head=act_head, concat_pooling=concat_pooling)

