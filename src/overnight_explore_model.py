"""Attention optimization probes; unchanged 1.736M residue transform/readout."""
import torch
from torch import nn
from torch.nn import functional as F

class PoolingModel(nn.Module):
 def __init__(self,mode,dropout=.3):
  super().__init__();self.mode=mode;self.diagnostics=[]
  self.first=nn.Linear(1038,1024)
  self.head=nn.Sequential(nn.Dropout(dropout),nn.Linear(1024,512),nn.GELU(),nn.Dropout(dropout),nn.Linear(512,256),nn.GELU(),nn.Dropout(dropout),nn.Linear(256,63))
  if mode!='mean_continue':
   self.score=nn.Sequential(nn.Linear(1024,64),nn.Tanh(),nn.Linear(64,1))
   if mode in ['uniform_scratch','uniform_warm']:
    nn.init.zeros_(self.score[-1].weight);nn.init.zeros_(self.score[-1].bias)
   elif mode=='gated_warm':self.gate=nn.Parameter(torch.zeros(()))
   elif mode!='random_attention':raise ValueError(mode)
 def forward(self,x,extra,mask):
  assert x.ndim==3 and mask.any(1).all()
  z=F.gelu(F.linear(x,self.first.weight[:,:960])+F.linear(extra,self.first.weight[:,960:],self.first.bias)[:,None])
  mean=(z*mask[:,:,None]).sum(1)/mask.sum(1)[:,None]
  if self.mode=='mean_continue':out=mean
  else:
   s=self.score(z).squeeze(-1).masked_fill(~mask,float('-inf'));a=s.softmax(1)
   attention=(a[:,:,None]*z).sum(1)
   out=mean+torch.tanh(self.gate)*(attention-mean) if self.mode=='gated_warm' else attention
   if not self.training:
    L=mask.sum(1);entropy=-(a*a.clamp_min(1e-30).log()).sum(1)/L.clamp_min(2).log()
    eff=1/(L*(a*a).sum(1));maxw=a.max(1).values
    self.diagnostics.append(torch.stack([entropy,eff,maxw],1).detach().cpu())
  return self.head(out)

def check_model():
 torch.manual_seed(1701);m=PoolingModel('mean_continue').eval();x=torch.randn(2,9,960);e=torch.randn(2,78);mask=torch.ones(2,9,dtype=torch.bool);mask[0,5:]=False
 result={}
 for mode in ['uniform_scratch','uniform_warm','gated_warm']:
  z=PoolingModel(mode).eval();z.load_state_dict(m.state_dict(),strict=False)
  assert torch.allclose(m(x,e,mask),z(x,e,mask),atol=1e-6),'mean identity at initialization'
  p=torch.cat([x,torch.randn(2,7,960)*100],1);pm=torch.cat([mask,torch.zeros(2,7,dtype=torch.bool)],1)
  assert torch.allclose(z(x,e,mask),z(p,e,pm),atol=1e-6),'padding contamination'
  z.zero_grad();z(x,e,mask).square().sum().backward()
  grad=z.gate.grad if mode=='gated_warm' else z.score[-1].weight.grad
  assert torch.isfinite(grad).all() and grad.abs().sum()>0,'dead attention pathway'
  result[mode]={'initial_mean_identity':True,'padding_invariance':True,'initial_attention_gradient_nonzero':True,'parameters':sum(p.numel() for p in z.parameters())}
 return result

if __name__=='__main__':print(check_model())
