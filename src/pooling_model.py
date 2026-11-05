"""Equal-parameter aggregation controls over the same frozen residue cache."""
import torch
from torch import nn
from torch.nn import functional as F

class PoolingModel(nn.Module):
 def __init__(self,mode,dropout=.3):
  super().__init__();self.mode=mode
  self.first=nn.Linear(1038,1024)
  self.head=nn.Sequential(nn.Dropout(dropout),nn.Linear(1024,512),nn.GELU(),nn.Dropout(dropout),nn.Linear(512,256),nn.GELU(),nn.Dropout(dropout),nn.Linear(256,63))
  if mode=='residue_attention':self.score=nn.Sequential(nn.Linear(1024,64),nn.Tanh(),nn.Linear(64,1))
 def forward(self,x,extra,mask=None):
  if self.mode in ['mean_pre','max_pre']:
   assert x.ndim==2
   z=F.gelu(self.first(torch.cat([x,extra],-1)))
  else:
   assert mask is not None and mask.any(1).all()
   z=F.gelu(F.linear(x,self.first.weight[:,:960])+F.linear(extra,self.first.weight[:,960:],self.first.bias)[:,None])
   if self.mode=='residue_mean':z=(z*mask[:,:,None]).sum(1)/mask.sum(1)[:,None]
   elif self.mode=='residue_attention':
    s=self.score(z).squeeze(-1).masked_fill(~mask,float('-inf'));z=(s.softmax(1)[:,:,None]*z).sum(1)
   else:raise ValueError(self.mode)
  return self.head(z)

def check_model():
 torch.manual_seed(1701);m=PoolingModel('residue_mean').eval();x=torch.randn(2,9,960);e=torch.randn(2,78);mask=torch.ones(2,9,dtype=torch.bool);mask[0,5:]=False
 with torch.no_grad():
  a=m(x,e,mask);p=torch.cat([x,torch.randn(2,7,960)*100],1);pm=torch.cat([mask,torch.zeros(2,7,dtype=torch.bool)],1)
  assert torch.allclose(a,m(p,e,pm),atol=1e-6), 'padding contamination'
  v=PoolingModel('mean_pre').eval();v.load_state_dict(m.state_dict());single=x[:,:1]
  assert torch.allclose(m(single,e,mask[:,:1]),v(single[:,0],e),atol=1e-6),'single residue equivalence'
 assert sum(p.numel() for p in m.parameters())==sum(p.numel() for p in v.parameters())==1736255
 return {'padding_invariance':True,'single_residue_equivalence':True,'matched_parameters':1736255}

if __name__=='__main__':print(check_model())
