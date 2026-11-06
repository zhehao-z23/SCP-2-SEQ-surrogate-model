"""Three fixed strong heads; no encoder or architecture search."""
import torch
from torch import nn
from torch.nn import functional as F
class PoolingModel(nn.Module):
 def __init__(self,mode,dropout=.3):
  super().__init__();self.mode=mode
  assert mode in ['residue_mean','mean_pre','charge_deep']
  self.first=nn.Linear(78 if mode=='charge_deep' else 1038,1024)
  self.head=nn.Sequential(nn.Dropout(dropout),nn.Linear(1024,512),nn.GELU(),nn.Dropout(dropout),nn.Linear(512,256),nn.GELU(),nn.Dropout(dropout),nn.Linear(256,63))
 def forward(self,x,extra,mask=None):
  if self.mode=='charge_deep':z=F.gelu(self.first(extra))
  elif self.mode=='mean_pre':z=F.gelu(self.first(torch.cat([x,extra],-1)))
  else:
   assert mask is not None and mask.any(1).all()
   z=F.gelu(F.linear(x,self.first.weight[:,:960])+F.linear(extra,self.first.weight[:,960:],self.first.bias)[:,None]);z=(z*mask[:,:,None]).sum(1)/mask.sum(1)[:,None]
  return self.head(z)

def check_model():
 torch.manual_seed(1701);a=PoolingModel('residue_mean').eval();b=PoolingModel('mean_pre').eval();b.load_state_dict(a.state_dict());x=torch.randn(2,7,960);e=torch.randn(2,78);m=torch.ones(2,7,dtype=torch.bool);m[0,4:]=False
 with torch.no_grad():
  assert torch.allclose(a(x[:,:1],e,m[:,:1]),b(x[:,0],e),atol=1e-6)
  xp=torch.cat([x,torch.randn(2,4,960)*100],1);mp=torch.cat([m,torch.zeros(2,4,dtype=torch.bool)],1);assert torch.allclose(a(x,e,m),a(xp,e,mp),atol=1e-6)
 return {'single_residue_equivalence':True,'padding_invariant':True,'parameters':{mode:sum(p.numel() for p in PoolingModel(mode).parameters()) for mode in ['residue_mean','mean_pre','charge_deep']}}
if __name__=='__main__':print(check_model())
