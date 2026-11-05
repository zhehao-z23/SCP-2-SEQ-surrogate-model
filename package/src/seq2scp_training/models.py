"""Trainable readouts over vectors or masked, ordered residue representations."""
import math
import torch
from torch import nn
from torch.nn import functional as F


def mlp(width, hidden, out, dropout):
    modules=[]
    for h in hidden:
        modules += [nn.Linear(width,h),nn.GELU(),nn.Dropout(dropout)];width=h
    return nn.Sequential(*modules,nn.Linear(width,out))


def masked_mean(x,mask):
    return (x*mask.unsqueeze(-1)).sum(1)/mask.sum(1,keepdim=True).clamp_min(1)


def position_encoding(length,width,device,dtype):
    pos=torch.arange(length,device=device,dtype=torch.float32)[:,None]
    scale=torch.exp(torch.arange(0,width,2,device=device,dtype=torch.float32)*(-math.log(10000)/width))
    pe=torch.zeros(length,width,device=device)
    pe[:,0::2]=torch.sin(pos*scale)
    pe[:,1::2]=torch.cos(pos*scale[:width//2])
    return pe.to(dtype)


class ResidueStack(nn.Module):
    def __init__(self,in_dim,width,family,layers=3,kernel=5,heads=4,ff=1024,dropout=.1):
        super().__init__();self.family=family
        if family == "cnn":
            if kernel%2!=1: raise ValueError("odd kernel required")
            self.first=nn.Conv1d(in_dim,width,kernel,padding=kernel//2)
            self.blocks=nn.ModuleList([nn.Conv1d(width,width,kernel,padding=(kernel//2)*2**i,dilation=2**i) for i in range(1,layers)])
            self.norms=nn.ModuleList([nn.LayerNorm(width) for _ in range(layers)])
            self.drop=nn.Dropout(dropout)
        elif family == "transformer":
            self.projection=nn.Linear(in_dim,width)
            self.transformer=nn.TransformerEncoder(nn.TransformerEncoderLayer(width,heads,ff,dropout,batch_first=True,norm_first=True),num_layers=layers,enable_nested_tensor=False)
        else: raise ValueError("unknown ordered family")

    def forward(self,x,mask):
        m=mask.unsqueeze(-1)
        if self.family=="cnn":
            x=self.drop(F.gelu(self.norms[0](self.first((x*m).transpose(1,2)).transpose(1,2))))*m
            for conv,norm in zip(self.blocks,self.norms[1:]):
                x=(x+self.drop(F.gelu(norm(conv(x.transpose(1,2)).transpose(1,2)))))*m
            return x
        x=(self.projection(x)+position_encoding(x.shape[1],self.projection.out_features,x.device,x.dtype))*m
        return self.transformer(x,src_key_padding_mask=~mask)*m


class SequenceModel(nn.Module):
    """One primary U output; optional fragment readout shares trainable residue layers."""
    def __init__(self,input_dim,bins,kind="matrix",pool="mean",width=256,hidden=(),residue_hidden=(256,128),summaries=1,score_width=64,layers=3,heads=4,ff=1024,kernel=5,dropout=.1,vocab_size=27,embedding_dim=128,auxiliary=False,stats_dim=0,concat_dim=0,layer_count=1):
        super().__init__()
        if bins<2 or kind not in {"vector","matrix","tokens","onehot"}: raise ValueError("model contract")
        if pool not in {"mean","attention","residue_mlp","cnn","transformer"}: raise ValueError("unknown pool")
        if kind=="vector" and (auxiliary or layer_count!=1 or pool!="mean"): raise ValueError("vector route is fixed pooling; fragment auxiliary requires residue context")
        self.kind=kind;self.pool=pool;self.vocab_size=vocab_size;self.concat_dim=concat_dim
        self.layer_logits=nn.Parameter(torch.zeros(layer_count)) if layer_count>1 else None
        self.embedding=nn.Embedding(vocab_size,embedding_dim,padding_idx=0) if kind=="tokens" else None
        in_dim=embedding_dim if kind=="tokens" else vocab_size if kind=="onehot" else input_dim
        self.residue=nn.Identity()
        if pool in {"cnn","transformer"}:
            self.residue=ResidueStack(in_dim,width,pool,layers,kernel,heads,ff,dropout);dim=width
        elif pool=="residue_mlp":
            if not residue_hidden: raise ValueError("residue hidden required")
            self.residue=mlp(in_dim,list(residue_hidden[:-1]),residue_hidden[-1],dropout);dim=residue_hidden[-1]
        elif pool=="attention" or auxiliary:
            self.residue=nn.Sequential(nn.Linear(in_dim,width),nn.GELU());dim=width
        else: dim=in_dim
        self.attention=nn.Sequential(nn.Linear(dim,score_width),nn.Tanh(),nn.Linear(score_width,summaries)) if pool=="attention" else None
        read_dim=dim*(summaries if self.attention is not None else 1)+concat_dim
        self.head=mlp(read_dim,hidden,bins,dropout)
        self.fragment_head=mlp(read_dim,hidden,bins,dropout) if auxiliary else None
        self.stats_head=nn.Linear(read_dim,stats_dim) if stats_dim else None

    def _pool(self,x,mask):
        if self.attention is None: return masked_mean(x,mask)
        scores=self.attention(x).masked_fill(~mask.unsqueeze(-1),float('-inf'))
        weight=torch.softmax(scores.float(),dim=1).to(x.dtype)
        return torch.einsum('bls,bld->bsd',weight,x).flatten(1)

    def forward(self,x,mask=None,fragment_mask=None,extra=None):
        if self.layer_logits is not None:
            if x.ndim!=4 or x.shape[1]!=len(self.layer_logits): raise ValueError("layer matrix mismatch")
            x=(x*self.layer_logits.softmax(0)[None,:,None,None]).sum(1)
        if self.kind=="vector":
            if x.ndim!=2: raise ValueError("vector shape")
            z=x
        else:
            if mask is None or mask.dtype!=torch.bool or mask.ndim!=2 or not mask.any(1).all(): raise ValueError("nonempty residue mask required")
            if self.kind=="tokens": x=self.embedding(x)
            elif self.kind=="onehot": x=F.one_hot(x,num_classes=self.vocab_size).float()
            if x.ndim!=3 or x.shape[:2]!=mask.shape: raise ValueError("matrix mask mismatch")
            x=x*mask.unsqueeze(-1)
            x=self.residue(x,mask) if self.pool in {"cnn","transformer"} else self.residue(x)
            z=self._pool(x,mask)
        def combine(z):
            if self.concat_dim:
                if extra is None or extra.shape!=(len(z),self.concat_dim): raise ValueError("concat schema mismatch")
                return torch.cat([z,extra],dim=-1)
            if extra is not None: raise ValueError("unexpected concat features")
            return z
        z=combine(z);out={"logits":self.head(z)}
        if self.stats_head is not None: out["stats"]=self.stats_head(z)
        if fragment_mask is not None:
            if self.fragment_head is None or mask is None or fragment_mask.shape!=mask.shape or fragment_mask.dtype!=torch.bool or (fragment_mask & ~mask).any() or not fragment_mask.any(1).all(): raise ValueError("invalid auxiliary fragment context")
            out["fragment_logits"]=self.fragment_head(combine(self._pool(x,fragment_mask)))
        return out


def parameter_counts(model):
    trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)
    total=sum(p.numel() for p in model.parameters())
    return {"total":total,"trainable":trainable,"frozen":total-trainable,"scope":"head/scratch model only; external pLM excluded"}
