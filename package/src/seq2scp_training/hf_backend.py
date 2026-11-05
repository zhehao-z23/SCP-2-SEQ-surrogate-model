"""Explicit local-checkpoint Hugging Face backend. Never acquires remote assets."""
from pathlib import Path
import hashlib,json
import numpy as np
from .contracts import sequence_hash
from .encoders import ENCODERS

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()

def json_safe(value):
    if isinstance(value,dict):return {k:json_safe(v) for k,v in value.items()}
    if isinstance(value,(set,tuple,list)):return [json_safe(v) for v in (sorted(value) if isinstance(value,set) else value)]
    return value

class LocalHFBackend:
    """Model runtime must be the interpreter fixed in the checkpoint manifest.

    Custom-code loading requires an explicit source-reviewed manifest. File hashes
    bind current bytes; they do not certify equivalence to another implementation.
    """
    def __init__(self,manifest,device='cpu'):
        import torch
        from transformers import AutoModel,AutoTokenizer
        self.torch=torch;self.manifest=manifest;self.device=device
        self.name=manifest['encoder'];self.root=Path(manifest['model_root']).resolve()
        if self.name not in ENCODERS:raise ValueError('out-of-scope encoder')
        if not self.root.is_dir() or not manifest.get('files'):raise ValueError('missing local checkpoint')
        for name,expected in manifest['files'].items():
            p=(self.root/name).resolve()
            if self.root not in p.parents or not p.is_file() or sha(p)!=expected:raise ValueError('checkpoint content drift: '+name)
        custom=bool(manifest.get('custom_code'))
        if custom and manifest.get('custom_source_reviewed') is not True:raise ValueError('custom runtime source not reviewed')
        self.tokenizer=AutoTokenizer.from_pretrained(str(self.root),local_files_only=True,trust_remote_code=custom)
        extra={'add_pooling_layer':False} if self.name=='esm2_t33_650M_UR50D' else {'attn_implementation':'sdpa'}
        self.model,load_info=AutoModel.from_pretrained(str(self.root),local_files_only=True,trust_remote_code=custom,torch_dtype=torch.float32,output_loading_info=True,**extra)
        if load_info.get('missing_keys') or load_info.get('mismatched_keys') or load_info.get('error_msgs'):
            raise ValueError('checkpoint did not initialize all encoder parameters: '+str(load_info))
        self.load_info=json_safe(load_info)
        self.model=self.model.to(device).eval()
        for p in self.model.parameters():p.requires_grad_(False)
        if self.model.config.hidden_size!=ENCODERS[self.name]:raise ValueError('checkpoint hidden width')
        self.layers=int(self.model.config.num_hidden_layers)

    def tokenize(self,sequences):
        data=self.tokenizer(sequences,padding=True,truncation=False,return_tensors='pt',return_special_tokens_mask=True)
        mask=data.pop('special_tokens_mask').bool()
        keep=data['attention_mask'].bool() & ~mask
        for i,s in enumerate(sequences):
            if int(keep[i].sum())!=len(s):raise ValueError('tokenizer changed residue count')
            ids=data['input_ids'][i,keep[i]].tolist()
            tokens=self.tokenizer.convert_ids_to_tokens(ids)
            if tokens!=list(s):raise ValueError('tokenizer cannot preserve an input residue; explicit policy required')
        return data,keep

    def extract_batch(self,sequences,layers,precision='fp32'):
        if not layers or any(type(n) is not int or n<1 or n>self.layers for n in layers):raise ValueError('invalid layer index')
        if precision not in {'fp32','bf16'}:raise ValueError('precision')
        data,keep=self.tokenize(sequences);torch=self.torch
        data={k:v.to(self.device) for k,v in data.items()}
        intermediate=any(n!=self.layers for n in layers)
        device_type=torch.device(self.device).type
        with torch.inference_mode(),torch.autocast(device_type=device_type,dtype=torch.bfloat16,enabled=precision=='bf16'):
            out=self.model(**data,output_hidden_states=intermediate,return_dict=True)
        result=[]
        for i,s in enumerate(sequences):
            values={}
            for layer in layers:
                tensor=out.hidden_states[layer] if intermediate else out.last_hidden_state
                arr=tensor[i,keep[i].to(tensor.device)].float().cpu().numpy()
                if arr.shape!=(len(s),ENCODERS[self.name]) or not np.isfinite(arr).all():raise ValueError('invalid full-U representation')
                values[layer]=arr
            result.append({'sequence_sha256':sequence_hash(s),'positions':list(range(len(s))),'layers':values})
        return result

    def __call__(self,*,sequence,layers,truncate,training):
        if truncate is not False or training is not False:raise ValueError('frozen full-U extraction required')
        return self.extract_batch([sequence],layers,self.manifest.get('compute_precision','fp32'))[0]
