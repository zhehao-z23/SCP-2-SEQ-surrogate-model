"""Stable float32 distribution objectives; CDF uses K-1 ordered cutpoints."""
import torch
from torch.nn import functional as F


def distribution_loss(logits,target,name="h2",cdf_weight=0.):
    if logits.ndim!=2 or logits.shape!=target.shape or logits.shape[1]<2: raise ValueError("target shape")
    q=target.float()
    if not torch.isfinite(logits).all() or not torch.isfinite(q).all() or (q<0).any() or not torch.allclose(q.sum(1),torch.ones_like(q[:,0]),atol=1e-6,rtol=0): raise ValueError("invalid distribution")
    logp=F.log_softmax(logits.float(),dim=1)
    h2=.5*(torch.exp(.5*logp)-torch.sqrt(q)).square().sum(1)
    if name=="h2":return h2
    if name=="ce":return -(q*logp).sum(1)
    if name=="h2_cdf":
        if not 0<cdf_weight< float('inf'):raise ValueError("explicit positive CDF weight required")
        return h2+cdf_weight*(logp.exp().cumsum(1)[:,:-1]-q.cumsum(1)[:,:-1]).square().mean(1)
    raise ValueError("out-of-scope loss")


def objective(output,batch,config):
    target_key="fragment_target" if batch.get("task","primary")=="fragment_auxiliary" else "target"
    logit_key="fragment_logits" if target_key=="fragment_target" else "logits"
    loss=distribution_loss(output[logit_key],batch[target_key],config["loss"],config.get("cdf_weight",0))
    if target_key=="fragment_target":
        weight=config.get("aux_weight")
        if weight is None or not 0<weight<float('inf'):raise ValueError("auxiliary weight must be calibrated")
        loss=loss*weight
    if "stats_target" in batch:
        weight=config.get("stats_weight")
        if weight is None or not 0<weight<float('inf'):raise ValueError("raw-stat weight must be calibrated")
        truth=batch["stats_target"].float();valid=batch["stats_mask"].bool()
        if truth.shape!=output["stats"].shape or valid.shape!=truth.shape or not torch.isfinite(truth[valid]).all():raise ValueError("stats contract")
        err=(output["stats"].float()-torch.where(valid,truth,torch.zeros_like(truth))).square()
        loss=loss+weight*(err*valid).sum(1)/valid.sum(1).clamp_min(1)
    return loss
