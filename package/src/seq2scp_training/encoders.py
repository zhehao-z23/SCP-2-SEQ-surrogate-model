"""Offline frozen-pLM boundary. No downloading, GPU access or SDK import here.

A target-verified backend returns selected residue matrices and residue positions;
this adapter validates exact full-U coverage. SDK-specific loading is deliberately
external until installed checkpoint/API identities are available.
"""
import numpy as np
from .contracts import digest, sequence_hash

ENCODERS={"esmc_300m":960,"esmc_600m":1152,"esm2_t33_650M_UR50D":1280}


def cache_identity(*, model, checkpoint_sha256, implementation_version, layer, normalization, sequence_sha256, dtype, pooling):
    if model not in ENCODERS or not all([checkpoint_sha256,implementation_version,normalization,sequence_sha256,dtype,pooling]):
        raise ValueError("incomplete encoder identity")
    return digest(locals())


def check_matrix(matrix, positions, sequence, width):
    a=np.asarray(matrix)
    if a.shape!=(len(sequence),width) or list(positions)!=list(range(len(sequence))):
        raise ValueError("truncation, BOS/EOS contamination, residue order or dimension mismatch")
    if not np.isfinite(a).all(): raise ValueError("nonfinite embedding")
    return a


class FrozenEncoderAdapter:
    def __init__(self, model, backend, checkpoint_sha256, implementation_version):
        if model not in ENCODERS: raise ValueError("out-of-scope encoder")
        self.model=model;self.backend=backend;self.checkpoint_sha256=checkpoint_sha256;self.implementation_version=implementation_version

    def extract(self, sequence, layers):
        if not sequence or not layers or len(set(layers))!=len(layers): raise ValueError("invalid extraction request")
        result=self.backend(sequence=sequence,layers=list(layers),truncate=False,training=False)
        if result.get("sequence_sha256")!=sequence_hash(sequence) or set(result["layers"])!=set(layers): raise ValueError("backend identity mismatch")
        return {layer:check_matrix(result["layers"][layer],result["positions"],sequence,ENCODERS[self.model]) for layer in layers}
