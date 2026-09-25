"""Hand-written LoRA for the AnyView DiT (README §4.4: peft does not fit the custom DiT).

Wraps bias-free nn.Linear layers: y = W x + (alpha / r) * B(A x). B starts at zero, so an injected model
is numerically the released model until B is trained. The base weights stay frozen (bf16); A and B are
kept in float32 and cast to the input dtype in forward. `set_enabled(model, False)` gives the reference
policy (LoRA off) for the KL term without a second copy of the model.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

DEFAULT_TARGETS = ("q_proj", "k_proj", "v_proj", "output_proj", "layer1", "layer2")


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float, generator: torch.Generator | None = None):
        super().__init__()
        assert base.bias is None, "LoRA wrapper assumes bias-free linears (true for the AnyView DiT)"
        self.base = base
        self.rank, self.scale, self.enabled = rank, alpha / rank, True
        dev = base.weight.device
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features, device=dev, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank, device=dev, dtype=torch.float32))
        # kaiming_uniform(a=sqrt(5)) bound, drawn from an explicit generator so the init is reproducible
        bound = 1.0 / math.sqrt(base.in_features)
        with torch.no_grad():
            a = torch.rand(self.lora_A.shape, generator=generator, dtype=torch.float32) * 2 * bound - bound
            self.lora_A.copy_(a.to(dev))

    def forward(self, x):
        y = self.base(x)
        if not self.enabled:
            return y
        return y + (x @ self.lora_A.to(x.dtype).t() @ self.lora_B.to(x.dtype).t()) * self.scale


def inject(model: nn.Module, rank: int = 16, alpha: float = 16.0, targets=DEFAULT_TARGETS, blocks_only: bool = True, seed: int = 0):
    """Replace target linears in place, freeze everything else. Returns the list of LoRA modules.
    A is initialized from a CPU generator seeded with `seed` (in module order), so runs are reproducible."""
    gen = torch.Generator().manual_seed(seed)
    for p in model.parameters():
        p.requires_grad_(False)
    wrapped = []
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            full = f"{name}.{child_name}" if name else child_name
            if child_name in targets and isinstance(child, nn.Linear) and (not blocks_only or "blocks." in full):
                lora = LoRALinear(child, rank, alpha, generator=gen)
                setattr(module, child_name, lora)
                wrapped.append(lora)
    return wrapped


def lora_parameters(model: nn.Module):
    return [p for n, p in model.named_parameters() if "lora_" in n]


def set_enabled(model: nn.Module, enabled: bool):
    for m in model.modules():
        if isinstance(m, LoRALinear):
            m.enabled = enabled


def lora_state_dict(model: nn.Module):
    return {n: p.detach().cpu() for n, p in model.named_parameters() if "lora_" in n}


def enable_block_checkpoint(dit: nn.Module):
    """Full activation recomputation per DiT block (only block inputs are kept for backward).

    The released `sac_mode="block_wise"` passes context_fn=None to checkpoint_wrapper, which fails under
    torch 2.7 (TypeError: 'NoneType' object is not callable). This wraps the blocks with the default
    (full recompute) context instead, leaving AnyView's code untouched. Build the DiT with sac_mode="none".
    """
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper

    for block_id, block in list(dit.blocks.named_children()):
        dit.blocks.register_module(block_id, checkpoint_wrapper(block, preserve_rng_state=False))
    return dit
