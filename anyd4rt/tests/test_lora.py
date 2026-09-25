import torch
import torch.nn as nn

from anyd4rt.anyview_rl import lora


class _Block(nn.Module):
    def __init__(self, d=16):
        super().__init__()
        self.q_proj = nn.Linear(d, d, bias=False)
        self.layer1 = nn.Linear(d, 2 * d, bias=False)
        self.layer2 = nn.Linear(2 * d, d, bias=False)
        self.other = nn.Linear(d, d, bias=False)  # not a target

    def forward(self, x):
        return x + self.layer2(torch.relu(self.layer1(self.q_proj(x)))) + self.other(x)


class _Net(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.blocks = nn.ModuleList([_Block(), _Block()])
        self.head = nn.Linear(16, 16, bias=False)  # outside blocks: not wrapped with blocks_only

    def forward(self, x):
        for b in self.blocks:
            x = b(x)
        return self.head(x)


def test_inject_targets_freeze_and_identity_at_B0():
    net, ref = _Net(), _Net()
    x = torch.randn(4, 16)
    wrapped = lora.inject(net, rank=4, alpha=4.0)
    assert len(wrapped) == 6  # 3 targets x 2 blocks; `other` and `head` untouched
    assert all(not p.requires_grad for n, p in net.named_parameters() if "lora_" not in n)
    assert all(p.requires_grad for p in lora.lora_parameters(net))
    assert torch.equal(net(x), ref(x))


def test_init_is_reproducible_and_switch_works():
    a, b = _Net(), _Net()
    wa, wb = lora.inject(a, rank=4, seed=3), lora.inject(b, rank=4, seed=3)
    assert all(torch.equal(m.lora_A, n.lora_A) for m, n in zip(wa, wb))
    x = torch.randn(4, 16)
    y0 = a(x)
    with torch.no_grad():
        for m in wa:
            m.lora_B.normal_()
    assert not torch.allclose(a(x), y0)
    lora.set_enabled(a, False)
    assert torch.equal(a(x), y0)


def test_gradients_through_block_checkpoint_match():
    x = torch.randn(4, 16)
    grads = []
    for ckpt in (False, True):
        net = _Net()
        wrapped = lora.inject(net, rank=4, seed=1)
        with torch.no_grad():
            for m in wrapped:
                m.lora_B.normal_(std=0.1)
        if ckpt:
            lora.enable_block_checkpoint(net)
        net(x).pow(2).mean().backward()
        grads.append(torch.cat([p.grad.flatten() for p in lora.lora_parameters(net)]))
    assert torch.allclose(grads[0], grads[1], atol=1e-6)
