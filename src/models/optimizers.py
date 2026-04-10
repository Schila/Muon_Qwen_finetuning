import torch
import math
from torch.optim import AdamW

@torch.compile
def zeropower_via_newtonschulz5(G, steps):
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(0) > G.size(1):
        X = X.T
    X = X / (X.norm() + 1e-7)
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X

class Muon(torch.optim.Optimizer):
    def __init__(self, lr=1e-3, wd=0.1, muon_params=None, momentum=0.95, 
                 nesterov=True, ns_steps=5, adamw_params=None, adamw_lr=None,
                 adamw_betas=(0.9, 0.95), adamw_eps=1e-8):
        adamw_lr = adamw_lr if adamw_lr is not None else lr
        defaults = dict(lr=lr, wd=wd, momentum=momentum, nesterov=nesterov, 
                        ns_steps=ns_steps, adamw_betas=adamw_betas, adamw_eps=adamw_eps)
        params = list(muon_params) if muon_params else []
        adamw_params = list(adamw_params) if adamw_params is not None else []
        
        param_groups = [
            {"params": params, "lr": lr},
            {"params": adamw_params, "lr": adamw_lr}
        ]
        super().__init__(param_groups, defaults)
        
        for p in params:
            self.state[p]["use_muon"] = True
        for p in adamw_params:
            self.state[p]["use_muon"] = False

    def adjust_lr_for_muon(self, lr, param_shape):
        A, B = param_shape[:2]
        return lr * (0.2 * math.sqrt(max(A, B)))

    def step(self, closure=None):
        loss = closure() if closure else None
        for group in self.param_groups:
            lr, wd, momentum = group["lr"], group["wd"], group["momentum"]
            
            # Muon Branch
            muon_p = [p for p in group["params"] if self.state[p].get("use_muon", False)]
            for p in muon_p:
                if p.grad is None:
                    continue
                g = p.grad.view(p.size(0), -1) if p.grad.ndim > 2 else p.grad
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                u = zeropower_via_newtonschulz5(g.add(buf, alpha=momentum) if group["nesterov"] else buf, steps=group["ns_steps"])
                p.data.mul_(1 - lr * wd)
                p.data.add_(u, alpha=-self.adjust_lr_for_muon(lr, p.shape))

            # AdamW Branch
            adamw_p = [p for p in group["params"] if not self.state[p].get("use_muon", False)]
            b1, b2 = group["adamw_betas"]
            eps = group["adamw_eps"]
            for p in adamw_p:
                if p.grad is None:
                    continue
                state = self.state[p]
                if "step" not in state:
                    state["step"], state["m1"], state["m2"] = 0, torch.zeros_like(p.grad), torch.zeros_like(p.grad)
                state["step"] += 1
                state["m1"].lerp_(p.grad, 1 - b1)
                state["m2"].lerp_(p.grad.square(), 1 - b2)
                m1_hat = state["m1"] / (1 - b1**state["step"])
                m2_hat = state["m2"] / (1 - b2**state["step"])
                p.data.mul_(1 - lr * wd)
                p.data.add_(m1_hat / (m2_hat.sqrt() + eps), alpha=-lr)
        return loss

def get_optimizer(model, args):
    if args.optimizer == "adamw":
        return AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    
    elif args.optimizer == "muon":
        # Стандартный Muon: 2D матрицы (кроме embed/head) -> Muon, остальные -> AdamW
        muon_params, adamw_params = [], []
        for name, p in model.named_parameters():
            if p.ndim >= 2 and "embed" not in name and "head" not in name:
                muon_params.append(p)
            else:
                adamw_params.append(p)
        return Muon(lr=args.lr, wd=args.wd, muon_params=muon_params, adamw_params=adamw_params, adamw_lr=args.adamw_lr)

    elif args.optimizer == "hybrid":
        # Разделение модели на две части (по слоям)
        layers = model.model.layers
        mid_point = len(layers) // 2
        muon_params, adamw_params = [], []
        for name, p in model.named_parameters():
            # Пример: первая половина слоев — Muon, вторая — AdamW
            is_first_half = False
            for i in range(mid_point):
                if f"layers.{i}." in name:
                    is_first_half = True
            
            if is_first_half and p.ndim >= 2 and "embed" not in name:
                muon_params.append(p)
            else:
                adamw_params.append(p)
        return Muon(lr=args.lr, wd=args.wd, muon_params=muon_params, adamw_params=adamw_params, adamw_lr=args.adamw_lr)
    
    return None
