import torch

class MeZO:
    """
    MeZO: Zeroth-Order Optimization for Memory-Efficient Fine-Tuning.
    Estimates gradient using two loss evaluations with random perturbation.
    """
    def __init__(self, model, lr=1e-5, eps=1e-3):
        self.model = model
        self.lr = lr
        self.eps = eps
        self.params = [p for p in model.parameters() if p.requires_grad]

    def step(self, batch):
        # 1. Generate random seed for perturbation
        seed = torch.randint(0, 1000000, (1,)).item()
        
        # 2. Perturb: theta + eps * z
        self._perturb_params(seed, scaling=self.eps)
        loss_pos = self._get_loss(batch)
        
        # 3. Perturb: theta - eps * z (or just theta)
        self._perturb_params(seed, scaling=-2*self.eps) # from (theta+eps*z) to (theta-eps*z)
        loss_neg = self._get_loss(batch)
        
        # 4. Estimate gradient: (L+ - L-) / (2 * eps)
        grad_estimate = (loss_pos - loss_neg) / (2 * self.eps)
        
        # 5. Restore original parameters and update: theta = theta - lr * grad_estimate * z
        self._perturb_params(seed, scaling=self.eps) # restore to theta
        self._update_params(seed, grad_estimate)
        
        return (loss_pos + loss_neg) / 2

    @torch.no_grad()
    def _perturb_params(self, seed, scaling):
        torch.manual_seed(seed)
        for p in self.params:
            z = torch.randn_like(p)
            p.data.add_(z, alpha=scaling)

    @torch.no_grad()
    def _update_params(self, seed, grad_estimate):
        torch.manual_seed(seed)
        for p in self.params:
            z = torch.randn_like(p)
            p.data.add_(z, alpha=-self.lr * grad_estimate)

    def _get_loss(self, batch):
        outputs = self.model(input_ids=batch, labels=batch)
        return outputs.loss.item()
