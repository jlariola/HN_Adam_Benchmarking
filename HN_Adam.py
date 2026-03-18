"""
HN_Adam: Hybrid and Adaptive Norming of Adam with AMSGrad — The Modified Algorithm

Implements Algorithm 2, Section 5 from:
    Reyad, M., Sarhan, A. & Arafa, M. "A modified Adam algorithm for deep
    neural network optimization." Neural Comput & Applic 35, 17095-17112 (2023).
    https://doi.org/10.1007/s00521-023-08568-z
"""

import torch
from torch.optim.optimizer import Optimizer


class HNAdam(Optimizer):
    r"""HN_Adam optimizer (Algorithm 2).

    Combines an adaptive norm technique with a hybrid switching mechanism
    between standard Adam and AMSGrad.  The letters "H" and "N" refer to the
    hybrid mechanism and the adaptive norm, respectively.

    Args:
        params (iterable): iterable of parameters to optimize or dicts
            defining parameter groups.
        lr (float): learning rate, step size :math:`\eta` (default: 1e-3).
        betas (Tuple[float, float]): coefficients :math:`(\beta_1, \beta_2)`
            for the exponential moving averages of the gradient and its
            powered absolute value (default: (0.9, 0.999)).
        eps (float): term added to the denominator for numerical stability
            :math:`\epsilon` (default: 1e-8).
        lambda_0 (float or None): initial threshold value of the norm
            :math:`\Lambda_{t_0}`.  When ``None`` it is drawn uniformly from
            [2, 4] as recommended in the paper (default: None).
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 lambda_0=None):
        # ---- validate hyper-parameters ----
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")

        # Require (Sec. 5): Λₜ₀ is randomly chosen in [2, 4]
        if lambda_0 is None:
            lambda_0 = 2.0 + 2.0 * torch.rand(1).item()  # uniform in [2, 4]
        if lambda_0 <= 0.0:
            raise ValueError(f"Invalid lambda_0 value: {lambda_0}, must be > 0")

        defaults = dict(lr=lr, betas=betas, eps=eps, lambda_0=lambda_0)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        r"""Perform a single optimisation step (Algorithm 2).

        Args:
            closure (callable, optional): a closure that re-evaluates the model
                and returns the loss.

        Returns:
            loss value (if *closure* was supplied).
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # ---- Algorithm 2: For all t = 1, …, T do ----
        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            lambda_0 = group['lambda_0']

            for p in group['params']:
                if p.grad is None:
                    continue

                # Step 5:  gₜ ← ∇ loss w.r.t. θₜ₋₁   //  f'(θₜ₋₁)
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError(
                        "HN_Adam does not support sparse gradients")

                state = self.state[p]

                # ---- Initialize: m₀=0, v₀=0, amsgrad=False, v̂₍₀₎=0 ----
                if len(state) == 0:
                    state['step'] = 0
                    state['m'] = torch.zeros_like(
                        p, memory_format=torch.preserve_format)
                    state['v'] = torch.zeros_like(
                        p, memory_format=torch.preserve_format)
                    state['v_hat'] = torch.zeros_like(
                        p, memory_format=torch.preserve_format)

                m = state['m']
                v = state['v']
                v_hat = state['v_hat']
                state['step'] += 1

                # Save mₜ₋₁ before the in-place update
                m_prev = m.clone()

                # Step 6:  mₜ ← β₁·mₜ₋₁ + (1 − β₁)·gₜ        // moving average
                m.mul_(beta1).add_(grad, alpha=1.0 - beta1)

                # Step 7:  m_max ← Max(mₜ₋₁ , |gₜ|)
                abs_grad = grad.abs()
                m_max = torch.max(m_prev, abs_grad)
                # Clamp to avoid division by zero when both are zero
                m_max = m_max.clamp(min=eps)

                # Step 8:  Λ(t) ← Λₜ₀ − mₜ₋₁ / m_max            (Eq. 20)
                lambda_t = lambda_0 - m_prev / m_max
                # Clamp to a safe range to prevent extreme exponents
                lambda_t = lambda_t.clamp(min=1.0, max=10.0)

                # Step 9:  vₜ ← β₂·vₜ₋₁ + (1 − β₂)·|gₜ|^{Λ(t)}
                abs_grad_powered = torch.pow(abs_grad + eps, lambda_t)
                v.mul_(beta2).add_(abs_grad_powered, alpha=1.0 - beta2)

                # Precompute 1 / Λ(t) for the denominator exponent
                inv_lambda = 1.0 / lambda_t

                # Step 10: If Λ(t) < 2  →  switch between Adam and AMSGrad
                amsgrad_mask = lambda_t < 2.0

                # Step 12: v̂₍ₜ₎ ← Max(v̂₍ₜ₋₁₎, vₜ)  (AMSGrad branch only)
                v_hat_candidate = torch.max(v_hat, v)
                v_hat.copy_(torch.where(amsgrad_mask, v_hat_candidate, v_hat))

                # Step 13 (AMSGrad branch):
                #   θₜ ← θₜ₋₁ − η · mₜ / ( v̂₍ₜ₎^{1/Λ(t)} + ε )
                denom_amsgrad = torch.pow(
                    v_hat_candidate.clamp(min=0.0), inv_lambda) + eps

                # Step 16 (Adam branch):
                #   θₜ ← θₜ₋₁ − η · mₜ / ( vₜ^{1/Λ(t)} + ε )
                denom_adam = torch.pow(
                    v.clamp(min=0.0), inv_lambda) + eps

                # Element-wise selection of the correct denominator
                denom = torch.where(amsgrad_mask, denom_amsgrad, denom_adam)

                # Apply parameter update:
                #   θₜ ← θₜ₋₁ − η · mₜ / denom
                p.addcdiv_(m, denom, value=-lr)

        # Step 18: return final parameter θ_T
        return loss


# ---------------------------------------------------------------------------
# Minimal verification
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(42)

    # --- Test 1: simple quadratic f(x) = Σ xᵢ² ---
    print("=" * 60)
    print("Test 1: Minimise f(x) = x₁² + x₂² + x₃²")
    print("=" * 60)
    x = torch.tensor([5.0, -3.0, 7.0], requires_grad=True)
    optimizer = HNAdam([x], lr=0.01, lambda_0=3.0)

    for step in range(1, 1001):
        optimizer.zero_grad()
        loss = (x ** 2).sum()
        loss.backward()
        optimizer.step()
        if step % 200 == 0:
            print(f"  step {step:4d}  |  loss = {loss.item():.6f}  |  x = {x.data.tolist()}")

    final_loss = (x ** 2).sum().item()
    print(f"  Final loss: {final_loss:.8f}")
    assert final_loss < 1e-2, f"Did not converge (loss={final_loss})"
    print("  PASSED\n")

    # --- Test 2: small neural network on synthetic data ---
    print("=" * 60)
    print("Test 2: Two-layer MLP on random binary classification")
    print("=" * 60)
    torch.manual_seed(0)
    X = torch.randn(256, 10)
    Y = (X[:, 0] + X[:, 1] > 0).float().unsqueeze(1)

    model = torch.nn.Sequential(
        torch.nn.Linear(10, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 1),
        torch.nn.Sigmoid(),
    )
    criterion = torch.nn.BCELoss()
    optimizer = HNAdam(model.parameters(), lr=1e-3, lambda_0=3.0)

    for epoch in range(1, 201):
        optimizer.zero_grad()
        out = model(X)
        loss = criterion(out, Y)
        loss.backward()
        optimizer.step()
        if epoch % 50 == 0:
            acc = ((out > 0.5).float() == Y).float().mean().item() * 100
            print(f"  epoch {epoch:3d}  |  loss = {loss.item():.4f}  |  acc = {acc:.1f}%")

    final_acc = ((model(X) > 0.5).float() == Y).float().mean().item() * 100
    print(f"  Final accuracy: {final_acc:.1f}%")
    assert final_acc > 80, f"Accuracy too low ({final_acc:.1f}%)"
    print("  PASSED\n")

    # --- Test 3: edge-case — zero gradients ---
    print("=" * 60)
    print("Test 3: Edge case — zero gradients do not cause errors")
    print("=" * 60)
    w = torch.zeros(4, requires_grad=True)
    opt = HNAdam([w], lr=1e-3, lambda_0=2.5)
    opt.zero_grad()
    loss = (w * 0).sum()  # gradient is zero everywhere
    loss.backward()
    opt.step()  # must not raise
    print("  No error on zero gradient.")
    print("  PASSED\n")

    print("All tests passed.")
