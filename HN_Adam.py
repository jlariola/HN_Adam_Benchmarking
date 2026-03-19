"""
HN_Adam: Hybrid and Adaptive Norming of Adam with AMSGrad — The Modified Algorithm

Implements Algorithm 2, Section 5 from:
    Reyad, M., Sarhan, A. & Arafa, M. "A modified Adam algorithm for deep
    neural network optimization." Neural Comput & Applic 35, 17095-17112 (2023).
    https://doi.org/10.1007/s00521-023-08568-z
"""

import torch
from torch.optim.optimizer import Optimizer


class HN_Adam(Optimizer):
    r"""HN_Adam optimizer (Algorithm 2, Section 5).

    Args:
        params (iterable): Parameters to optimize.
        lr (float): Step size :math:`\eta`.
        betas (Tuple[float, float]): Decay rates :math:`(\beta_1, \beta_2)`.
        eps (float): Numerical stability term :math:`\epsilon`.
        lambda_0 (float or None): Initial :math:`\Lambda_{t_0}`.
            If None, sampled uniformly from [2, 4].
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 lambda_0=None):
        if params is None:
            raise ValueError("Parameter iterable cannot be None")
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")

        # Require (Section 5): Λ_t0 is randomly chosen in [2, 4]
        if lambda_0 is None:
            lambda_0 = 2.0 + 2.0 * torch.rand(1).item()
        if lambda_0 <= 0.0:
            raise ValueError(f"Invalid lambda_0 value: {lambda_0}, must be > 0")

        defaults = dict(lr=lr, betas=betas, eps=eps, lambda_0=lambda_0)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        r"""Perform a single optimization step using Algorithm 2."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Step 3: For all t = 1, ..., T do
        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            lambda_0 = group['lambda_0']

            for p in group['params']:
                if p.grad is None:
                    continue

                # Step 5: g_t <- gradient at theta_(t-1)
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("HN_Adam does not support sparse gradients")

                state = self.state[p]

                # Initialize: m_0 = 0, v_0 = 0, amsgrad = False, v_hat(0) = 0
                if len(state) == 0:
                    state['step'] = 0
                    state['m'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['v'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['v_hat'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['amsgrad'] = False

                m = state['m']
                v = state['v']
                v_hat = state['v_hat']
                state['step'] += 1

                # Keep m_(t-1) for Steps 7 and 8
                m_prev = m.clone()

                # Step 6: m_t <- beta1 * m_(t-1) + (1 - beta1) * g_t
                m.mul_(beta1).add_(grad, alpha=1.0 - beta1)

                # Step 7: m_max <- Max(m_(t-1), |g_t|)
                abs_grad = grad.abs()
                m_max = torch.max(m_prev, abs_grad)
                safe_m_max = torch.where(
                    m_max == 0,
                    torch.ones_like(m_max),
                    m_max,
                )

                # Step 8: Lambda(t) <- Lambda_t0 - m_(t-1) / m_max
                lambda_t = lambda_0 - (m_prev / safe_m_max)

                # Step 9: v_t <- beta2 * v_(t-1) + (1 - beta2) * (|g_t| ^ Lambda(t))
                abs_grad_powered = torch.pow(abs_grad.clamp(min=eps), lambda_t)
                v.mul_(beta2).add_(abs_grad_powered, alpha=1.0 - beta2)

                # Compute 1 / Lambda(t), guarded for zero values
                safe_lambda_t = torch.where(
                    lambda_t == 0,
                    torch.full_like(lambda_t, eps),
                    lambda_t,
                )
                inv_lambda = 1.0 / safe_lambda_t

                # Step 10: If Lambda(t) < 2
                amsgrad_mask = lambda_t < 2.0
                state['amsgrad'] = bool(torch.any(amsgrad_mask).item())

                # Step 11: amsgrad <- True
                # Step 15: amsgrad <- False

                # Step 12: v_hat(t) <- Max(v_hat(t-1), v_t)
                v_hat_candidate = torch.max(v_hat, v)
                v_hat.copy_(torch.where(amsgrad_mask, v_hat_candidate, v_hat))

                # Step 13: theta_t <- theta_(t-1) - eta * m_t / ((v_hat(t)^(1/Lambda(t))) + eps)
                denom_amsgrad = torch.pow(v_hat.clamp(min=0.0), inv_lambda) + eps

                # Step 16: theta_t <- theta_(t-1) - eta * m_t / ((v_t^(1/Lambda(t))) + eps)
                denom_adam = torch.pow(v.clamp(min=0.0), inv_lambda) + eps

                # Step 14/17: else branch and end if
                denom = torch.where(amsgrad_mask, denom_amsgrad, denom_adam)

                # Apply parameter update to theta
                p.addcdiv_(m, denom, value=-lr)

        # Step 18: return final parameter theta_T
        return loss
