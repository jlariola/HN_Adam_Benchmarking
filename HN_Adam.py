"""
HN_Adam: Hybrid and Adaptive Norming of Adam with AMSGrad — The Modified Algorithm

Implements Algorithm 2, Section 5 from:
    Reyad, M., Sarhan, A. & Arafa, M. "A modified Adam algorithm for deep
    neural network optimization." Neural Comput & Applic 35, 17095-17112 (2023).
    https://doi.org/10.1007/s00521-023-08568-z
"""

import random

import tensorflow as tf
from keras.optimizers import Optimizer


class HN_Adam(Optimizer):
    """HN_Adam optimizer (Algorithm 2, Section 5)."""

    def __init__(
        self,
        learning_rate=1e-3,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-8,
        lambda_0=None,
        name="HN_Adam",
        **kwargs,
    ):
        if learning_rate is None:
            raise ValueError("learning_rate cannot be None")
        if learning_rate < 0.0:
            raise ValueError(f"Invalid learning rate: {learning_rate}")
        if beta_1 is None or beta_2 is None:
            raise ValueError("beta_1 and beta_2 cannot be None")
        if not 0.0 <= beta_1 < 1.0:
            raise ValueError(f"Invalid beta_1 value: {beta_1}")
        if not 0.0 <= beta_2 < 1.0:
            raise ValueError(f"Invalid beta_2 value: {beta_2}")
        if epsilon is None or epsilon <= 0.0:
            raise ValueError(f"Invalid epsilon value: {epsilon}")

        # Step 2: Randomly choose Λ_t0 from [2, 4]
        if lambda_0 is None:
            lambda_0 = random.uniform(2.0, 4.0)
        if lambda_0 <= 0.0:
            raise ValueError(f"Invalid lambda_0 value: {lambda_0}, must be > 0")

        super().__init__(learning_rate=learning_rate, name=name, **kwargs)
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.lambda_0 = float(lambda_0)

        self._m = []
        self._v = []
        self._v_hat = []

    def build(self, var_list):
        super().build(var_list)
        if self.built:
            return

        # Step 1: m_0 <- 0, v_0 <- 0, v_hat(0) <- 0
        self._m = [self.add_variable_from_reference(var, "m") for var in var_list]
        self._v = [self.add_variable_from_reference(var, "v") for var in var_list]
        self._v_hat = [self.add_variable_from_reference(var, "v_hat") for var in var_list]

    def update_step(self, gradient, variable, learning_rate=None):
        if gradient is None or variable is None:
            return
        if isinstance(gradient, tf.IndexedSlices):
            raise ValueError("HN_Adam does not support sparse gradients")

        lr = learning_rate if learning_rate is not None else self.learning_rate
        grad = tf.cast(gradient, variable.dtype)
        lr = tf.cast(lr, variable.dtype)
        beta_1 = tf.cast(self.beta_1, variable.dtype)
        beta_2 = tf.cast(self.beta_2, variable.dtype)
        epsilon = tf.cast(self.epsilon, variable.dtype)
        lambda_0 = tf.cast(self.lambda_0, variable.dtype)

        index = self._get_variable_index(variable)
        m = self._m[index]
        v = self._v[index]
        v_hat = self._v_hat[index]

        # Step 5: g_t <- gradient at theta_(t-1)
        abs_grad = tf.abs(grad)

        # Keep m_(t-1) for Steps 7 and 8
        m_prev = tf.identity(m)

        # Step 6: m_t <- beta1 * m_(t-1) + (1 - beta1) * g_t
        m.assign(beta_1 * m + (1.0 - beta_1) * grad)

        # Step 7: m_max <- Max(m_(t-1), |g_t|)
        m_max = tf.maximum(m_prev, abs_grad)
        safe_m_max = tf.where(
            tf.equal(m_max, 0.0),
            tf.ones_like(m_max),
            m_max,
        )

        # Step 8: Lambda(t) <- Lambda_t0 - m_(t-1) / m_max
        lambda_t = lambda_0 - (m_prev / safe_m_max)

        # Step 9: v_t <- beta2 * v_(t-1) + (1 - beta2) * (|g_t| ^ Lambda(t))
        abs_grad_powered = tf.pow(tf.maximum(abs_grad, epsilon), lambda_t)
        v.assign(beta_2 * v + (1.0 - beta_2) * abs_grad_powered)

        # Guard division by zero for 1 / Lambda(t)
        safe_lambda_t = tf.where(
            tf.equal(lambda_t, 0.0),
            tf.fill(tf.shape(lambda_t), epsilon),
            lambda_t,
        )
        inv_lambda_t = 1.0 / safe_lambda_t

        # Step 10: if Lambda(t) < 2 then
        amsgrad_mask = lambda_t < 2.0

        # Step 12: v_hat(t) <- Max(v_hat(t-1), v_t)
        v_hat_candidate = tf.maximum(v_hat, v)
        v_hat.assign(tf.where(amsgrad_mask, v_hat_candidate, v_hat))

        # Step 13: theta_t <- theta_(t-1) - eta * m_t / ((v_hat(t)^(1/Lambda(t))) + epsilon)
        denom_amsgrad = tf.pow(tf.maximum(v_hat, 0.0), inv_lambda_t) + epsilon

        # Step 16: theta_t <- theta_(t-1) - eta * m_t / ((v_t^(1/Lambda(t))) + epsilon)
        denom_adam = tf.pow(tf.maximum(v, 0.0), inv_lambda_t) + epsilon

        # Step 14/17: else branch and end if
        denom = tf.where(amsgrad_mask, denom_amsgrad, denom_adam)

        # Step 18: update theta
        variable.assign_sub(lr * m / denom)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "beta_1": self.beta_1,
                "beta_2": self.beta_2,
                "epsilon": self.epsilon,
                "lambda_0": self.lambda_0,
            }
        )
        return config
