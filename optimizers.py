"""
optimizers.py
=============

Drop-in optimizers for the NES black-box attack from
labsix/limited-blackbox-attacks (Ilyas, Engstrom, Athalye, Lin, 2018,
"Black-box Adversarial Attacks with Limited Queries and Information",
arXiv:1804.08598).

The original attacks.py uses a single hard-coded update:

    prev_g = g
    l, g = get_grad(adv, args.samples_per_draw, batch_size)
    # SIMPLE MOMENTUM
    g = args.momentum * prev_g + (1.0 - args.momentum) * g
    ...
    proposed_adv = adv - is_targeted * current_lr * np.sign(g)

We factor that one line out so other classical optimizers can be
swapped in. Each optimizer takes a NumPy gradient and returns the
update direction that should replace ``g`` going into the existing
sign-line-search.

Interface
---------
    opt = make_optimizer("adam")
    ...
    l, raw_g = get_grad(...)
    g = opt.update(raw_g)
    proposed_adv = adv - is_targeted * current_lr * np.sign(g)

Note on signing
---------------
The line search in attacks.py applies ``np.sign(g)``, which throws
away the per-coordinate magnitude of the optimizer step. For
Adam/AdaGrad that means we keep their effect on *direction* (a
coordinate with a small running-average grad gets relatively
amplified) but not on absolute step size. That is the right
behaviour for an L_inf attack and matches how Madry-style PGD adapts
Adam. If you want unsigned updates, drop the np.sign and let the
optimizer's own scale do the work; you'll typically need to lower
max_lr by ~100x.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------
class Optimizer(object):
    """Stateful optimizer. Subclasses implement update(grad)."""

    name = "base"

    def update(self, grad):
        raise NotImplementedError

    def reset(self):
        """Wipe internal state (useful between attacks on different images)."""
        for k in list(self.__dict__):
            if k not in self._hyperparams():
                self.__dict__[k] = None

    def _hyperparams(self):
        # subclasses override
        return set()


# ---------------------------------------------------------------------------
# Simple momentum  (the original attacks.py update)
# ---------------------------------------------------------------------------
class SimpleMomentum(Optimizer):
    """g_t = mu * g_{t-1} + (1 - mu) * grad_t

    Identical to the existing attacks.py block when mu = args.momentum.
    """

    name = "momentum"

    def __init__(self, momentum=0.9):
        self.mu = momentum
        self.prev = None

    def update(self, grad):
        if self.prev is None:
            self.prev = np.zeros_like(grad)
        g = self.mu * self.prev + (1.0 - self.mu) * grad
        self.prev = g
        return g

    def _hyperparams(self):
        return {"mu"}


# ---------------------------------------------------------------------------
# Nesterov momentum
# ---------------------------------------------------------------------------
class Nesterov(Optimizer):
    """Nesterov accelerated gradient, simplified form.

    Strict Nesterov needs the gradient at the lookahead point
    ``theta - mu * v``. NES estimates the gradient at ``theta``
    itself, so we use the standard simplified form
    (Sutskever 2013; Bengio et al. 2012, "Advances in Optimizing
    Recurrent Networks", Eq. 7) which is what PyTorch's
    ``SGD(nesterov=True)`` and TF's ``MomentumOptimizer(use_nesterov=True)``
    implement:

        v_t   = mu * v_{t-1} + grad_t
        out_t = mu * v_t + grad_t        # the look-ahead update
    """

    name = "nesterov"

    def __init__(self, momentum=0.9):
        self.mu = momentum
        self.v = None

    def update(self, grad):
        if self.v is None:
            self.v = np.zeros_like(grad)
        self.v = self.mu * self.v + grad
        return self.mu * self.v + grad

    def _hyperparams(self):
        return {"mu"}


# ---------------------------------------------------------------------------
# AdaGrad
# ---------------------------------------------------------------------------
class AdaGrad(Optimizer):
    """AdaGrad (Duchi, Hazan, Singer, 2011).

        G_t   = G_{t-1} + grad_t^2
        out_t = grad_t / (sqrt(G_t) + eps)

    Per-coordinate inverse-RMS-of-history scaling. Effective step
    monotonically decreases, which is fine for the bounded NES attack
    but caps how long it can keep moving before plateau-LR kicks in.
    """

    name = "adagrad"

    def __init__(self, eps=1e-8):
        self.eps = eps
        self.G = None

    def update(self, grad):
        if self.G is None:
            self.G = np.zeros_like(grad)
        self.G += grad * grad
        return grad / (np.sqrt(self.G) + self.eps)

    def _hyperparams(self):
        return {"eps"}


# ---------------------------------------------------------------------------
# Adam
# ---------------------------------------------------------------------------
class Adam(Optimizer):
    """Adam (Kingma & Ba, 2014, arXiv:1412.6980).

        m_t   = b1 * m_{t-1} + (1 - b1) * grad_t
        v_t   = b2 * v_{t-1} + (1 - b2) * grad_t^2
        m_hat = m_t / (1 - b1^t)
        v_hat = v_t / (1 - b2^t)
        out_t = m_hat / (sqrt(v_hat) + eps)
    """

    name = "adam"

    def __init__(self, beta1=0.9, beta2=0.999, eps=1e-8):
        self.b1 = beta1
        self.b2 = beta2
        self.eps = eps
        self.m = None
        self.v = None
        self.t = 0

    def update(self, grad):
        if self.m is None:
            self.m = np.zeros_like(grad)
            self.v = np.zeros_like(grad)
        self.t += 1
        self.m = self.b1 * self.m + (1.0 - self.b1) * grad
        self.v = self.b2 * self.v + (1.0 - self.b2) * (grad * grad)
        m_hat = self.m / (1.0 - self.b1 ** self.t)
        v_hat = self.v / (1.0 - self.b2 ** self.t)
        return m_hat / (np.sqrt(v_hat) + self.eps)

    def _hyperparams(self):
        return {"b1", "b2", "eps"}

# ---------------------------------------------------------------------------
# SGD
# ---------------------------------------------------------------------------
class SGD(Optimizer):
    """Stochastic Gradient Descent basique. 
    Renvoie le gradient tel quel (attacks.py se charge d'appliquer le sign).
    """
    name = "sgd"

    def update(self, grad):
        return grad

# ---------------------------------------------------------------------------
# Lion (EvoLved Sign Momentum)
# ---------------------------------------------------------------------------
class Lion(Optimizer):
    """Lion (Chen et al., 2023).
    Algorithm discovered by AI at Google. Uses only signs and momentum,
    making it fast and memory-efficient.
    """
    name = "lion"

    def __init__(self, beta1=0.9, beta2=0.99):
        self.b1 = beta1
        self.b2 = beta2
        self.m = None

    def update(self, grad):
        if self.m is None:
            self.m = np.zeros_like(grad)
        
        # Compute update direction (with beta1)
        c = self.b1 * self.m + (1.0 - self.b1) * grad
        
        # Update momentum for next iteration (with beta2)
        self.m = self.b2 * self.m + (1.0 - self.b2) * grad
        
        # Lion applique directement un signe sur la direction
        return np.sign(c)

    def _hyperparams(self):
        return {"b1", "b2"}


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
_REGISTRY = {
    "momentum":  SimpleMomentum,
    "nesterov":  Nesterov,
    "adagrad":   AdaGrad,
    "adam":      Adam,
    "sgd":       SGD,
    "lion":      Lion,
}


def make_optimizer(name, **kwargs):
    """Build an optimizer by name.

    Parameters
    ----------
    name : str
        One of "momentum", "nesterov", "adagrad", "adam".
    **kwargs
        Forwarded to the constructor. Unknown kwargs are silently
        dropped so the same `args` namespace can feed any optimizer.

    Returns
    -------
    Optimizer
    """
    name = name.lower()
    if name not in _REGISTRY:
        raise ValueError(
            "unknown optimizer %r; choose from %s"
            % (name, sorted(_REGISTRY))
        )
    cls = _REGISTRY[name]
    # only forward kwargs the constructor actually wants
    valid = cls.__init__.__code__.co_varnames[1:cls.__init__.__code__.co_argcount]
    filtered = {k: v for k, v in kwargs.items() if k in valid}
    return cls(**filtered)


# ---------------------------------------------------------------------------
# Tiny self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.RandomState(0)
    grad = rng.randn(2, 3).astype(np.float32)
    for name in _REGISTRY:
        opt = make_optimizer(name, momentum=0.9, beta1=0.9, beta2=0.999, eps=1e-8)
        out = opt.update(grad)
        assert out.shape == grad.shape, name
        # call it a second time to exercise state
        out2 = opt.update(grad)
        assert out2.shape == grad.shape, name
        print("%-9s OK  first-call mean=%+.4f  second-call mean=%+.4f"
              % (name, out.mean(), out2.mean()))
