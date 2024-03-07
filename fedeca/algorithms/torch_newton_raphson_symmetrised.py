"""A symmetrized more stable version of NR."""
from typing import Tuple

import torch
from substrafl.algorithms.pytorch.torch_newton_raphson_algo import (
    TorchNewtonRaphsonAlgo,
)


class TorchSNewtonRaphsonAlgo(TorchNewtonRaphsonAlgo):
    """Small wrapper to be fixed in substrafl that improved stability`.

    S stands for Symmetrised.
    """

    def _compute_gradients_and_hessian(
        self, loss: torch.Tensor
    ) -> Tuple[torch.Tensor]:  # noqa: E501
        """Numerically stable computation of gradients and hessian."""
        gradients, hessian = super()._compute_gradients_and_hessian(loss)
        hessian = 0.5 * (hessian + hessian.T)
        return gradients, hessian
