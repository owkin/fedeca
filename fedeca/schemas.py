"""Schemas used in the application."""
from enum import Enum
from typing import List

import numpy as np
import pydantic


class StrategyName(str, Enum):
    """Strategy name class."""

    FEDERATED_AVERAGING = "Federated Averaging"
    SCAFFOLD = "Scaffold"
    ONE_ORGANIZATION = "One organization"
    NEWTON_RAPHSON = "Newton Raphson"


class _Model(pydantic.BaseModel):
    """Base model configuration."""

    class Config:
        arbitrary_types_allowed = True


class WebDiscoAveragedStates(_Model):
    """Shared state sent by the aggregate_organization in the Newton Raphson strategy.

    Args
    ----
        parameters_update (numpy.ndarray): the new parameters_update sent to the clients
    """

    risk_phi: List[np.ndarray]
    risk_phi_x: List[np.ndarray]
    risk_phi_x_x: List[np.ndarray]


class WebDiscoSharedState(_Model):
    r"""WebDisco shared state class.

    Shared state returned by the train method of the algorithm for each client,
    received by the aggregate function in the Newton Raphson strategy.

    Args
    ----
        n_samples (int): number of samples of the client dataset.
        gradients (numpy.ndarray): gradients of the model parameters :math:`\\theta`.
        hessian (numpy.ndarray): second derivative of the loss function regarding
        the model parameters :math:`\\theta`.
    """

    risk_phi: List[np.ndarray]
    risk_phi_x: List[np.ndarray]
    risk_phi_x_x: List[np.ndarray]
