"""Decorate FedAvg strategy."""
from substrafl.strategies import FedAvg
from fedeca.utils.logging.logging_decorators import log_remote
from substrafl.strategies.schemas import FedAvgAveragedState
from substrafl.strategies.schemas import FedAvgSharedState
from typing import List
from substrafl.remote import remote


class FedAvgDecorated(FedAvg):
    """Define the decorated class.

    Parameters
    ----------
    FedAvg : _type_
        _description_

    Returns
    -------
    _type_
        _description_

    Raises
    ------
    EmptySharedStatesError
        _description_
    """
    @remote
    @log_remote
    def avg_shared_states(self, shared_states: List[FedAvgSharedState]) -> FedAvgAveragedState:
        """Decorate original method.

        Parameters
        ----------
        shared_states : List[FedAvgSharedState]
            _description_

        Returns
        -------
        FedAvgAveragedState
            _description_
        """
        return super().avg_shared_states(shared_states=shared_states, _skip=True)
