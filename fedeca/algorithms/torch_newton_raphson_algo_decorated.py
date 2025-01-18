"""Decorate TorchNewtonRahsonAlgorithms"""
from typing import Any

from typing import Optional

from substrafl.remote import remote_data
from substrafl.strategies.schemas import NewtonRaphsonAveragedStates
from substrafl.strategies.schemas import NewtonRaphsonSharedState

from substrafl.algorithms.pytorch import TorchNewtonRaphsonAlgo
from fedeca.utils.logging.logging_decorators import log_remote_data


class TorchNewtonRaphsonAlgoDecorated(TorchNewtonRaphsonAlgo):
    """_summary_

    Parameters
    ----------
    TorchNewtonRaphsonAlgo : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    @remote_data
    @log_remote_data
    def train(
        self,
        data_from_opener: Any,
        # Set shared_state to None per default for clarity reason as
        # the decorator will do it if the arg shared_state is not passed.
        shared_state: Optional[NewtonRaphsonAveragedStates] = None,
    ) -> NewtonRaphsonSharedState:
        """_summary_

        Parameters
        ----------
        data_from_opener : Any
            _description_
        shared_state : Optional[NewtonRaphsonAveragedStates], optional
            _description_, by default None

        Returns
        -------
        NewtonRaphsonSharedState
            _description_
        """
        return super().train(data_from_opener=data_from_opener, shared_state=shared_state, _skip=True)
