"""Decorate NewtonRaphson strategy."""
from typing import List

from substrafl.strategies import NewtonRaphson
from fedeca.utils.logging.logging_decorators import log_remote

from substrafl.remote import remote
from substrafl.strategies.schemas import NewtonRaphsonAveragedStates
from substrafl.strategies.schemas import NewtonRaphsonSharedState


class NewtonRaphsonDecorated(NewtonRaphson):
    """Decorate NewtonRaphson strategy.

    Parameters
    ----------
    NewtonRaphson : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    @remote
    @log_remote
    def compute_averaged_states(
        self,
        shared_states: List[NewtonRaphsonSharedState],
    ) -> NewtonRaphsonAveragedStates:
        """Decorate original method.

        Parameters
        ----------
        shared_states : List[NewtonRaphsonSharedState]
            _description_

        Returns
        -------
        NewtonRaphsonAveragedStates
            _description_
        """
        return super().compute_averaged_states(shared_states=shared_states, _skip=True)
