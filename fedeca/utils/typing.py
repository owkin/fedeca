"""Module defining type alias."""
from collections.abc import Callable, Sequence
from typing import Any, Optional, Union

from numpy.random import BitGenerator, Generator, SeedSequence

# following the typing of numpy.random
_SeedType = Optional[
    Union[
        int,
        Sequence[int],
        BitGenerator,
        SeedSequence,
        Generator,
    ]
]

# function defining for each patient the probability of being in the treatment group
_FuncPropensityType = Callable[[Sequence[Any]], float]

# function defining for each patient the treatment effect as hazard ratio.
_FuncCateType = Callable[[Sequence[Any]], float]
