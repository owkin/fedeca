from typing import NamedTuple
from typing import Union


class Item(NamedTuple):
    """
    Represents an item with a type, shape, and optional description.

    Attributes
    ----------
    type_str : str
        The type of the item as a string.
    shape : Optional[tuple[int, ...]], optional
        The shape of the item, by default None.
    description : Optional[str], optional
        A description of the item, by default None.
    """

    type_str: str
    shape: Union[tuple[int, ...], None] = None
    description: Union[str, None] = None


class SharedState(NamedTuple):
    """
    Represents a shared state with items, origin, destination.

    Attributes
    ----------
    items : dict[str, Item]
        A dictionary of items in the shared state.
    origin : Optional[int], optional
        The origin ID of the shared state, by default None.
    destination : Optional[int], optional
        The destination ID of the shared state, by default None.
    shared_with_aggregator : Optional[bool], optional
        A flag indicating if the state is shared with an aggregator, by default None.
    """

    items: dict[str, Item]
    origin: Union[int, None] = None
    destination: Union[int, None] = None
    shared_with_aggregator: Union[bool, None] = None


class IterationBlock(NamedTuple):
    """
    Represents an iteration block with the number of iterations.

    Attributes
    ----------
    num_iterations : int
        The number of iterations in the block.
    first_remote : int
        The first remote ID in the block.
    last_remote : int
        The last remote ID in the block.
    last_shared_state : int
        The last shared state ID in the block.
    """

    num_iterations: int
    first_remote: int
    last_remote: int
    last_shared_state: int


class RemoteFunctionRpz(NamedTuple):
    """
    Represents a remote function.

    Attributes
    ----------
    name : str
        The name of the remote function.
    local : bool
        A flag indicating if the function is local.
    input_id : int
        The input ID of the function.
    output_id : int
        The output ID of the function.
    description : Optional[str], optional
        A description of the function, by default None.
    """

    name: str
    local: bool
    input_id: int
    output_id: int
    description: Union[str, None] = None


class LeafFunctionBlock(NamedTuple):
    """
    Represents a leaf function block.

    Attributes
    ----------
    name : str
        The name of the leaf function block.
    local : bool
        A flag indicating if the function block is local.
    input_id : int
        The input ID of the function block.
    output_id : int
        The output ID of the function block.
    description : Optional[str], optional
        A description of the function block, by default None.
    remote_id : Optional[int], optional
        The remote ID of the function block, by default None.
    """

    name: str
    local: bool
    input_id: int
    output_id: int
    description: Union[str, None] = None
    remote_id: Union[int, None] = None


class FunctionBlock(NamedTuple):
    """
    Represents a function block.

    Attributes
    ----------
    name : str
        The name of the function block.
    first_remote : int
        The first remote ID in the block.
    last_remote : int
        The last remote ID in the block.
    first_shared_state : int
        The first shared state ID in the block.
    last_shared_state : int
        The last shared state ID in the block.
    sub_blocks : list[Union['FunctionBlock', LeafFunctionBlock]]
        A list of sub-blocks within the function block.
    iteration_blocks : Optional[list[IterationBlock]], optional
        A list of iteration blocks within the function block, by default None.
    """

    name: str
    first_remote: int
    last_remote: int
    first_shared_state: int
    last_shared_state: int
    sub_blocks: list[Union["FunctionBlock", LeafFunctionBlock]]
    iteration_blocks: Union[list[IterationBlock], None] = None
