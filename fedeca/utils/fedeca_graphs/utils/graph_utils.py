"""Define useful utils for graph."""
from __future__ import annotations

from collections import OrderedDict
from typing import Literal

from graphviz import Digraph  # type: ignore

from fedeca.utils.fedeca_graphs.utils.data_classes import FunctionBlock
from fedeca.utils.fedeca_graphs.utils.data_classes import LeafFunctionBlock
from fedeca.utils.fedeca_graphs.utils.data_classes import SharedState

ITERATION_MAPPING = {
    20: r"x N_irls",
    98: r"x N_PQN -2",
    1: r"x N_gs - 2",
    39: r"x N_trim -1",
}

colors = OrderedDict(
    [
        ("sky_blue", "#00AEEF"),
        ("bluish_green", "#00C25E"),
        ("blue", "#0072BC"),
        ("reddish_purple", "#C6007E"),
        ("yellow", "#FECB00"),
        ("black", "#000000"),
        ("vermillion", "#E60012"),
        ("orange", "#F39800"),
        ("pink", "#FFBDE0"),
        ("teal", "#32C6C6"),
        ("green", "#2CB546"),
    ]
)


function_node_colors = {
    "local": colors["pink"],
    "aggregator": colors["teal"],
    "group": colors["green"],
}


def get_first_last_shared_state_id_from_block(
    block, max_depth,
):
    """
    Retrieve the first and last shared state IDs from a given function block.

    This function traverses a hierarchical structure of function blocks to
    find the first and last shared state IDs.
    It handles both composite function blocks (FunctionBlock) and leaf
    function blocks (LeafFunctionBlock).
    The traversal can be limited by a specified maximum depth.

    Parameters
    ----------
    block : FunctionBlock | LeafFunctionBlock
        The function block to traverse. It can be either a composite
        function block containing sub-blocks or a leaf function block.
    max_depth : int or None, optional
        The maximum depth to traverse. If None, the traversal is not
        depth-limited. If specified, the traversal will stop after reaching
        the given depth.

    Returns
    -------
    tuple[int, int] or None
        A tuple containing the first and last shared state IDs if found,
        otherwise None.

    Notes
    -----
    - If the input block is a LeafFunctionBlock, the function returns None.
    - If max_depth is specified and equals 0, the function returns None.
    - If the block contains only one sub-block, the function recursively
      calls itself on that sub-block.
    - The function checks for the first shared state in the first sub-block
      and the last shared state in the last sub-block.
    - If a shared state is not found in a sub-block,
      the function uses the output ID of the first sub-block and the
      input ID of the last sub-block as fallbacks.

    """
    if isinstance(block, LeafFunctionBlock):
        return None
    elif max_depth is not None and max_depth == 0:
        return None
    elif len(block.sub_blocks) == 1:
        return get_first_last_shared_state_id_from_block(
            block.sub_blocks[0], max_depth - 1 if max_depth is not None else None
        )
    else:
        # Check if there is a first shared state in the first sub block
        first_shared_state_sb = get_first_last_shared_state_id_from_block(
            block.sub_blocks[0], max_depth - 1 if max_depth is not None else None
        )
        if first_shared_state_sb is not None:
            first_shared_state = first_shared_state_sb[0]
        elif isinstance(block.sub_blocks[0], LeafFunctionBlock):
            first_shared_state = block.sub_blocks[0].output_id
        else:
            first_shared_state = block.sub_blocks[0].last_shared_state + 1
        # Check if there is a last shared state in the last sub block
        last_shared_state_sb = get_first_last_shared_state_id_from_block(
            block.sub_blocks[-1], max_depth - 1 if max_depth is not None else None
        )
        if last_shared_state_sb is not None:
            last_shared_state = last_shared_state_sb[1]
        elif isinstance(block.sub_blocks[-1], LeafFunctionBlock):
            last_shared_state = block.sub_blocks[-1].input_id
        else:
            last_shared_state = block.sub_blocks[-1].first_shared_state - 1
        return first_shared_state, last_shared_state


def add_function_block(
    dot: Digraph,
    block,
    shared_states: list[SharedState],
    max_depth,
    depth: int,
    shared_state_depth_map: dict[int, tuple[int, Digraph]],
    connect_with_previous_shared_state: bool = True,
    flatten_first_depth: bool = True,
    shared_state_naming: Literal["id", "content"] = "id",
    shared_state_mapping = None,
) -> tuple[
    tuple[set[int], set[int], set[tuple[int, int]], dict[int, int], dict[int, int]],
    dict[int, int],
    dict[int, tuple[int, Digraph]],
]:
    """
    Add a function block to the Graphviz Digraph.

    This function adds a function block
    (either a composite FunctionBlock or a LeafFunctionBlock)
    to a Graphviz Digraph. It handles the hierarchical
    structure of function blocks and shared states,
    and can limit the depth of traversal.

    Parameters
    ----------
    dot : Digraph
        The Graphviz Digraph to which the function block will be added.
    block : FunctionBlock or LeafFunctionBlock
        The function block to be added.
    shared_states : list[SharedState]
        A list of shared states.
    max_depth : int or None
        The maximum depth to which the function blocks should be added.
        If None, there is no depth limit.
    depth : int
        The current depth of the function block.
    shared_state_depth_map : dict[int, tuple[int, Digraph]]
        A dictionary mapping shared state IDs to their depth and subgraph.
    connect_with_previous_shared_state : bool, optional
        Whether to connect with the previous shared state, by default True.
    flatten_first_depth : bool, optional
        Whether to flatten the first depth, by default True.
    shared_state_naming : Literal["id", "content"], optional
        The naming convention for shared states, by default "id".
    shared_state_mapping : dict[int, int] or None, optional
        A mapping from shared state IDs to new IDs, by default None.

    Returns
    -------
    nodes_info: tuple[set[int], set[int], set[tuple[int, int]], dict[int, int],
        dict[int, int]]
        A tuple containing the added shared states, functions, and blocks.
        Shared states are identified by their IDs.
        Functions are identified by their remote IDs.
        Blocks are identified by their start and end remote IDs.
        The dictionaries map shared state IDs to function IDs and vice versa.
    remote_id_mapping: dict[int, int]
        A mapping from remote IDs to remote present in the graph.
        This is particularly useful for states with iteration blocks.
    shared_state_depth_map: dict[int, tuple[int, Digraph]]
        A dictionary mapping shared state IDs to their depth and subgraph in which
        they were created, so that the edges starting from the shared state can be
        connected to the proper subgraph.

    """
    if isinstance(block, LeafFunctionBlock):
        # Determine the color based on whether the block is local or aggregator
        color = (
            function_node_colors["local"]
            if block.local
            else function_node_colors["aggregator"]
        )
        # Add the function block node to the Digraph
        dot.node(
            f"func_{block.remote_id}",
            f"{block.name}",
            shape="box",
            style="filled",
            fillcolor=color,
        )
        state_to_func: dict[int, int] = {}
        func_to_state: dict[int, int] = {}
        # Connect with input shared state if specified
        block_remote_id = block.remote_id
        assert block_remote_id is not None
        if connect_with_previous_shared_state:
            shared_state_depth_map[block.input_id][1].edge(
                f"state_{block.input_id}",
                f"func_{block.remote_id}",
                arrowhead="normal",
                constraint="false"
                if (
                    flatten_first_depth
                    and shared_state_depth_map[block.input_id][0] == 0
                )
                else "true",
            )
            state_to_func[block.input_id] = block_remote_id
        added_states: set[int] = set()
        added_funcs: set[int] = {block_remote_id}
        added_blocks: set[tuple[int, int]] = set()

        return (
            (added_states, added_funcs, added_blocks, state_to_func, func_to_state),
            {block_remote_id: block_remote_id},
            shared_state_depth_map,
        )
    elif max_depth is not None and max_depth == 0:
        # Add the function block node to the Digraph with group color
        color = function_node_colors["group"]
        dot.node(
            f"func_{block.first_remote}",
            f"{block.name}",
            shape="box",
            style="filled",
            fillcolor=color,
        )
        state_to_func = {}
        func_to_state = {}
        if connect_with_previous_shared_state:
            previous_shared_state_idx = block.first_shared_state - 1
            shared_state_depth_map[previous_shared_state_idx][1].edge(
                f"state_{previous_shared_state_idx}",
                f"func_{block.first_remote}",
                arrowhead="normal",
                constraint="false"
                if (
                    flatten_first_depth
                    and shared_state_depth_map[previous_shared_state_idx][0] == 0
                )
                else "true",
            )
            state_to_func[previous_shared_state_idx] = block.first_remote
        return (
            (
                set(),
                set(),
                {(block.first_remote, block.last_remote)},
                state_to_func,
                func_to_state,
            ),
            {
                remote_id: block.first_remote
                for remote_id in range(block.first_remote, block.last_remote + 1)
            },
            shared_state_depth_map,
        )
    else:
        # Create a subgraph for the function block
        sub = Digraph(name=f"cluster_{block.first_remote}_{block.last_remote}")
        if depth > 0:
            sub.attr(
                label=block.name,
                style="dashed",
                labelloc="t",
                labeljust="l",
                rank="LR",
                splines="ortho",
            )
        else:
            sub.attr(
                label=block.name,
                style="dashed",
                labelloc="t",
                labeljust="l",
                rank="TB",
                splines="ortho",
            )
        remote_id_mapping = {}
        added_states = set()
        added_funcs = set()
        added_blocks = set()
        state_to_func = {}
        func_to_state = {}
        # Determine the span of each subblock and map each missing
        # state to the next subblock
        for idx_sub_block, sub_block in enumerate(block.sub_blocks):
            (
                (
                    added_states_sub_block,
                    added_funcs_sub_block,
                    added_blocks_sub_blocks,
                    state_to_func_sub_block,
                    func_to_state_sub_block,
                ),
                remote_id_mapping_block,
                shared_state_depth_map,
            ) = add_function_block(
                dot=sub,
                block=sub_block,
                shared_states=shared_states,
                max_depth=max_depth - 1 if max_depth is not None else None,
                depth=depth + 1,
                shared_state_depth_map=shared_state_depth_map,
                connect_with_previous_shared_state=connect_with_previous_shared_state,
                flatten_first_depth=flatten_first_depth,
                shared_state_naming=shared_state_naming,
                shared_state_mapping=shared_state_mapping,
            )
            added_states.update(added_states_sub_block)
            added_funcs.update(added_funcs_sub_block)
            added_blocks.update(added_blocks_sub_blocks)
            remote_id_mapping.update(remote_id_mapping_block)
            state_to_func.update(state_to_func_sub_block)
            func_to_state.update(func_to_state_sub_block)

            if idx_sub_block < len(block.sub_blocks) - 1:
                # Add the shared node
                if isinstance(sub_block, LeafFunctionBlock):
                    idx_next_shared_state = sub_block.output_id
                else:
                    idx_next_shared_state = sub_block.last_shared_state + 1

                next_shared_state = shared_states[idx_next_shared_state]
                # Trying to get the previous shared state
                origin = next_shared_state.origin
                assert origin is not None
                item_names = "\n".join(next_shared_state.items.keys())
                if shared_state_naming == "id":
                    if shared_state_mapping is not None:
                        name = f"{shared_state_mapping[idx_next_shared_state]}"
                    else:
                        name = f"{idx_next_shared_state}"
                else:
                    name = item_names
                # Create a fake subgraph containing the node whose
                # name is origin description
                if flatten_first_depth and depth == 0:
                    invisible_subgraph = Digraph(
                        name=(
                            f"cluster_{next_shared_state.origin}"
                            f"_{next_shared_state.destination}"
                        )
                    )
                    # No name, no border, no label for the subgraph
                    invisible_subgraph.attr(style="invis", color="white", label="")
                    invisible_subgraph.node(
                        f"state_{idx_next_shared_state}", name, shape="box"
                    )
                    # Add the subgraph to the main graph
                    sub.subgraph(invisible_subgraph)
                    # Get the first and last shared state of the subblock
                    sub.edge(
                        f"func_{remote_id_mapping[origin]}",
                        f"state_{idx_next_shared_state}",
                        arrowhead="none",
                        constraint="false",
                    )
                else:
                    sub.node(f"state_{idx_next_shared_state}", name, shape="box")
                    sub.edge(
                        f"func_{remote_id_mapping[origin]}",
                        f"state_{idx_next_shared_state}",
                        arrowhead="none",
                        constraint="true",
                    )
                added_states.add(idx_next_shared_state)
                shared_state_depth_map[idx_next_shared_state] = (depth, sub)
                func_to_state[remote_id_mapping[origin]] = idx_next_shared_state

        if block.iteration_blocks is not None:
            iteration_blocks = block.iteration_blocks
        else:
            iteration_blocks = []

        for iteration_block in iteration_blocks:
            if iteration_block.num_iterations > 1:
                last_shared_state = shared_states[iteration_block.last_shared_state]
                if shared_state_naming == "id":
                    idx = iteration_block.last_shared_state
                    name = (
                        f"{shared_state_mapping[idx]}"
                        if shared_state_mapping is not None
                        else f"{idx}"
                    )
                else:
                    name = "\n".join(last_shared_state.items.keys())
                sub.node(
                    f"state_{iteration_block.last_shared_state}_iter",
                    name,
                    shape="box",
                )
                sub.edge(
                    f"state_{iteration_block.last_shared_state}_iter",
                    f"func_{iteration_block.last_remote}",
                    arrowhead="normal",
                    style="dashed",
                    dir="back",
                    label=ITERATION_MAPPING[iteration_block.num_iterations - 1],
                )
                sub.edge(
                    f"func_{iteration_block.first_remote}",
                    f"state_{iteration_block.last_shared_state}_iter",
                    arrowhead="none",
                    style="dashed",
                    dir="back",
                    label=ITERATION_MAPPING[iteration_block.num_iterations - 1],
                )
        dot.subgraph(sub)
        return (
            (added_states, added_funcs, added_blocks, state_to_func, func_to_state),
            remote_id_mapping,
            shared_state_depth_map,
        )


def get_function_block_by_name(
    name: str,
    function_blocks,
    rank: int = 0,
):
    """
    Recursively search for a FunctionBlock by name within a sequence of function blocks.

    Parameters
    ----------
    name : str
        The name of the FunctionBlock to search for.
    function_blocks : Sequence[FunctionBlock | LeafFunctionBlock]
        A sequence of FunctionBlock or LeafFunctionBlock objects to search within.
    rank : int, optional
        The rank to search for, by default 0.

    Returns
    -------
    tuple[FunctionBlock | None, int | None]
        A tuple containing the found FunctionBlock (or None if not found) and
        the remaining rank.
    """
    current_rank = rank
    for block in function_blocks:
        if isinstance(block, LeafFunctionBlock):
            continue
        assert isinstance(block, FunctionBlock)
        if block.name == name and current_rank == 0:
            return block, None
        elif block.name == name and current_rank > 0:
            current_rank -= 1
        sub_block_res, output_rank = get_function_block_by_name(
            name, block.sub_blocks, current_rank
        )
        if sub_block_res is not None:
            return sub_block_res, None
        else:
            assert output_rank is not None
            current_rank = output_rank
    return None, current_rank


def create_workflow_graph(
    shared_states: list[SharedState],
    function_blocks: list[FunctionBlock],
    render_path,
    max_depth,
    function_block_name,
    rank: int = 0,
    flatten_first_depth: bool = True,
    shared_state_naming: Literal["id", "content"] = "id",
    render: bool = True,
    shared_state_mapping = None,
) -> tuple[set[int], set[int], set[tuple[int, int]]]:
    """
    Create a workflow graph from shared states and function blocks.

    This function generates a workflow graph using Graphviz,
    representing the relationships
    between shared states and function blocks.
     It can render the graph to a specified path
    and allows customization of various parameters.

    Parameters
    ----------
    shared_states : list[SharedState]
        A list of shared states. Each SharedState object contains
        information about the state,
        including its items, type, shape, and description.
    function_blocks : list[FunctionBlock]
        A list of function blocks. Each FunctionBlock object
        represents a block of functions
        in the workflow.
    render_path : str or None, optional
        The path to render the graph to, by default None. If None,
        the graph is not saved to a file.
    max_depth : int or None, optional
        The maximum depth to plot, by default None. If None,
        there is no depth limit.
    function_block_name : str or None, optional
        The name of the function block to plot, by default None.
        If None, the first function block is used.
    rank : int, optional
        The rank to search for, by default 0.
    flatten_first_depth : bool, optional
        Whether to flatten the first depth, by default True.
    shared_state_naming : Literal["id", "content"], optional
        The naming convention for shared states, by default "id".
        If "id", shared states are named by their IDs.
        If "content", shared states are named by their content.
    render : bool, optional
        Whether to render the graph, by default True.
    shared_state_mapping : dict[int, int] or None, optional
        A mapping from shared state IDs to new IDs, by default None.

    Returns
    -------
    tuple[set[int], set[int], set[tuple[int, int]]]
        A tuple containing sets of added states, functions, blocks,
        and dictionaries mapping shared state IDs
        to function IDs and vice versa.
        - The first set contains the IDs of the added shared states.
        - The second set contains the IDs of the added functions.
        - The third set contains tuples representing the start and end
          remote IDs of the added blocks.

    Raises
    ------
    ValueError
        If the specified function block name is not found in
        the list of function blocks.

    Notes
    -----
    - The function initializes the initial and final
      shared states and adds them to the graph.
    - It creates a fake subgraph for the initial and
      final shared states to ensure proper visualization.
    - The function block is added to the graph,
      and edges are created to connect the shared states and functions.
    - The graph can be rendered to a specified path
      if the render parameter is set to True.
    """
    dot = Digraph(comment="Workflow Graph")
    dot.attr(
        splines="ortho", ratio="compress", fontsize="10", nodesep="0.1", ranksep="0.25"
    )

    # Find the function block by name if provided
    if function_block_name is not None:
        function_block, _ = get_function_block_by_name(
            function_block_name,
            function_blocks,
            rank=rank,
        )
        print(function_block)
        if function_block is None:
            raise ValueError(
                f"Function block with name {function_block_name} not found"
            )
    else:
        function_block = function_blocks[0]
        print(function_block)

    # Initialize the initial shared state
    initial_shared_state_idx = function_block.first_shared_state - 1
    item_names = "\n".join(shared_states[initial_shared_state_idx].items.keys())
    if shared_state_naming == "id":
        name = (
            f"{shared_state_mapping[initial_shared_state_idx]}"
            if shared_state_mapping is not None
            else f"{initial_shared_state_idx}"
        )
    else:
        name = item_names

    # Create a fake subgraph containing the node whose name is origin description
    state_subgraph = Digraph(name="cluster_00")
    state_subgraph.attr(label="", style="invis", color="white")
    state_subgraph.node(f"state_{initial_shared_state_idx}", name, shape="box")
    dot.subgraph(state_subgraph)
    shared_state_depth_map = {initial_shared_state_idx: (0, dot)}

    # Add the function block to the graph
    nodes_info, remote_id_mapping, _ = add_function_block(
        dot=dot,
        block=function_block,
        shared_states=shared_states,
        max_depth=max_depth,
        depth=0,
        flatten_first_depth=flatten_first_depth,
        shared_state_depth_map=shared_state_depth_map,
        connect_with_previous_shared_state=True,
        shared_state_naming=shared_state_naming,
        shared_state_mapping=shared_state_mapping,
    )

    # Initialize the final shared state
    final_shared_state_idx = function_block.last_shared_state + 1
    item_names = "\n".join(shared_states[final_shared_state_idx].items.keys())
    if shared_state_naming == "id":
        name = (
            f"{shared_state_mapping[final_shared_state_idx]}"
            if shared_state_mapping is not None
            else f"{final_shared_state_idx}"
        )
    else:
        name = item_names

    # Create a fake subgraph containing the node whose name is origin description
    state_subgraph = Digraph(name=f"cluster_{function_block.last_remote + 1}")
    state_subgraph.attr(label="", style="invis", color="white")
    state_subgraph.node(f"state_{final_shared_state_idx}", name, shape="box")
    dot.subgraph(state_subgraph)

    # Add edge from the last function to the final shared state
    origin = shared_states[final_shared_state_idx].origin
    if origin is not None:
        dot.edge(
            f"func_{remote_id_mapping[origin]}",
            f"state_{final_shared_state_idx}",
            arrowhead="none",
            constraint="false" if flatten_first_depth else "true",
        )

    # Update the first nodes_info
    states_info, funcs_info, blocks_info, _, _ = nodes_info
    states_info.add(initial_shared_state_idx)
    states_info.add(final_shared_state_idx)

    # Render the graph if required
    if render:
        dot.render(render_path, format="png")

    return (states_info, funcs_info, blocks_info)
