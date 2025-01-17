"""Define utils related to shared_state"""
import re

from loguru import logger

from fedeca.utils.fedeca_graphs.utils.data_classes import Item
from fedeca.utils.fedeca_graphs.utils.data_classes import RemoteFunctionRpz
from fedeca.utils.fedeca_graphs.utils.data_classes import SharedState
from fedeca.utils.fedeca_graphs.utils.utils import remove_shape_tags


def create_items_from_text(text: str) -> SharedState:
    """
    Create a SharedState object from the given text.

    Parameters
    ----------
    text : str
        The input text containing item definitions.

    Returns
    -------
    SharedState
        A SharedState object containing the parsed items.
    """
    items: dict[str, Item] = {}
    pattern = re.compile(r"<item>\s*(.*?)\s*<\/item>", re.DOTALL)
    for match in pattern.finditer(text):
        item_text = match.group(1)
        name_match = re.search(r"<key>(.*?)<\/key>", item_text)
        assert name_match is not None, "Item name not found"
        name = name_match.group(1)
        type_match = re.search(r"<type>(.*?)<\/type>", item_text)
        assert type_match is not None, "Item type not found"
        type_str = type_match.group(1)
        shape_match = re.search(r"<shape>(.*?)<\/shape>", item_text)
        shape = None
        if shape_match:
            list_of_str = shape_match.group(1)[1:-1].split(",")
            if list_of_str[-1] == "":
                list_of_str.pop()
            shape = tuple(map(int, list_of_str))

        description = None
        description_match = re.search(r"<description>(.*?)<\/description>", item_text)
        if description_match:
            description = description_match.group(1)

        items[name] = Item(type_str, shape, description)

    return SharedState(items)


def process_input_output_blocks(input_text: str) -> tuple[str, dict[int, SharedState]]:
    """
    Process input and output blocks in the given text and replace them with unique IDs.

    Parameters
    ----------
    input_text : str
        The input text containing input and output blocks.

    Returns
    -------
    tuple[str, dict[int, SharedState]]
        A tuple containing the processed text and a dictionary mapping
        IDs to SharedState objects.
    """
    input_pattern = re.compile(r"(<input>\s*(.*?)\s*<\/input>\n)", re.DOTALL)
    output_pattern = re.compile(r"(<output>\s*(.*?)\s*<\/output>\n)", re.DOTALL)

    input_matches = [
        (match.group(1), match.start(), match.group(2))
        for match in input_pattern.finditer(input_text)
    ]
    output_matches = [
        (match.group(1), match.start(), match.group(2))
        for match in output_pattern.finditer(input_text)
    ]

    assert len(input_matches) == len(
        output_matches
    ), "Number of input and output blocks must be equal"

    id_counter = 1
    n_potential_pairs = len(input_matches) - 1
    id_map: dict[int, str] = {0: input_matches[0][2]}
    input_map: dict[int, int] = {0: 0}
    output_map: dict[int, int] = {}

    for i in range(n_potential_pairs):
        if remove_shape_tags(input_matches[i + 1][2]) == remove_shape_tags(
            output_matches[i][2]
        ):
            input_map[i + 1] = id_counter
            output_map[i] = id_counter
            id_map[id_counter] = input_matches[i + 1][2]
            id_counter += 1
        else:
            output_map[i] = id_counter
            id_map[id_counter] = output_matches[i][2]
            input_map[i + 1] = id_counter + 1
            id_map[id_counter + 1] = input_matches[i + 1][2]
            id_counter += 2

    output_map[n_potential_pairs] = id_counter
    id_map[id_counter] = input_matches[n_potential_pairs][2]
    logger.info(f"{id_counter + 1} unique input-output pairs found")

    output_text = ""
    starting_point = 0
    for i in range(n_potential_pairs + 1):
        before_input = input_text[starting_point : input_matches[i][1]]
        output_text += before_input
        output_text += f"<input>{input_map[i]}</input>\n"
        starting_point = input_matches[i][1] + len(input_matches[i][0])
        before_output = input_text[starting_point : output_matches[i][1]]
        output_text += before_output
        output_text += f"<output>{output_map[i]}</output>\n"
        starting_point = output_matches[i][1] + len(output_matches[i][0])
    output_text += input_text[starting_point:]

    output_id_map = {k: create_items_from_text(v) for k, v in id_map.items()}

    return output_text, output_id_map


def create_remote_functions_from_text(text: str) -> tuple[list[RemoteFunctionRpz], str]:
    """
    Create RemoteFunctionRpz objects from the given text.

    Replace the original blocks with unique IDs.

    Parameters
    ----------
    text : str
        The input text containing <remote> or <remote_data> blocks.

    Returns
    -------
    tuple[list[RemoteFunctionRpz], str]
        A tuple containing a list of RemoteFunctionRpz objects and
        the processed text with unique IDs.
    """
    remote_functions: list[RemoteFunctionRpz] = []
    pattern = re.compile(r"<(remote|remote_data)>\s*(.*?)\s*<\/\1>", re.DOTALL)
    output_text = ""
    last_pos = 0
    function_counter = 0

    for match in pattern.finditer(text):
        output_text += text[last_pos : match.start()]
        output_text += f"<{match.group(1)}>{function_counter}</{match.group(1)}>"
        last_pos = match.end()
        item_text = match.group(2)
        name_match = re.search(r"<name>(.*?)<\/name>", item_text)
        assert name_match is not None, "Function name not found"
        name = name_match.group(1)
        input_id_match = re.search(r"<input>(.*?)<\/input>", item_text)
        assert input_id_match is not None, "Input ID not found"
        input_id = int(input_id_match.group(1))
        output_id_match = re.search(r"<output>(.*?)<\/output>", item_text)
        assert output_id_match is not None, "Output ID not found"
        output_id = int(output_id_match.group(1))
        local = match.group(1) == "remote_data"
        remote_functions.append(RemoteFunctionRpz(name, local, input_id, output_id))
        function_counter += 1

    output_text += text[last_pos:]

    return remote_functions, output_text


def update_items_from_functions(
    remote_functions: list[RemoteFunctionRpz], shared_states: dict[int, SharedState]
) -> dict[int, SharedState]:
    """
    Update the origin, destination, and shared_with_aggregator fields.

    This is based on remote_functions.

    Parameters
    ----------
    remote_functions : list[RemoteFunctionRpz]
        A list of RemoteFunctionRpz objects.
    shared_states : dict[int, SharedState]
        A dictionary mapping IDs to SharedState objects.

    Returns
    -------
    dict[int, SharedState]
        The updated dictionary of SharedState objects.
    """
    for i, remote_function in enumerate(remote_functions):
        assert (
            remote_function.input_id in shared_states
        ), f"input_id {remote_function.input_id} not found in shared_states"
        shared_states[remote_function.input_id] = shared_states[
            remote_function.input_id
        ]._replace(destination=i, shared_with_aggregator=not remote_function.local)

        assert (
            remote_function.output_id in shared_states
        ), f"output_id {remote_function.output_id} not found in shared_states"
        shared_states[remote_function.output_id] = shared_states[
            remote_function.output_id
        ]._replace(origin=i)

    return shared_states


def create_shared_states_remote_functions(
    input_text: str,
) -> tuple[str, dict[int, SharedState], list[RemoteFunctionRpz]]:
    """
    Create SharedState objects and RemoteFunctionRpz objects from the given text.

    Parameters
    ----------
    input_text : str
        The input text containing input, output, and remote blocks.

    Returns
    -------
    tuple[str, dict[int, SharedState], list[RemoteFunctionRpz]]
        A tuple containing the processed text with unique IDs, a dictionary
        mapping IDs to SharedState objects, and a list of RemoteFunctionRpz objects.
    """
    output_text, id_mapping = process_input_output_blocks(input_text)
    remote_functions, output_text_v2 = create_remote_functions_from_text(output_text)
    shared_states = update_items_from_functions(remote_functions, id_mapping)
    return output_text_v2, shared_states, remote_functions
