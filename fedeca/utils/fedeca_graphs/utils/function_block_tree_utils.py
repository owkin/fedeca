import re
from typing import Any

from fedeca.utils.fedeca_graphs.utils.data_classes import FunctionBlock
from fedeca.utils.fedeca_graphs.utils.data_classes import IterationBlock
from fedeca.utils.fedeca_graphs.utils.data_classes import LeafFunctionBlock
from fedeca.utils.fedeca_graphs.utils.data_classes import RemoteFunctionRpz
from typing import Union

def parse_block(block_content: str) -> dict[str, Any]:
    """
    Parse a block of content to extract its name and initialize sub_blocks.

    Parameters
    ----------
    block_content : str
        The content of the block to be parsed.

    Returns
    -------
    dict[str, Any]
        A dictionary containing the block name and an empty list for sub_blocks.
    """
    name_pattern = re.compile(r"<name>(.*?)<\/name>")
    name_match = name_pattern.search(block_content)
    name = name_match.group(1) if name_match else None

    return {
        "name": name,
        "sub_blocks": [],
    }


def parse_remote_function(remote_content: str) -> dict[str, int]:
    """
    Parse a remote function block to extract its remote ID.

    Parameters
    ----------
    remote_content : str
        The content of the remote function block to be parsed.

    Returns
    -------
    dict[str, int]
        A dictionary containing the remote ID.
    """
    remote_pattern = re.compile(r"<(remote|remote_data)>\s*(.*?)\s*<\/\1>", re.DOTALL)
    remote_matches = remote_pattern.findall(remote_content)

    assert (
        len(remote_matches) == 1
    ), f"Expected 1 remote block, found {len(remote_matches)}"
    remote_match = remote_matches[0]
    return {"remote_id": int(remote_match[1])}


def recursive_parse(content: str) -> list[dict[str, Any]]:
    """
    Recursively parse the content to extract blocks and their sub_blocks.

    Parameters
    ----------
    content : str
        The content to be parsed.

    Returns
    -------
    list[dict[str, Any]]
        A list of dictionaries representing the parsed blocks.
    """
    blocks: list[dict[str, Any]] = []
    stack: list[dict[str, Any]] = []
    current_block: Union[dict[str, Any], None] = None

    for line in content.splitlines():
        if "<bloc>" in line:
            if current_block:
                stack.append(current_block)
            current_block = {"lines": [line], "sub_blocks": []}
        elif "<remote>" in line or "<remote_data>" in line:
            assert (
                "</remote>" in line or "</remote_data>" in line
            ), f"Expected closing tag for remote block, found {line}"
            if current_block:
                stack.append(current_block)
            block_info = parse_remote_function(line)
            if stack:
                parent_block = stack.pop()
                parent_block["sub_blocks"].append(block_info)
                current_block = parent_block
            else:
                blocks.append(block_info)
                current_block = None
        elif "</bloc>" in line:
            assert current_block is not None, "No current block to close"
            current_block["lines"].append(line)
            block_content = "\n".join(current_block["lines"])
            block_info = parse_block(block_content)
            block_info["sub_blocks"] = current_block["sub_blocks"]
            if stack:
                parent_block = stack.pop()
                parent_block["sub_blocks"].append(block_info)
                current_block = parent_block
            else:
                blocks.append(block_info)
                current_block = None
        else:
            if current_block:
                current_block["lines"].append(line)

    return blocks


def build_function_blocks(
    blocks: list[dict[str, Any]],
    remote_functions: list[RemoteFunctionRpz],
    iteration_blocks: list[IterationBlock],
) -> tuple[list[Union[FunctionBlock, LeafFunctionBlock]], list[IterationBlock]]:
    """
    Build function blocks from the parsed blocks, remote functions, iteration blocks.

    Parameters
    ----------
    blocks : list[dict[str, Any]]
        The parsed blocks.
    remote_functions : list[RemoteFunctionRpz]
        A list of RemoteFunctionRpz objects.
    iteration_blocks : list[IterationBlock]
        A list of IterationBlock objects.

    Returns
    -------
    tuple[list[FunctionBlock | LeafFunctionBlock], list[IterationBlock]]
        A tuple containing a list of FunctionBlock objects and a list of
        used IterationBlock objects.
    """
    function_blocks: list[FunctionBlock | LeafFunctionBlock] = []
    used_iteration_blocks: list[IterationBlock] = []

    for block in blocks:
        if "remote_id" in block:
            remote_id = block["remote_id"]
            remote_function = remote_functions[remote_id]
            function_blocks.append(
                LeafFunctionBlock(
                    remote_function.name,
                    remote_function.local,
                    remote_function.input_id,
                    remote_function.output_id,
                    remote_function.description,
                    remote_id,
                )
            )
            continue
        if block["sub_blocks"] is not None:
            sub_blocks, used_iteration_blocks_sub_blocks = build_function_blocks(
                block["sub_blocks"], remote_functions, iteration_blocks=iteration_blocks
            )
            first_remote = (
                sub_blocks[0].first_remote
                if isinstance(sub_blocks[0], FunctionBlock)
                else sub_blocks[0].remote_id
            )
            last_remote = (
                sub_blocks[-1].last_remote
                if isinstance(sub_blocks[-1], FunctionBlock)
                else sub_blocks[-1].remote_id
            )
            assert first_remote is not None
            assert last_remote is not None
            first_shared_state = (
                sub_blocks[0].first_shared_state
                if isinstance(sub_blocks[0], FunctionBlock)
                else sub_blocks[0].output_id
            )
            last_shared_state = (
                sub_blocks[-1].last_shared_state
                if isinstance(sub_blocks[-1], FunctionBlock)
                else sub_blocks[-1].input_id
            )
            iteration_blocks_block = []
            for iteration_block in iteration_blocks:
                iteration_block_first_remote = iteration_block.first_remote
                iteration_block_last_remote = iteration_block.last_remote
                assert iteration_block_first_remote is not None
                assert iteration_block_last_remote is not None
                if (
                    (iteration_block_first_remote >= first_remote)
                    and (iteration_block_last_remote <= last_remote)
                    and (iteration_block not in used_iteration_blocks_sub_blocks)
                ):
                    iteration_blocks_block.append(iteration_block)

            function_blocks.append(
                FunctionBlock(
                    block["name"],
                    first_remote,
                    last_remote,
                    first_shared_state,
                    last_shared_state,
                    sub_blocks,
                    iteration_blocks_block,
                )
            )
            used_iteration_blocks.extend(iteration_blocks_block)
            used_iteration_blocks.extend(used_iteration_blocks_sub_blocks)
        else:
            raise ValueError("Block should have sub_blocks")

    return function_blocks, used_iteration_blocks


def extract_top_level_function_blocks(
    content: str,
    remote_functions: list[RemoteFunctionRpz],
    iteration_blocks: list[IterationBlock],
) -> list[Union[FunctionBlock, LeafFunctionBlock]]:
    """
    Extract block information from the content and build function blocks.

    Parameters
    ----------
    content : str
        The content to be parsed.
    remote_functions : list[RemoteFunctionRpz]
        A list of RemoteFunctionRpz objects.
    iteration_blocks : list[IterationBlock]
        A list of IterationBlock objects.

    Returns
    -------
    list[FunctionBlock | LeafFunctionBlock]
        A list of top-level FunctionBlock objects.
    """
    extracted_info = recursive_parse(content)
    top_level_blocks, _ = build_function_blocks(
        extracted_info,
        remote_functions=remote_functions,
        iteration_blocks=iteration_blocks,
    )
    return top_level_blocks
