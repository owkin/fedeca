import re

from fedeca.utils.fedeca_graphs.utils.data_classes import IterationBlock
from fedeca.utils.fedeca_graphs.utils.data_classes import RemoteFunctionRpz


def parse_iteration_blocks(
    input_text: str, remote_functions: list[RemoteFunctionRpz]
) -> list[IterationBlock]:
    """
    Parse iteration blocks from the given text and create IterationBlock objects.

    Parameters
    ----------
    input_text : str
        The input text containing iteration blocks.
    remote_functions : list[RemoteFunctionRpz]
        A list of RemoteFunctionRpz objects.

    Returns
    -------
    list[IterationBlock]
        A list of IterationBlock objects parsed from the input text.
    """
    pattern = re.compile(
        r"<iteration>\s*<number>(\d+)-(\d+)<\/number>\s*(.*?)\s*<\/iteration>",
        re.DOTALL,
    )
    matches = pattern.findall(input_text)

    results: list[tuple[int, int | None, int | None]] = []
    for match in matches:
        start, end, inner_content = int(match[0]), int(match[1]), match[2]
        num_iterations = end - start + 1

        remote_pattern = re.compile(r"<(remote|remote_data)>(\d+)<\/\1>")
        remote_matches = remote_pattern.findall(inner_content)

        if remote_matches:
            first_remote: int | None = int(remote_matches[0][1])
            last_remote: int | None = int(remote_matches[-1][1])
        else:
            first_remote = last_remote = None

        results.append((num_iterations, first_remote, last_remote))

    iteration_blocks: list[IterationBlock] = []

    for num_iterations, first_remote, last_remote in results:
        if first_remote is not None and last_remote is not None:
            last_shared_state = remote_functions[last_remote].output_id
            iteration_blocks.append(
                IterationBlock(
                    num_iterations, first_remote, last_remote, last_shared_state
                )
            )

    return iteration_blocks
