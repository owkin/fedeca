"""Create intermediary representations of the workflow."""
# All utils here were originally written by Ulysse Marteau @umarteauowkin for
# fed-pydeseq2 (https://github.com/owkin/fed-pydeseq2) and are reproduced here with
# his permissions.
import pathlib
import pickle

import yaml  # type: ignore

from fedeca.utils.fedeca_graphs.constants import PATHS_FILE
from fedeca.utils.fedeca_graphs.utils.function_block_tree_utils import (
    extract_top_level_function_blocks,
)
from fedeca.utils.fedeca_graphs.utils.iteration_blocks_utils import parse_iteration_blocks  # noqa: E501
from fedeca.utils.fedeca_graphs.utils.shared_states_remote_functions_utils import (
    create_shared_states_remote_functions,
)


def process_cleaned_log_file_to_tree(
    input_path: pathlib.Path, output_dir: pathlib.Path
):
    """Create the tree of the workflow.

    Parameters
    ----------
    input_path : pathlib.Path
        The path to the input log file.

    output_dir : pathlib.Path
        The path to the output directory.

    """
    with open(input_path) as f:
        input_text = f.read()

    output_dir.mkdir(exist_ok=True, parents=True)

    output_text_path = output_dir / "processed_workflow.txt"

    shared_states_path = output_dir / "shared_states.pkl"
    remote_functions_path = output_dir / "remote_functions.pkl"
    iteration_blocks_path = output_dir / "iteration_blocks.pkl"
    function_blocks_path = output_dir / "function_blocks.pkl"

    (
        output_text,
        shared_states,
        remote_functions,
    ) = create_shared_states_remote_functions(input_text)

    iteration_blocks = parse_iteration_blocks(output_text, remote_functions)
    function_blocks = extract_top_level_function_blocks(
        output_text, remote_functions, iteration_blocks
    )

    with open(output_text_path, "w") as f:
        f.write(output_text)

    # Pickle write the output
    with open(shared_states_path, "wb") as f:
        pickle.dump(shared_states, f)

    # Pickle write the remote functions
    with open(remote_functions_path, "wb") as f:
        pickle.dump(remote_functions, f)

    # Pickle write the iteration blocks
    with open(iteration_blocks_path, "wb") as f:
        pickle.dump(iteration_blocks, f)

    # Pickle write the function blocks
    with open(function_blocks_path, "wb") as f:
        pickle.dump(function_blocks, f)


def main():
    """Create the tree of the workflow."""
    # load the paths from the yaml file
    with open(PATHS_FILE) as f:
        paths = yaml.safe_load(f)
    cleaned_workflow_file = pathlib.Path(paths["cleaned_workflow_file"])
    processed_workflow_dir = pathlib.Path(paths["processed_workflow_dir"])

    process_cleaned_log_file_to_tree(cleaned_workflow_file, processed_workflow_dir)


if __name__ == "__main__":
    main()
