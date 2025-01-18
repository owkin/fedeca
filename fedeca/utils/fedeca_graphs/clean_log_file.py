"""Clean the raw workflow file produced by the tracing."""
# All utils here were originally written by Ulysse Marteau @umarteauowkin for
# fed-pydeseq2 (https://github.com/owkin/fed-pydeseq2) and are reproduced here with
# his permissions.
import pathlib
import re

import yaml  # type: ignore
from loguru import logger

from fedeca.utils.fedeca_graphs.constants import PATHS_FILE
from fedeca.utils.fedeca_graphs.utils.utils import remove_shape_tags


def filter_nested_remotes(input_text: str) -> str:
    """
    Filter out nested <remote_data> blocks.

    Parameters
    ----------
    input_text : str
        The input text containing <remote_data> blocks.

    Returns
    -------
    str
        The input text with nested <remote_data> blocks removed.
    """
    logger.info("Filtering nested remote_data blocks")
    nested_pattern = re.compile(
        r"(<remote_data>(?:(?!<\/?remote_data>).)*<remote_data>(?:(?!<\/?remote_data>).)*<\/remote_data>(?:(?!<\/?remote_data>).)*<\/remote_data>)", re.DOTALL  # noqa: E501
    )

    iteration = 0
    while True:
        match = nested_pattern.search(input_text)
        if not match:
            break

        logger.info(f"Iteration {iteration}: Found nested remote_data block")
        full_match = match.group(0)
        start = match.start()
        end = match.end()
        logger.debug(f"Removing nested block: {full_match}")
        # Remove the inner <remote_data> block while keeping the outer block intact
        inner_pattern = re.compile(r"<remote_data>(?:(?!<\/?remote_data>).)*<\/remote_data>", re.DOTALL)  # noqa: E501
        cleaned_inner_content = inner_pattern.sub('', full_match, count=1)
        logger.debug(f"Cleaned inner content: {cleaned_inner_content}")
        input_text = input_text[:start] + cleaned_inner_content + input_text[end:]

        iteration += 1

    return input_text


def merge_iterations(input_text: str) -> str:
    """
    Merge consecutive <iteration> blocks with the same content, combining their numbers.

    Parameters
    ----------
    input_text : str
        The input text containing <iteration> blocks.

    Returns
    -------
    str
        The input text with merged <iteration> blocks.
    """
    pattern = re.compile(
        r"(<iteration>\s*<number>(\d+)<\/number>\s*(.*?)\s*<\/iteration>)", re.DOTALL
    )
    matches = [
        (match.group(1), match.start(), match.group(2), match.group(3))
        for match in pattern.finditer(input_text)
    ]

    if not matches:
        return input_text

    merged_blocks = []
    current_text = matches[0][3]
    first_number = matches[0][2]
    previous_last_position = 0
    first_position = matches[0][1]
    last_number = matches[0][2]
    last_position: int | None = None
    text_outside_blocks = []

    for i in range(1, len(matches)):
        full_match, position, number, text = matches[i]

        if (
            remove_shape_tags(text) == remove_shape_tags(current_text)
            and int(number) == int(last_number) + 1
        ):
            last_number = number
        else:
            merged_blocks.append((first_number, last_number, current_text))
            text_outside_blocks.append(
                input_text[previous_last_position:first_position]
            )
            assert last_position is not None
            previous_last_position = last_position
            first_position = position
            current_text = text
            first_number = number
            last_number = number

        last_position = position + len(full_match)

    merged_blocks.append((first_number, last_number, current_text))
    text_outside_blocks.append(input_text[previous_last_position:first_position])
    text_outside_blocks.append(input_text[last_position:])

    merged_results = []
    for first, last, text in merged_blocks:
        merged_results.append(
            f"<iteration>\n<number>{first}-{last}</number>\n{text}\n</iteration>"
        )

    result = ""
    for i in range(len(merged_results)):
        result += text_outside_blocks[i]
        result += merged_results[i]

    result += text_outside_blocks[-1]
    return result


def filter_remote_data(input_text: str) -> str:
    """
    Filter out duplicate <remote_data> blocks with the same content.

    Parameters
    ----------
    input_text : str
        The input text containing <remote_data> blocks.

    Returns
    -------
    str
        The input text with duplicate <remote_data> blocks removed.
    """
    logger.info("Filtering remote_data")
    pattern = re.compile(r"(<remote_data>\s*(.*?)\s*<\/remote_data>)", re.DOTALL)
    matches = [
        (match.group(1), match.start(), match.group(2))
        for match in pattern.finditer(input_text)
    ]
    logger.info(f"{len(matches)} remote_data blocks found")

    if not matches:
        return input_text

    logger.info("Computing complement")
    complement = []
    for i in range(len(matches) + 1):
        if i == 0:
            start = 0
            end = matches[i][1]
        elif i == len(matches):
            start = matches[i - 1][1] + len(matches[i - 1][0])
            end = len(input_text)
        else:
            start = matches[i - 1][1] + len(matches[i - 1][0])
            end = matches[i][1]
        complement.append(input_text[start:end])

    kept_blocks = []
    kept_complement = []
    n_removed = 0
    reference_text = None
    for i, (full_match, start, text) in enumerate(matches):
        end = start + len(full_match)
        if reference_text is None:
            reference_text = remove_shape_tags(text)
            remove = False
        elif remove_shape_tags(text) == reference_text:
            if complement[i] == "\n":
                n_removed += 1
                remove = True
            else:
                remove = False
                reference_text = remove_shape_tags(text)
        else:
            reference_text = remove_shape_tags(text)
            remove = False
        if not remove:
            kept_blocks.append(full_match)
            kept_complement.append(complement[i])

    logger.info(f"{n_removed} remote_data blocks removed")
    kept_complement.append(complement[-1])
    result = ""
    for i in range(len(kept_blocks)):
        result += kept_complement[i]
        result += kept_blocks[i]
    result += kept_complement[-1]
    return result


def process_log_file(input_path: pathlib.Path, output_path: pathlib.Path) -> None:
    """
    Process a log file by filtering remote_data blocks and merging iteration blocks.

    Parameters
    ----------
    input_path : pathlib.Path
        The path to the input log file.

    output_path : pathlib.Path
        The path to the output log file.
    """
    with open(input_path) as file:
        input_text = file.read()

    output_text = filter_nested_remotes(input_text)
    output_text = filter_remote_data(output_text)
    output_text = merge_iterations(output_text)
    with open(output_path, "w") as file:
        file.write(output_text)


def main():
    """
    Clean the raw workflow log file.

    This function loads the paths from a YAML file,
    processes the raw workflow log file
    by filtering out duplicate <remote_data> blocks and merging
    consecutive <iteration> blocks,
    and writes the cleaned log file to the
    specified output path.
    """
    # load the paths from the yaml file
    with open(PATHS_FILE) as f:
        paths = yaml.safe_load(f)
    raw_workflow_file = paths["raw_workflow_file"]
    cleaned_workflow_file = paths["cleaned_workflow_file"]

    process_log_file(raw_workflow_file, cleaned_workflow_file)


if __name__ == "__main__":
    main()
