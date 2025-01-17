""""Remove shape informations."""
import re


def remove_shape_tags(input_text: str) -> str:
    """
    Remove all <shape> tags and their content from the input text.

    Parameters
    ----------
    input_text : str
        The input text containing <shape> tags.

    Returns
    -------
    str
        The input text with all <shape> tags and their content removed.
    """
    pattern = re.compile(r"<shape>.*?<\/shape>", re.DOTALL)
    result = pattern.sub("", input_text)
    return result
