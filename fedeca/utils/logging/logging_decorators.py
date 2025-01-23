"""
Module containing decorators to log the input and outputs of a method.

All logging is controlled through a logging configuration file.
This configuration file can be either set by the log_config_path attribute of the class,
or by the default_config.ini file in the same directory as this module.
"""
import logging
import logging.config
import os
import pathlib
from collections.abc import Callable
from functools import wraps
from typing import Any
from typing import Union
import pandas as pd
import numpy as np

from fedeca.utils.logging.constants import LOGGING_SAVE_FILE


def log_save_local_state(method: Callable):
    """
    Decorate a method to log the size of the local state saved.

    This function is destined to decorate the save_local_state method of a class.

    It logs the size of the local state saved in the local state path, in MB.
    This is logged as an info message.

    Parameters
    ----------
    method : Callable
        The method to decorate. This method is expected to have the following signature:
        method(self, path: pathlib.Path).

    Returns
    -------
    Callable
        The decorated method, which logs the size of the local state saved.

    """

    @wraps(method)
    def remote_method_inner(
        self,
        path: pathlib.Path,
    ):
        logger = get_method_logger(self, method)

        output = method(self, path)

        logger.info(
            "Size of local state saved : "
            f"{os.path.getsize(path) / 1024 / 1024}"
            " MB"
        )

        return output

    return remote_method_inner


def log_organisation_method(method: Callable):
    """
    Decorate a method to log when it is called and when it ends.

    Parameters
    ----------
    method : Callable
        The method to decorate. This method is expected to have the following signature:
        method(self, *args, **kwargs).

    Returns
    -------
    Callable
        The decorated method, which logs when it is called and when it ends.

    """

    @wraps(method)
    def method_inner(
        self,
        *args,
        **kwargs,
    ):
        write_info_before_organisation_method(method, LOGGING_SAVE_FILE)
        output = method(self, *args, **kwargs)
        write_info_after_organisation_method(LOGGING_SAVE_FILE)

        return output

    return method_inner


def start_loop():
    """Add the <iterations> balise to the logging file."""
    # Add <iterations> balise
    text_to_add = "<iterations>\n"
    # Append the text to the file
    if LOGGING_SAVE_FILE is not None and LOGGING_SAVE_FILE.exists():
        with open(LOGGING_SAVE_FILE, "a") as file:
            file.write(text_to_add)


def end_loop():
    """Add the </iterations> balise to the logging file."""
    # Add </iterations> balise
    text_to_add = "</iterations>\n"
    # Append the text to the file
    if LOGGING_SAVE_FILE is not None and LOGGING_SAVE_FILE.exists():
        with open(LOGGING_SAVE_FILE, "a") as file:
            file.write(text_to_add)


def start_iteration(iteration_number: int):
    """
    Add the <iteration> balise to the logging file.

    Parameters
    ----------
    iteration_number : int
        The number of the iteration.
    """
    # Add <iteration> balise
    text_to_add = "<iteration>\n"
    # Add iteration number balise
    text_to_add += f"<number>{iteration_number}</number>\n"
    # Append the text to the file
    if LOGGING_SAVE_FILE is not None and LOGGING_SAVE_FILE.exists():
        with open(LOGGING_SAVE_FILE, "a") as file:
            file.write(text_to_add)


def end_iteration():
    """Add the </iteration> balise to the logging file."""
    # Add </iteration> balise
    text_to_add = "</iteration>\n"
    # Append the text to the file
    if LOGGING_SAVE_FILE is not None and LOGGING_SAVE_FILE.exists():
        with open(LOGGING_SAVE_FILE, "a") as file:
            file.write(text_to_add)


def log_remote_data(method: Callable):
    """
    Decorate a remote_data to log the input and outputs.

    This decorator logs the shared state keys with the info level,
    and the different layers of the local_adata and refit_adata with the debug level.

    This is done before and after the method call.

    Parameters
    ----------
    method : Callable
        The method to decorate. This method is expected to have the following signature:
        method(self, data_from_opener: ad.AnnData,
        shared_state: Any = None, **method_parameters).

    Returns
    -------
    Callable
        The decorated method, which logs the shared state keys with the info level
        and the different layers of the local_adata and refit_adata with the debug
        level.
    """

    @wraps(method)
    def remote_method_inner(
        self,
        data_from_opener: pd.DataFrame,
        shared_state: Any = None,
        **method_parameters,
    ):

        logger = get_method_logger(self, method)
        logger.info("---- Before running the method ----")
        log_shared_state_adatas(self, method, shared_state)

        write_info_before_function(
            method, shared_state, LOGGING_SAVE_FILE, "remote_data"
        )
        shared_state = method(self, data_from_opener, shared_state, **method_parameters)
        write_info_after_function(shared_state, LOGGING_SAVE_FILE, "remote_data")

        logger.info("---- After method ----")
        log_shared_state_adatas(self, method, shared_state)
        return shared_state

    return remote_method_inner


def log_remote(method: Callable):
    """
    Decorate a remote method to log the input and outputs.

    This decorator logs the shared state keys with the info level.

    Parameters
    ----------
    method : Callable
        The method to decorate. This method is expected to have the following signature:
        method(self, shared_states: Optional[list], **method_parameters).

    Returns
    -------
    Callable
        The decorated method, which logs the shared state keys with the info level.

    """

    @wraps(method)
    def remote_method_inner(
        self,
        shared_states: Union[list, None],
        **method_parameters,
    ):
        logger = get_method_logger(self, method)
        if shared_states is not None:
            shared_state = shared_states[0]
            if shared_state is not None:
                logger.info(
                    f"First input shared state keys : {shared_state}"
                )
            else:
                logger.info("First input shared state is None.")
        else:
            logger.info("No input shared states.")

        write_info_before_function(
            method,
            shared_states[0] if isinstance(shared_states, list) else None,  # type: ignore
            LOGGING_SAVE_FILE,
            "remote",
        )

        shared_state = method(self, shared_states, **method_parameters)

        write_info_after_function(shared_state, LOGGING_SAVE_FILE, "remote")

        if shared_state is not None:
            logger.info(f"Output shared state keys : {shared_state}")
        else:
            logger.info("No output shared state.")

        return shared_state

    return remote_method_inner


def log_shared_state_adatas(self: Any, method: Callable, shared_state: Union[dict, None]):
    """
    Log the information of the local step.

    Precisely, log the shared state keys (info),
    and the different layers of the local_adata and refit_adata (debug).

    Parameters
    ----------
    self : Any
        The class instance
    method : Callable
        The class method.
    shared_state : Optional[dict]
        The shared state dictionary, whose keys we log with the info level.

    """
    logger = get_method_logger(self, method)

    if shared_state is not None:
        logger.info(f"Shared state keys : {list(shared_state.keys())}")
    else:
        logger.info("No shared state")

    for adata_name in ["local_adata", "refit_adata"]:
        if hasattr(self, adata_name) and getattr(self, adata_name) is not None:
            adata = getattr(self, adata_name)
            logger.debug(f"{adata_name} layers : {list(adata.layers.keys())}")
            if "_available_layers" in self.local_adata.uns:
                available_layers = self.local_adata.uns["_available_layers"]
                logger.debug(f"{adata_name} available layers : {available_layers}")
            logger.debug(f"{adata_name} uns keys : {list(adata.uns.keys())}")
            logger.debug(f"{adata_name} varm keys : {list(adata.varm.keys())}")
            logger.debug(f"{adata_name} obsm keys : {list(adata.obsm.keys())}")


def write_info_before_organisation_method(method: Callable, file_path: pathlib.Path):
    """
    Append the information of the local step to a file.

    Parameters
    ----------
    method : Callable
        The method whose name will be logged.
    file_path : pathlib.Path
        The path to the file where the information will be appended.

    Notes
    -----
    This function appends the following information to the specified file:
    - The name of the method enclosed in `<name>` tags.
    """
    text_to_add = "<bloc>\n"
    # Add name balise
    text_to_add += f"<name>{method.__name__}</name>\n"
    # Append the text to the file
    if file_path is not None and file_path.exists():
        with open(file_path, "a") as file:
            file.write(text_to_add)


def write_info_after_organisation_method(file_path: pathlib.Path):
    """
    Append the information of the local step to a file.

    Parameters
    ----------
    file_path : pathlib.Path
        The path to the file where the information will be appended.

    Notes
    -----
    This function appends the following information to the specified file:
    - The name of the method enclosed in `<name>` tags.
    """
    text_to_add = "</bloc>\n"
    # Append the text to the file
    if file_path is not None and file_path.exists():
        with open(file_path, "a") as file:
            file.write(text_to_add)


def write_info_before_function(
    method: Callable, shared_state: Any, file_path: pathlib.Path, function_type: str
):
    """
    Append the information of the local step to a file.

    Parameters
    ----------
    method : Callable
        The method whose name will be logged.
    shared_state : Any
        The shared state containing the inputs to be logged.
        Expected to be a dictionary.
    file_path : pathlib.Path
        The path to the file where the information will be appended.
    function_type : str
        The type of the function (local, remote or remote_data).

    Notes
    -----
    This function appends the following information to the specified file:
    - The name of the method enclosed in `<name>` tags.
    - The inputs from the shared state enclosed in `<inputs>` tags. Each input includes:
        - `<key>`: The key of the input.
        - `<type>`: The type of the input.
        - `<shape>`: The shape of the input, if applicable.
    """
    text_to_add = f"<{function_type}>\n"
    # Add name balise
    text_to_add += f"<name>{method.__name__}</name>\n"
    # Add inputs opening balise
    text_to_add += "<input>\n"
    # For each key in the shared state, add a input basise withe three sub
    # balises : key, type and shape if relevant
    text_to_add += get_shared_state_balises(shared_state)

    text_to_add += "</input>\n"
    # Append the text to the file
    if file_path is not None and file_path.exists():
        with open(file_path, "a") as file:
            file.write(text_to_add)


def write_info_after_function(
    shared_state: Any, file_path: pathlib.Path, function_type: str
):
    """
    Append the information of the local step to a file.

    Parameters
    ----------
    shared_state : Any
        The shared state containing the inputs to be logged.
        Expected to be a dictionary.
    file_path : pathlib.Path
        The path to the file where the information will be appended.
    function_type : str
        The type of the function (local, remote or remote_data).

    Notes
    -----
    This function appends the following information to the specified file:
    - The name of the method enclosed in `<name>` tags.
    - The outputs from the shared state enclosed in `<outputs>` tags.
      Each output includes:
        - `<key>`: The key of the output.
        - `<type>`: The type of the output.
        - `<shape>`: The shape of the output, if applicable.
    """
    # Add outputs opening balise
    text_to_add = "<output>\n"
    # For each key in the shared state, add a output balise withe three sub
    # balises : key, type and shape if relevant
    text_to_add += get_shared_state_balises(shared_state)
    text_to_add += "</output>\n"
    text_to_add += f"</{function_type}>\n"
    # Append the text to the file
    if file_path is not None and file_path.exists():
        with open(file_path, "a") as file:
            file.write(text_to_add)


def get_shared_state_balises(shared_state: Any) -> str:
    """
    Get the shared state balises.

    Parameters
    ----------
    shared_state : Any
        The shared state containing the inputs to be logged.
        Expected to be a dictionary.

    Returns
    -------
    str
        The shared state balises.
    """
    text_to_add = ""
    # For each key in the shared state, add a input basise withe three sub
    # balises : key, type and shape if relevant
    if isinstance(shared_state, dict):
        for key, value in shared_state.items():
            text_to_add += "<item>\n"
            text_to_add += f"<key>{key}</key>\n"
            text_to_add += f"<type>{type(value)}</type>\n"
            if hasattr(value, "shape"):
                text_to_add += f"<shape>{value.shape}</shape>\n"
            text_to_add += "</item>\n"
        return text_to_add

    elif isinstance(shared_state, np.ndarray):
        # Do as if dictionnary with one key
        text_to_add += "<item>\n"
        text_to_add += "<key>array</key>\n"
        text_to_add += f"<type>{type(shared_state)}</type>\n"
        text_to_add += f"<shape>{shared_state.shape}</shape>\n"
        text_to_add += "</item>\n"
        return text_to_add

    return ""


def get_method_logger(self: Any, method: Callable) -> logging.Logger:
    """
    Get the method logger from a configuration file.

    If the class instance has a log_config_path attribute,
    the logger is configured with the file at this path.

    Parameters
    ----------
    self: Any
        The class instance
    method: Callable
        The class method.

    Returns
    -------
    logging.Logger
        The logger instance.
    """
    if hasattr(self, "log_config_path"):
        log_config_path = pathlib.Path(self.log_config_path)
    else:
        log_config_path = pathlib.Path(__file__).parent / "default_config.ini"
    logging.config.fileConfig(log_config_path, disable_existing_loggers=False)
    logger = logging.getLogger(method.__name__)
    return logger
