"""Common class and functions for tests."""
import tempfile
import unittest
from pathlib import Path
from typing import Union

import fedeca


def rmdir(directory: Union[Path, str]) -> None:
    """Remove all local-folder file.

    Parameters
    ----------
    directory : Union[Path, str]
        The directory in which to recursively search.
    """
    directory = Path(directory)
    for item in directory.iterdir():
        if item.is_dir() and "local-worker" in str(item):
            rmdir(item)


class TestTempDir(unittest.TestCase):
    """Base class for tests.

    Base class which should be used for every tests that need
    a temporary directory (to store data, logs etc).
    The directory is shared across the tests of a TestCase, and
    it's removed at the end of the TestCase (not at each test !).

    Attributes
    ----------
    test_dir: str
        the path to the temporary directory of the TestCase.

    Notes
    -----
        If the class methods setUpClass or tearDownClass are overridden,
        please make sure to call `super()...``
    """

    _test_dir = None
    test_dir = None

    @classmethod
    def setUpClass(cls):
        """Set up the class."""
        super(TestTempDir, cls).setUpClass()
        cls._test_dir = tempfile.TemporaryDirectory()
        cls.test_dir = cls._test_dir.name  # Keep a reference to the path

    @classmethod
    def tearDownClass(cls):
        """Tear down the class."""
        super(TestTempDir, cls).tearDownClass()
        # This function rm the directory
        cls._test_dir.cleanup()
        rmdir(Path(fedeca.__file__).parent.parent)
