"""Utils for substra."""
from substra import Client as SubstraClient


class Client(SubstraClient):
    """Wrapper around substra clients to take the simu argument.

    This class mimics the behaviour of substra clients, except that it accepts
    an additional backend type, "simu", that behavaves like the subprocess mode
    but can be carried to the `Experiment` with only 1 parameter.

    Parameters
    ----------
    backend_type : str, optional
        Backend type to use.
    """

    def __init__(self, *args, **kwargs):
        if "backend_type" in kwargs:
            if kwargs["backend_type"] == "simu":
                # We remove it not to raise Errors for unrecognized backend
                kwargs["backend_type"] = None
                # We init it with default backend which is subprocess
                super().__init__(*args, **kwargs)
                # We tag it with a mark
                self.is_simu = True
            else:
                super().__init__(*args, **kwargs)
                # We tag it with a mark
                self.is_simu = False

        else:
            super().__init__(*args, **kwargs)
            # We tag it with a mark
            self.is_simu = False
