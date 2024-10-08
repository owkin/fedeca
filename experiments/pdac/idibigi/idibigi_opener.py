"""Implement IDIBIGi opener."""
from base_opener import FedECACenters, ZPDACOpener


class IDIBIGIOpener(ZPDACOpener):
    """Implement IDIBIGI specific opener.

    Parameters
    ----------
    PDACOpener : _type_
        _description_
    """

    def __init__(self):
        super().__init__(center_name=FedECACenters.IDIBIGI)
