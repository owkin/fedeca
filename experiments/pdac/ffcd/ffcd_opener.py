"""Implement FFCD opener."""
from base_opener import FedECACenters, ZPDACOpener


class FFCDOpener(ZPDACOpener):
    """Implement FFCD specific opener.

    Parameters
    ----------
    PDACOpener : _type_
        _description_
    """

    def __init__(self):
        super().__init__(center_name=FedECACenters.FFCD)
