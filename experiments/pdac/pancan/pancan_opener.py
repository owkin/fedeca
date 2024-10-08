"""Implement Pancan opener."""
from base_opener import FedECACenters, ZPDACOpener


class PanCanOpener(ZPDACOpener):
    """Implement PanCan specific opener.

    Parameters
    ----------
    PDACOpener : _type_
        _description_
    """

    def __init__(self):
        super().__init__(center_name=FedECACenters.PANCAN)
