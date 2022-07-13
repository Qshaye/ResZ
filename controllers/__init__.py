REGISTRY = {}

from .basic_controller import BasicMAC
from .basic_central_controller import CentralBasicMAC
from .base_controller import BaseCentralMAC
from .resz_controller import ReszMAC

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["basic_central_mac"] = CentralBasicMAC
REGISTRY['base_central_mac'] = BaseCentralMAC
REGISTRY['resz_mac'] = ReszMAC
