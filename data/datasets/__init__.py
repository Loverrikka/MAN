
from .dukemtmcreid import DukeMTMCreID
from .market1501 import Market1501
from .msmt17 import MSMT17
from .RGBNT100 import RGBNT100
from .RGBNT201 import RGBNT201
from .msvr310 import MSVR310
from .MSVwild863 import MSVwild863
from .market_to_RGBNT201 import market_to_RGBNT201
from .MSVwild863_neo import MSVwild863_neo
from .MSVR310_neo import MSVR310_neo
from  .MSVR310_neo_100 import MSVR310_neo_100


__factory = {
    'market1501': Market1501,
    'dukemtmc': DukeMTMCreID,
    'msmt17': MSMT17,
    'RGBNT100': RGBNT100,
    'RGBNT201': RGBNT201,
    'msvr310': MSVR310,
    'MSVwild863' :MSVwild863,
    'market_to_RGBNT201' :market_to_RGBNT201,
    'MSVwild863_neo' : MSVwild863_neo,
    'MSVR310_neo' : MSVR310_neo,
    'MSVR310_neo_100' : MSVR310_neo_100,
}


def get_names():
    return __factory.keys()


def init_dataset(name, *args, **kwargs):
    if name not in __factory.keys():
        raise KeyError("Unknown datasets: {}".format(name))
    return __factory[name](*args, **kwargs)
