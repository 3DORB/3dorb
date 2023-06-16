import importlib
from types import SimpleNamespace
import logging
from typing import List, Union, Tuple, Dict, Any


logger = logging.getLogger(__name__)


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def build_from_config(config, **kwargs):
    if config is None:
        return None
    if isinstance(config, str):
        return get_obj_from_str(config)
    obj = get_obj_from_str(config['__target__'])
    return obj(**config['kwargs'], **kwargs)


def list_of_dicts__to__dict_of_lists(lst: Union[List, Tuple]):
    assert isinstance(lst, (list, tuple)), type(lst)
    if len(lst) == 0:
        return {}
    keys = lst[0].keys()
    output_dict = dict()
    for d in lst:
        assert set(d.keys()) == set(keys), (d.keys(), keys)
        for k in keys:
            if k not in output_dict:
                output_dict[k] = []
            output_dict[k].append(d[k])
    return output_dict


def overwrite_cfg(cfg: Dict, key: str, value: Any, recursive=False, check_exists=True):
    if check_exists:
        assert key in cfg, key
    if key in cfg and recursive and isinstance(value, dict):
        for k, v in value.items():
            overwrite_cfg(cfg[key], k, v, recursive=recursive, check_exists=check_exists)
    else:
        logger.info(f'overwrite key {key}: {cfg.get(key)} -> {value}')
        cfg[key] = value
    return cfg


def get_attrdict(d: Dict) -> SimpleNamespace:
    for k, v in d.items():
        if isinstance(v, dict):
            d[k] = get_attrdict(v)
    return SimpleNamespace(**d)
