import lightgbm
import xgboost
import catboost
from typing import Callable, Dict, List
import importlib

_modules = [lightgbm, xgboost, catboost]


def get_instance(cls, kws: Dict):
    """ get instance from class name and kw args """
    class_ = get_class(cls)
    instance = class_(**kws)
    return instance


def get_class(cls, modules: List = None):
    """ get class object from class name or path.
        modules: list of modules class may belong to, default = _modules
    """

    if isinstance(cls, Callable):
        return cls

    if '.' in cls:
        module_path, cls = cls.rsplit('.', 1)
        modules = [importlib.import_module(module_path)]
    elif not modules:
        modules = _modules

    classes_ = [getattr(module, cls) for module in modules if hasattr(module, cls)]
    if len(classes_) != 1:
        raise ValueError("Cannot find class" if not classes_ else "Class name is not unique")
    return classes_[0]
