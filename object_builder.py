import lightgbm
import xgboost
import catboost
from typing import Callable, Dict, List
from types import ModuleType
import importlib

_modules = [lightgbm, xgboost, catboost]


def get_class(cls, modules: List = None):
    """ get class object from class name or path.
        modules: list of classes or modules, default = _modules
    """

    if isinstance(cls, Callable):
        # already a class
        return cls

    if '.' in cls:
        # path to class in module
        module_path, cls = cls.rsplit('.', 1)
        modules = [importlib.import_module(module_path)]
    elif not modules:
        modules = _modules

    classes_ = []
    for module in modules:
        if not isinstance(module, ModuleType) and module.__name__ == cls:
            classes_.append(module)
        elif hasattr(module, cls):
            classes_.append(getattr(module, cls))
    if len(classes_) != 1:
        raise ValueError("Cannot find class" if not classes_ else "Class name is not unique")
    return classes_[0]
