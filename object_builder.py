from typing import Callable, Dict, List
from types import ModuleType
from importlib import import_module


class ObjectBuilder:
    """ constructs objects from class names and arguments """

    def __init__(self, modules: List):
        """ modules: list of modules, module names, or classes.
            classes will be searched for within this list.
        """
        self.modules = [import_module(m) if isinstance(m, str) else m for m in modules]

    def get_instance(self, cls, kws: Dict) -> Callable:
        class_ = self.get_class(cls)
        return class_(**kws)

    def get_class(self, cls) -> Callable:
        """ get class object from class name or path """

        if isinstance(cls, Callable):
            # already a class
            return cls

        if '.' in cls:
            # path to class in module, overrides modules list
            module_path, cls = cls.rsplit('.', 1)
            modules = [import_module(module_path)]
        else:
            modules = self.modules

        classes_ = []
        for module in modules:
            if not isinstance(module, ModuleType) and module.__name__ == cls:
                classes_.append(module)
            elif hasattr(module, cls):
                classes_.append(getattr(module, cls))
        if len(classes_) != 1:
            raise ValueError(("Cannot find class: " if not classes_ else "Class name is not unique: ") + cls)
        return classes_[0]
