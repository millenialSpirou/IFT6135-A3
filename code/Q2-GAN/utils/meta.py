r"""
:mod:`utils.meta` -- Miscellaneous boilerplate code for Python software patterns
================================================================================

.. module:: meta
   :platform: Unix
   :synopsis: Singleton, Factories via metaclasses
   :author: Christos Tsirigotis

"""
from abc import ABCMeta
from collections import defaultdict
from glob import glob
from importlib import import_module
import os

import pkg_resources


class SingletonError(ValueError):
    """Exception to be raised when someone provides arguments to build
    an object from a already-instantiated `SingletonType` class.
    """

    def __init__(self, cls):
        """Pass the same constant message to ValueError underneath."""
        msg = "A singleton instance of '{}' has already been instantiated."
        super().__init__(msg.format(cls.__name__))


class SingletonType(type):
    """Metaclass that implements the singleton pattern for a Python class."""

    def __init__(cls, name, bases, dictionary):
        """Create a class instance variable and initiate it to None object."""
        super(SingletonType, cls).__init__(name, bases, dictionary)
        cls.instance = None

    def __call__(cls, *args, **kwargs):
        """Create an object if does not already exist, otherwise return what there is."""
        if cls.instance is None:
            cls.instance = super(SingletonType, cls).__call__(*args, **kwargs)
        elif args or kwargs:
            raise SingletonError(cls)
        return cls.instance


def get_all_subclasses(parent):
    """Get set of subclasses recursively"""
    subclasses = list()
    for subclass in parent.__subclasses__():
        subclasses.append(subclass)
        subclasses += get_all_subclasses(subclass)

    return subclasses


class Factory(ABCMeta):
    """Instantiate appropriate wrapper for the infrastructure based on input
    argument, ``of_type``.

    Attributes
    ----------
    types : dict of subclasses of ``cls.__base__``
       Updated to contain all possible implementations currently. Check out code.

    """

    def __init__(cls, names, bases, dictionary):
        """Search in directory for attribute names subclassing `bases[0]`"""
        super(Factory, cls).__init__(names, bases, dictionary)

        cls.modules = []
        cls.module_priority = defaultdict(int)
        cls.module_priority['__main__'] = -1000
        base = import_module(cls.__base__.__module__)
        try:
            py_files = glob(os.path.abspath(os.path.join(base.__path__[0], '[A-Za-z]*.py')))
            py_mods = map(lambda x: '.' + os.path.split(os.path.splitext(x)[0])[1], py_files)
            for py_mod in py_mods:
                cls.modules.append(import_module(py_mod, package=cls.__base__.__module__))
        except AttributeError:
            # This means that base class and implementations reside in a module
            # itself and not a subpackage.
            pass

        # Get types advertised through entry points!
        for entry_point in pkg_resources.iter_entry_points(cls.__name__):
            entry_point.load()

    def find_types(cls):
        types = get_all_subclasses(cls.__base__)
        exclude = [kls.__name__
                   for kls in types
                   if kls.__name__.startswith('_')]
        exclude.append(cls.__name__)
        if hasattr(cls, 'exclude'):
            exclude += list(cls.exclude)
        return [class_ for class_ in types
                if class_.__name__ not in exclude]

    def set_module_priority(cls, mod, prior=0):
        cls.module_priority[mod.__name__] = prior

    @property
    def typenames(cls):
        return set([class_.__name__ for class_ in cls.find_types()])

    def __call__(cls, type, *args, **kwargs):
        """Create an object, instance of ``cls.__base__``, on first call.

        :param of_type: Name of class, subclass of ``cls.__base__``, wrapper
           of a database framework that will be instantiated on the first call.
        :param args: positional arguments to initialize ``cls.__base__``'s instance (if any)
        :param kwargs: keyword arguments to initialize ``cls.__base__``'s instance (if any)

        .. seealso::
           `Factory.typenames` for values of argument `of_type`.

        .. seealso::
           Attributes of ``cls.__base__`` and ``cls.__base__.__init__`` for
           values of `args` and `kwargs`.

        .. note:: New object is saved as `Factory`'s internal state.

        :return: The object which was created on the first call.
        """
        found = []
        types = cls.find_types()
        for class_ in types:
            if class_.__name__.lower() == type.lower():
                found.append(class_)

        if found:
            found.sort(key=lambda x: cls.module_priority[x.__module__])
            return found[0].__call__(*args, **kwargs)

        error = "Could not find implementation of {0}, type = '{1}'".format(
            cls.__base__.__name__, type)
        error += "\nCurrently, there is an implementation for types:\n"
        error += str(cls.typenames)
        raise NotImplementedError(error)
