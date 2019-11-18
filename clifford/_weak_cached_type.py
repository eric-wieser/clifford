import weakref


class _WeakCachedType(type):
    """
    A metaclass for cacheable types

    Types with this metaclass must define a ``__canonicalizenewargs__``,
    which converts their constructor arguments into a hashable type.

    If possible, these types will return an existing instance with the same
    canonical new args, rather than building a new type. As such, instances
    thereof should be considered immutable.

    As a result of this, any work done in the `__init__` of these types will
    likely be run just once. This work will _not_ be stored in any pickle files,
    but unpickling of an instance of these types will again avoid repeating the
    work.

    No effort has been made to ensure instances are shared when constructed
    across multiple threads - as a result, comparison and hash are implemented.
    """
    def __init__(cls, names, bases, dict):
        cls.__instances = weakref.WeakValueDictionary()
        cls.__hash__ = _WeakCachedType.__instance__hash
        cls.__eq__ = _WeakCachedType.__instance__eq
        cls.__ne__ = _WeakCachedType.__instance__ne
        cls.__reduce__ = _WeakCachedType.__instance_reduce

    def __instance__hash(cls_self):
        return cls_self.__newargs

    def __instance__eq(cls_self, other):
        if cls is other:
            return True
        if isinstance(other, type(cls_self)):
            return cls_self.__newargs == other.__newargs
        return NotImplemented

    def __instance__ne(cls_self, other):
        if cls is other:
            return False
        if isinstance(other, type(cls_self)):
            return cls_self.__newargs != other.__newargs
        return NotImplemented

    def __instance_reduce(self):#
        return (type(self).__newfromcanonical__, self.__newargs)

    def __call__(cls, *args, **kwargs):
        args = cls.__canonicalizenewargs__(*args, **kwargs)
        return cls.__newfromcanonical__(*args)

    def __newfromcanonical__(cls, *args):
        try:
            return cls.__instances[cls, args]
        except KeyError:
            # construct the object as normal
            self = super(_WeakCachedType, cls).__call__(*args)
            self.__newargs = args
            cls.__instances[cls, args] = self
            return self
