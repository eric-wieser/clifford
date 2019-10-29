import numba
from clifford.numba import MultiVectorType

from numba.extending import lower_getattr
from numba import types
from clifford import MultiVector
import numpy as np
import weakref

@numba.njit
def add(a):
    return MultiVector(a.layout, a.value*2)

from clifford.g3c import e1
e = add(e1)
print(e)

v = weakref.ref(e.value)

del e

print(v)

print(e1)

