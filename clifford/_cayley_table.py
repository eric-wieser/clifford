import weakref
import numpy as np
import functools
import operator
import sparse
from ._weak_cached_type import _WeakCachedType
from . import construct_tables, _powerset, get_mult_function


class CayleyTable:
    """
    Represents a multiplication table over an algebra

    Attributes
    ----------
    value : 3d array
        If :math:`c = ab`, this array :math:`m` is such that
        :math:`c_j = a_i m_{ijk} b_k`, using einstein summation notation.

        It is recommended that a sparse matrix be used
    """
    def __init__(self, value):
        self.value = value


class _CombinedDiagonalMetricProductTable(metaclass=_WeakCachedType):
    @classmethod
    def __canonicalizenewargs__(cls, metric):
        return (tuple(metric),)

    def __init__(self, metric):
        self.metric = np.array(metric)
        N = len(metric)

        blade_bitmaps     = np.empty(2**N, dtype=np.intp)
        blade_bitmaps_rev = np.empty(2**N, dtype=np.intp)
        grades            = np.empty(2**N, dtype=np.intp)

        for i, blade in enumerate(_powerset([1 << i for i in range(N)])):
            grades[i] = len(blade)
            combined_blade = functools.reduce(operator.or_, blade, 0)
            blade_bitmaps[i] = combined_blade
            blade_bitmaps_rev[combined_blade] = i

        self.geometric, imt_prod_mask, omt_prod_mask, lcmt_prod_mask = construct_tables(
            grades,
            blade_bitmaps,
            blade_bitmaps_rev,
            self.metric
        )
        self.outer = sparse.where(omt_prod_mask, self.geometric, self.geometric.dtype.type(0))
        self.inner = sparse.where(imt_prod_mask, self.geometric, self.geometric.dtype.type(0))
        self.left_contraction = sparse.where(lcmt_prod_mask, self.geometric, self.geometric.dtype.type(0))

        self.gmt_func = get_mult_function(self.geometric, grades)
        self.imt_func = get_mult_function(self.inner, grades)
        self.omt_func = get_mult_function(self.outer, grades)
        self.lcmt_func = get_mult_function(self.left_contraction, grades)

class _DiagonalMetricProductTable(CayleyTable,  metaclass=_WeakCachedType):
    @classmethod
    def __canonicalizenewargs__(cls, metric):
        return (tuple(metric),)

    def __init__(self, metric):
        self._combined = _CombinedDiagonalMetricProductTable(metric)
        super().__init__(getattr(self._combined, self._table_name))
        self.func = get_mult_function(self.value, None)

class DiagonalMetricGeometricProductTable(_DiagonalMetricProductTable):
    _table_name = 'geometric'


class DiagonalMetricOuterProductTable(_DiagonalMetricProductTable):
    _table_name = 'outer'


class DiagonalMetricInnerProductTable(_DiagonalMetricProductTable):
    _table_name = 'inner'

