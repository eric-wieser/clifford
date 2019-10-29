"""
Numba support for MultiVector objects.

For now, this just supports .value wrapping / unwrapping
"""
import numpy as np
import numba
from numba.extending import (
    models, register_model, box, unbox, NativeValue, make_attribute_wrapper,
    overload_method, lower_getattr, type_callable, lower_builtin,
    overload_attribute
)
from numba.targets.imputils import impl_ret_borrowed
from numba.typing.typeof import typeof_impl
from numba import cgutils, types, from_dtype

from .._multivector import MultiVector
from .._layout import Layout

from ._layout import LayoutType

__all__ = ['MultiVectorType']


class MultiVectorType(types.Type):
    def __init__(self, layout: Layout, dtype: np.dtype):
        assert isinstance(layout, Layout)
        assert isinstance(dtype, np.dtype)
        self.layout = layout
        self.dtype = dtype
        super().__init__(name='MultiVector[{!r}, {!r}]'.format(layout, numba.from_dtype(dtype)))

    @property
    def value_type(self):
        return numba.from_dtype(self.dtype)[:]

    @property
    def layout_type(self):
        return numba.typeof(self.layout)


@typeof_impl.register(MultiVector)
def _typeof_MultiVector(val: MultiVector, c) -> MultiVectorType:
    return MultiVectorType(layout=val.layout, dtype=val.value.dtype)


@register_model(MultiVectorType)
class MultiVectorModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [
            ('value', numba.from_dtype(fe_type.dtype)[:]),
        ]
        models.StructModel.__init__(self, dmm, fe_type, members)


@type_callable(MultiVector)
def type_MultiVector(context):
    def typer(layout, value):
        if isinstance(layout, LayoutType) and isinstance(value, types.Array):
            return MultiVectorType(layout.layout, numba.numpy_support.as_dtype(value.dtype))
    return typer


@lower_builtin(MultiVector, LayoutType, types.Any)
def impl_MultiVector(context, builder, sig, args):
    typ = sig.return_type
    _, value = args
    mv = cgutils.create_struct_proxy(typ)(context, builder)
    mv.value = value
    return impl_ret_borrowed(context, builder, sig.return_type, mv._getvalue())


@unbox(MultiVectorType)
def unbox_MultiVector(typ: MultiVectorType, obj: MultiVector, c) -> NativeValue:
    value = c.pyapi.object_getattr_string(obj, "value")
    mv = cgutils.create_struct_proxy(typ)(c.context, c.builder)
    mv.value = c.unbox(typ.value_type, value).value
    c.pyapi.decref(value)
    is_error = cgutils.is_not_null(c.builder, c.pyapi.err_occurred())
    return NativeValue(mv._getvalue(), is_error=is_error)


@box(MultiVectorType)
def box_MultiVector(typ: MultiVectorType, val: NativeValue, c) -> MultiVector:
    mv = cgutils.create_struct_proxy(typ)(c.context, c.builder, value=val)
    mv_obj = c.box(typ.value_type, mv.value)
    layout_obj = c.pyapi.unserialize(c.pyapi.serialize_object(typ.layout))
    class_obj = c.pyapi.unserialize(c.pyapi.serialize_object(MultiVector))
    res = c.pyapi.call_function_objargs(class_obj, (layout_obj, mv_obj))
    c.pyapi.decref(mv_obj)
    c.pyapi.decref(class_obj)
    c.pyapi.decref(layout_obj)
    return res


make_attribute_wrapper(MultiVectorType, 'value', 'value')

@overload_attribute(MultiVectorType, 'layout')
def get_layout(mv):
    l = mv.layout
    return lambda mv: l
