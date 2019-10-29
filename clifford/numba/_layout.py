import numba
import numba.extending as nex

from .._layout import Layout


class LayoutType(numba.types.Opaque):
    def __init__(self, layout):
        assert isinstance(layout, Layout)
        self.layout = layout
        super().__init__("{!r}".format(layout))

    @property
    def key(self):
        return self.layout


@numba.typing.typeof.typeof_impl.register(Layout)
def _typeof_Layout(val: Layout, c) -> LayoutType:
    return LayoutType(val)


@nex.box(LayoutType)
def box_layout(typ: LayoutType, val, c):
    return c.pyapi.unserialize(c.pyapi.serialize_object(typ.layout))


@nex.unbox(LayoutType)
def unbox_layout(typ: LayoutType, val, c):
    return nex.NativeValue(c.context.get_dummy_value())


nex.register_model(LayoutType)(nex.models.OpaqueModel)
