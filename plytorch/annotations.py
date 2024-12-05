from collections import OrderedDict
from dataclasses import dataclass
from typing import Annotated, get_origin, get_args, Optional, Callable, Literal
from functools import wraps, partial
import torch


__field_annotations = OrderedDict()


@dataclass
class FieldAnnotation:
    """
    Attributes annotated with `field` are linked to PLY file elements and attributes
    to automatically produce the implementation of following BasicGeometry methods: TODO

    A number of parameters could be specified to clarify the field usage and enable
    some functionality

    Parameters
    ----------
    element: str
        The name of PLY element this field corresponds to.
        Multiple fields corresponding to the same PLY element can be created.
        However, note that they must have the same length.
    props: list[str]
        Properties of PLY element that form this field. Must have the same data type.
    list_t: bool = False
        Whether this element of list type. Must be set for declaring mesh faces field.
    required: bool = False
        If this field is required in geometry constructor
    dtype: torch.dtype = None
        If set, dtype is checked during geometry construction, loading and saving.
    cast_fn: Callable | 'cast' = None
        Function with `(data: torch.Tensor, target_dtype: torch.dtype) -> torch.Tensor`
        semantics or `cast` literal that casts input `data` to `target_dtype` dtype.
        If `cast` literal is provided,
        Called only if given tensor dtype does not correspond to the `dtype`.
        Ignored if `cast_fn`=None or `dtype`=None.
    index_of: str = None
        If provided, this field is treated as indices of element specified in this field.
        Influences concatenation of objects.
        E.g. setting `index_of=vertex` for `faces` field will ensure correct enumeration
        of vertices in faces after concatenation.
    """

    element: str
    props: list[str]
    list_t: Optional[bool] = False
    required: Optional[bool] = False
    dtype: Optional[torch.dtype] = None
    cast_fn: Callable | Literal['cast'] = None
    index_of: Optional[str] = None

    def __post_init__(self):
        if self.cast_fn == 'cast':
            self.cast_fn = lambda x, target_dtype: x.to(dtype=target_dtype)


@wraps(partial(FieldAnnotation.__init__, None))
def field(*args, **kwargs):
    return Annotated[torch.Tensor, FieldAnnotation(*args, **kwargs)]

field.__doc__ = FieldAnnotation.__doc__


def vertex_field(*args, list_t: bool = False, required: bool = False, dtype=None):
    return field('vertex', props=args,
                 list_t=list_t, required=required, dtype=dtype)


def _gather_annotations(cls, result):
    if '__annotations__' in cls.__dict__:
        cur_anno = cls.__dict__['__annotations__']
        for field_name, anno in reversed(cur_anno.items()):
            if get_origin(anno) is Annotated:
                result.append((field_name, get_args(anno)[1]))

    for base_cls in cls.__bases__:
        if base_cls != object:
            _gather_annotations(base_cls, result)


def gather_annotations(cls):
    if cls not in __field_annotations:
        result = []
        _gather_annotations(cls, result)
        __field_annotations[cls] = result[::-1]

    return __field_annotations[cls]


def get_fields(cls) -> OrderedDict[str, FieldAnnotation]:
    """
    Get all fields (including inherited from parent class) declared
    for specific class with `field`-annotated attributes.

    Parameters
    ----------
    cls: type
        Class type

    Returns
    -------
    OrderedDict[str, FieldAnnotation]:
        Dictionary with field names as keys and `FieldAnnotation` values
    """
    return OrderedDict(gather_annotations(cls))
