from collections import defaultdict
from typing import Annotated, get_origin, get_args
import torch

from .plydata import PLYData


def field(element: str, props: list[str] | str, list_t: bool = False, required: bool = False, dtype=None):
    return Annotated[torch.Tensor, element, props, list_t, required, dtype]


def vertex_field(*args, list_t: bool = False, required: bool = False, dtype=None):
    return Annotated[torch.Tensor, 'vertex', args, list_t, required, dtype]

__field_annotations = {}


def _gather_annotations(cls, result):
    if '__annotations__' in cls.__dict__:
        cur_anno = cls.__dict__['__annotations__']
        for field_name, anno in reversed(cur_anno.items()):
            if get_origin(anno) is Annotated:
                result.append((field_name, get_args(anno)[1:]))

    for base_cls in cls.__bases__:
        if base_cls != object:
            _gather_annotations(base_cls, result)


def gather_annotations(cls):
    if cls not in __field_annotations:
        result = []
        _gather_annotations(cls, result)
        __field_annotations[cls] = result[::-1]

    return __field_annotations[cls]


class BasicGeometry:
    """
    A base class for 3D geometry representations.

    This class provides basic functionality for handling 3D point cloud data,
    including methods for data conversion, device transfer, and file I/O.
    """

    def __init__(self, **kwargs):
        annots = dict(gather_annotations(self.__class__))
        for field_name, (element_name, props, is_list, required, dtypes) in annots.items():
            if field_name not in kwargs:
                if required:
                    raise ValueError("Field '{}' is required, but not given to {} constructor".format(
                        field_name, self.__class__.__name__
                    ))
                else:
                    setattr(self, field_name, None)

        for key, value in kwargs.items():
            if key not in annots.keys():
                raise ValueError(
                    'Unknown attribute "{}" for class "{}". Existing attributes are: '.format(
                    key, self.__class__.__name__, list(annots.keys()))
                )
            setattr(self, key, value)

    def to(self, device: torch.device):
        """
        Move the geometry to the specified device.

        Parameters
        ----------
        device : torch.device
            The target device to move the data to.

        Returns
        -------
        BasicGeometry
            A new instance of the geometry with data on the specified device.
        """
        field_lst = [f[0] for f in gather_annotations(self.__class__)]

        return self.__class__(
            **{
                field_name:
                    getattr(self, field_name).to(device)
                    if getattr(self, field_name) is not None
                    else None
                for field_name in field_lst
            }
        )

    def cuda(self):
        """
        Move the geometry to CUDA device.

        Returns
        -------
        BasicGeometry
            A new instance of the geometry with data on CUDA device.
        """
        return self.to('cuda')

    def cpu(self):
        """
        Move the geometry to CPU.

        Returns
        -------
        BasicGeometry
            A new instance of the geometry with data on CPU.
        """
        return self.to('cpu')

    def _split(self, name, prop_names):
        t = getattr(self, name)
        if t is None:
            return dict()
        if isinstance(prop_names, str):
            return {prop_names: t}
        props = t.unbind(dim=-1)
        return {k: v for k,v in zip(prop_names, props)}

    @classmethod
    def load(cls, path: str):
        """
        Load geometry from a PLY file.

        Parameters
        ----------
        path : str
            The file path to load the PLY data from.

        Returns
        -------
        BasicGeometry
            An instance of the geometry loaded from the file.
        """
        return cls(**cls.from_data(PLYData.load(path)))

    def save(self, path: str):
        """
        Save the geometry to a PLY file.

        Parameters
        ----------
        path : str
            The file path to save the PLY data to.
        """
        PLYData(**self.to_data()).save(path)

    @classmethod
    def from_data(cls, data: PLYData):
        annots = gather_annotations(cls)
        result = {}
        for field_name, (element_name, props, is_list, required, dtypes) in annots:
            if (getattr(data, element_name) is not None) and (getattr(data, element_name)[props] is not None):
                result[field_name] = getattr(data, element_name)[props]
            else:
                if required:
                    raise ValueError(
                        "Field '{}' is required for class '{}', but cannot be loaded. "
                        "Requested from element '{}', property '{}'".format(
                            field_name, cls.__name__, element_name, props
                        )
                    )
        return result

    def to_data(self):
        annots = gather_annotations(self.__class__)
        result = defaultdict(dict)

        for field_name, (element_name, props, is_list, required, dtypes) in annots:
            if hasattr(self, field_name) and (getattr(self, field_name) is not None):
                result[element_name] |= self._split(field_name, props)
            else:
                if required:
                    raise ValueError("Field '{}' is required for class '{}', "
                                     "but instance has no valid value for it.".format(
                        field_name, self.__class__.__name__
                    ))

        return result

    def __repr__(self):
        class_name = self.__class__.__name__
        fields = gather_annotations(self.__class__)
        repr_str = '{}:\n'.format(class_name)
        for field_name, field_annot in fields:
            repr_str += '  ' + field_name

            if (not hasattr(self, field_name)) or (getattr(self, field_name) is None):
                repr_str += ' (None)\n'
            else:
                cur_field = getattr(self, field_name)
                repr_str += ' (shape: {}, dtype: {})\n'.format(cur_field.shape, cur_field.dtype)
        return repr_str
