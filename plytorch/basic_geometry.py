from collections import defaultdict
from typing import Optional

import torch

from .annotations import get_fields, FieldAnnotation
from .plydata import PLYData


class BasicGeometry:
    """
    A base class for 3D geometry representations.

    This class provides basic functionality for handling 3D point cloud data,
    including methods for data conversion, device transfer, and file I/O.
    """

    def __init__(self, **kwargs):
        fields = get_fields(self.__class__)

        for field_name, field in fields.items():
            if field_name not in kwargs:
                if field.required:
                    raise ValueError(
                        "Field '{}' is required, but not given to '{}' constructor".format(
                            field_name, self.__class__.__name__
                        )
                    )
                else:
                    setattr(self, field_name, None)

        for key, value in kwargs.items():
            if key not in fields.keys():
                raise ValueError(
                    'Unknown attribute "{}" for class "{}". '
                    'Existing attributes are: '.format(
                    key, self.__class__.__name__, list(fields.keys()))
                )
            setattr(self, key, value)

    def __setattr__(self, key, value):
        if key.startswith('_'):
            super().__setattr__(key, value)
            return

        fields = get_fields(self.__class__)

        # Check if such field exists
        if key not in fields:
            raise ValueError('Can not set attribute "{}" for class "{}", '
                             'since no such fields exist.'
                             'Existing fields are: {}'.format(
                key, self.__class__.__name__, list(fields.keys())
            ))

        # Check if value is required
        if value is None:
            if fields[key].required:
                raise ValueError('Can not set field "{}" to None for class "{}". '
                                 'Field is required.'.format(
                    key, self.__class__.__name__))
            else:
                super().__setattr__(key, value)
                return

        # Check if value is tensor
        if not isinstance(value, torch.Tensor):
            raise ValueError('Can not set field "{}" to {}, '
                             'since value is not a valid torch.Tensor'.format(key, value))

        # Check if it has different number of elements
        cur_element = fields[key].element

        for field_name, field in fields.items():
            if hasattr(self, field_name):
                field_value = getattr(self, field_name)
                if (field.element == cur_element) and field_value is not None:
                    if len(value) != len(field_value):
                        raise ValueError('Can not set field "{}" for class "{}", '
                                         'conflicting number of elements. '
                                         'Got {} elements, but field "{}" already has '
                                         '{} elements.'.format(
                            key, self.__class__.__name__, len(value),
                            field_name, len(field_value)
                        ))

        # Check dtypes and perform type casting if needed
        if fields[key].dtype is not None:
            if value.dtype != fields[key].dtype:
                if fields[key].cast_fn is not None:
                    # perform cast
                    value = fields[key].cast_fn(value, fields[key].dtype)
                else:
                    raise ValueError("Class '{}', field '{}': value has '{}' dtype, "
                                     "but field requires '{}' dtype".format(
                        self.__class__.__name__, key, value.dtype, fields[key].dtype
                    ))

        # Everything is ok, set attribute
        super().__setattr__(key, value)

    def to(self, device: torch.device | str):
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
        field_names = list(get_fields(self.__class__).keys())

        return self.__class__(
            **{
                field_name:
                    getattr(self, field_name).to(device)
                    if getattr(self, field_name) is not None
                    else None
                for field_name in field_names
            }
        )

    def cuda(self, device_index: Optional[int] = None):
        """
        Move the geometry to CUDA device.

        Returns
        -------
        BasicGeometry
            A new instance of the geometry with data on CUDA device.
        """
        return self.to('cuda:{}'.format(device_index)
                       if device_index is not None
                       else 'cuda')

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
        fields = get_fields(cls)
        result = {}

        for field_name, field in fields.items():
            if ((getattr(data, field.element) is not None)
                    and (getattr(data, field.element)[field.props] is not None)):
                result[field_name] = getattr(data, field.element)[field.props]
            else:
                if field.required:
                    raise ValueError(
                        "Field '{}' is required for class '{}', but cannot be loaded. "
                        "Requested from element '{}', property '{}'".format(
                            field_name, cls.__name__, field.element, field.props
                        )
                    )
        return result

    def to_data(self):
        fields = get_fields(self.__class__)
        result = defaultdict(dict)

        for field_name, field in fields.items():
            if hasattr(self, field_name) and (getattr(self, field_name) is not None):
                result[field.element] |= self._split(field_name, field.props)
            else:
                if field.required:
                    raise ValueError("Field '{}' is required for class '{}', "
                                     "but instance has no valid value for it.".format(
                        field_name, self.__class__.__name__
                    ))

        return result

    def __repr__(self):
        class_name = self.__class__.__name__
        fields = get_fields(self.__class__)
        repr_str = '{}:\n'.format(class_name)
        for field_name, field_annot in fields.items():
            repr_str += '  ' + field_name

            if (not hasattr(self, field_name)) or (getattr(self, field_name) is None):
                repr_str += ' (None)\n'
            else:
                cur_field = getattr(self, field_name)
                repr_str += ' (shape: {}, dtype: {}, device: {})\n'.format(
                    list(cur_field.shape),
                    str(cur_field.dtype).split('.')[1],
                    cur_field.device
                )
        return repr_str

    @staticmethod
    def cat(objects):
        result = {}
        target_class = objects[0].__class__

        for obj in objects:
            if obj.__class__ != target_class:
                raise ValueError('Concatenation got mixed object types. '
                                 'E.g. {} and {}'.format(
                    obj.__class__.__name__, target_class.__name__))

        for field_name, field in get_fields(target_class).items():
            if getattr(objects[0], field_name) is None:
                continue

            result[field_name] = torch.cat(
                [getattr(obj, field_name) for obj in objects], dim=0)

            if field.index_of is not None:
                index_field_name = None

                for fn, f in get_fields(target_class).items():
                    if f.element == field.index_of and f.required:
                        index_field_name = fn

                index_sizes = torch.as_tensor([0] + [
                    len(getattr(obj, index_field_name)) for obj in objects
                ]).int()
                field_sizes = torch.as_tensor([
                    len(getattr(obj, field_name)) for obj in objects
                ]).int()
                offsets = torch.cumsum(index_sizes, dim=0)[:-1]
                offsets = offsets.repeat_interleave(field_sizes).view(-1, 1)
                result[field_name] += offsets

        return target_class(**result)
