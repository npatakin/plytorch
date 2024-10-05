import os
from collections import OrderedDict

import torch
import _plytorch_extension as pte

"""
PLYData: Fast loading and writing of .ply files with direct PyTorch interface.

This module provides functionality for efficient handling of PLY (Polygon File Format) files,
with seamless integration with PyTorch tensors. It allows for quick reading and writing of
3D point cloud data and associated properties.

The module uses a custom C++ extension (_plytorch_extension) for optimized I/O operations,
while presenting a Pythonic interface for ease of use.

Classes:
    PLYElement: Represents a single element in a PLY file (e.g., vertex, face).
    PLYData: Main class for loading, manipulating, and saving PLY files.

Functions:
    PLYData.load: Loads a PLY file from a file.
    PLYData.save: Saves PLY data to a file.

Example:
    >>> ply_data = PLYData.load('model.ply')
    >>> ply_data.vertex.x.shape
    torch.Size([1000])
    >>> vertices = ply_data.vertex['x', 'y', 'z']
    >>> vertices.shape
    torch.Size([1000, 3])
    >>> ply_data.face.vertex_index.shape
    torch.Size([2000, 3])
    >>> ply_data.face.vertex_index.dtype
    torch.int32
    >>> ply_data.save('output.ply')

"""


class PLYElement(OrderedDict):
    __getattr__ = OrderedDict.get
    __setattr__ = OrderedDict.__setitem__
    __delattr__ = OrderedDict.__delitem__

    def __init__(self, props: OrderedDict[str, torch.Tensor]):
        super().__init__(props)

    @property
    def properties(self):
        props = list(self.keys())
        return sorted(props)

    def __getitem__(self, x):
        if isinstance(x, list) or isinstance(x, tuple):
            for v in x:
                if v not in self:
                    return None
            return torch.stack([getattr(self, v) for v in x], dim=1)
        else:
            return super().__getitem__(x)

    def __repr__(self):
        props = self.properties
        return 'PLYElement ({} properties). Properties: {}'.format(len(self), ', '.join(props))


class PLYData(OrderedDict):
    __getattr__ = OrderedDict.get
    __setattr__ = OrderedDict.__setitem__
    __delattr__ = OrderedDict.__delitem__

    @property
    def elements(self):
        return sorted(self.keys())

    @staticmethod
    def load(path: str):
        if not os.path.isfile(path):
            raise FileNotFoundError('File not found: "{}"'.format(path))
        return PLYData({name: PLYElement(props) for name, props in pte.read_ply(path)})

    def save(self, path: str):
        if not os.path.isdir(os.path.dirname(os.path.abspath(path))):
            raise FileNotFoundError("Parent directory does not exist for path: '{}'".format(path))

        pte.write_ply(
            path, [
                (element_name, [
                    (prop_name, prop.cpu().contiguous())
                    for prop_name, prop in element.items()
                ])
                for element_name, element in self.items()
            ]
        )

    def __repr__(self):
        repr_str = 'PLYData ({} elements):\n'.format(len(self))
        for key, value in self.items():
            repr_str += '  {}: {}\n'.format(key, str(value))
        return repr_str
