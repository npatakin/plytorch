from dataclasses import dataclass
import torch

from .plydata import PLYData
from .basic_geometry import BasicGeometry


@dataclass
class PointCloud(BasicGeometry):
    normals: torch.Tensor | None = None
    colors: torch.Tensor | None = None
    uv: torch.Tensor | None = None

    @classmethod
    def from_data(cls, data: PLYData):
        result = BasicGeometry.from_data(data)

        return result | dict(
            normals=data.vertex['nx', 'ny', 'nz'],
            colors=data.vertex['red', 'green', 'blue'],
            uv=data.vertex['s', 't']
        )

    def to_data(self):
        result = super().to_data()

        return dict(
            vertex=(
                    result['vertex'] |
                    self._split('normals', ['nx', 'ny', 'nz']) |
                    self._split('colors', ['red', 'green', 'blue']) |
                    self._split('uv', ['s', 't'])
            )
        )


@dataclass
class Mesh(PointCloud):
    faces: torch.Tensor = None

    @property
    def num_faces(self):
        return len(self.faces)

    @classmethod
    def from_data(cls, data: PLYData):
        result = PointCloud.from_data(data)
        if (data.face is None) or (data.face.vertex_index is None):
            raise RuntimeError("Loaded file is not a valid mesh, since it has no element 'face'"
                               " with property 'vertex_index'. Use PointCloud instead to load vertex data.")
        return result | dict(faces=data.face.vertex_index)

    def to_data(self):
        if self.faces is None:
            raise RuntimeError("Mesh has no faces. Unable to properly serialize data.")
        result = super().to_data()
        return dict(vertex=result['vertex'], face=dict(vertex_index=self.faces))
