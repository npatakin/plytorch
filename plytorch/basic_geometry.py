from dataclasses import dataclass, fields
import torch

from .plydata import PLYData


@dataclass
class BasicGeometry:
    points: torch.Tensor

    @property
    def num_vertices(self):
        return len(self.points)

    def to(self, device: torch.device):
        field_lst = [field.name for field in fields(self.__class__)]

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
        return self.to('cuda')

    def cpu(self):
        return self.to('cpu')

    def _split(self, name, prop_names):
        t = getattr(self, name)
        if t is None:
            return dict()
        props = t.unbind(dim=-1)
        return {k: v for k,v in zip(prop_names, props)}

    @classmethod
    def load(cls, path: str):
        return cls(**cls.from_data(PLYData.load(path)))

    def save(self, path: str):
        PLYData(**self.to_data()).save(path)

    @staticmethod
    def from_data(data: PLYData):
        return dict(points=data.vertex['x', 'y', 'z'])

    def to_data(self):
        return dict(vertex=self._split('points', ['x', 'y', 'z']))
