import os

import torch
import _plytorch_extension as pte


class PLYElement(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, props: dict[str, torch.Tensor]):
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
            return self[x]

    def __repr__(self):
        props = self.properties
        return 'PLYElement ({} properties). Properties: {}'.format(len(self)-1, ', '.join(props))


class PLYData(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    @property
    def elements(self):
        return sorted(self.keys())

    @staticmethod
    def load(path: str):
        if not os.path.isfile(path):
            raise FileNotFoundError('File not found: "{}"'.format(path))
        return PLYData({name: PLYElement(props) for name, props in pte.read_ply(path).items()})

    def save(self, path: str):
        if not os.path.isdir(os.path.dirname(path)):
            raise FileNotFoundError("Parent directory does not exist for path: '{}'".format(path))

        pte.write_ply(
            path, {
                element_name: {
                    prop_name: prop.cpu().contiguous()
                    for prop_name, prop in element.items()
                }
                for element_name, element in self.items()
            }
        )
