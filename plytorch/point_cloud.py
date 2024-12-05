import torch

from .annotations import field, vertex_field
from .basic_geometry import BasicGeometry


class PointCloud(BasicGeometry):
    points: vertex_field('x', 'y', 'z', required=True)
    normals: vertex_field('nx', 'ny', 'nz')
    colors: vertex_field('red', 'green', 'blue', dtype=torch.uint8)
    uv: vertex_field('s', 't')

    @property
    def num_vertices(self):
        return len(self.points)


class Mesh(PointCloud):
    faces: field('face', 'vertex_index', list_t=True, required=True,
                 index_of='vertex', dtype=torch.int32, cast_fn='cast')

    @property
    def num_faces(self):
        return len(self.faces)


class Lines(PointCloud):
    edges: field('edge', ['vertex_1', 'vertex_2'], required=True, index_of='vertex')

    @property
    def num_edges(self):
        return len(self.edges)
