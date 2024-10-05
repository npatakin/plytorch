from .basic_geometry import BasicGeometry, field, vertex_field


class PointCloud(BasicGeometry):
    points: vertex_field('x', 'y', 'z', required=True)
    normals: vertex_field('nx', 'ny', 'nz')
    colors: vertex_field('red', 'green', 'blue')
    uv: vertex_field('s', 't')

    @property
    def num_vertices(self):
        return len(self.points)


class Mesh(PointCloud):
    faces: field('face', 'vertex_index', list_t=True, required=True)

    @property
    def num_faces(self):
        return len(self.faces)
