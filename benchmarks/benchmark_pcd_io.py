import time

import numpy as np
import torch

import plytorch as pt
import trimesh
import open3d as o3d
import plyfile as pf


class Open3DTest:
    def __init__(self, name: str = 'open3d', load_faces: bool = False):
        self.name = name
        self.load_faces = load_faces

    def write(self, points, colors, normals, path, faces=None):
        if self.load_faces:
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(points)
            mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
            mesh.vertex_normals = o3d.utility.Vector3dVector(normals)
            mesh.triangles = o3d.utility.Vector3iVector(faces)
            o3d.io.write_triangle_mesh(path, mesh)
        else:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(colors)
            pcd.normals = o3d.utility.Vector3dVector(normals)
            o3d.io.write_point_cloud(path, pcd)

    def read(self, path):
        if self.load_faces:
            mesh: o3d.geometry.TriangleMesh = o3d.io.read_triangle_mesh(path)
            return dict(
                points=torch.as_tensor(np.asarray(mesh.vertices)),
                colors=torch.as_tensor(np.asarray(mesh.vertex_colors)),
                normals=torch.as_tensor(np.asarray(mesh.vertex_normals)),
                faces=torch.as_tensor(np.asarray(mesh.triangles))
            )
        else:
            pcd: o3d.geometry.PointCloud = o3d.io.read_point_cloud(path)
            return dict(
                points=torch.as_tensor(np.asarray(pcd.points)),
                colors=torch.as_tensor(np.asarray(pcd.colors)),
                normals=torch.as_tensor(np.asarray(pcd.normals))
            )


class TrimeshTest:
    def __init__(self, name: str = 'trimesh', load_faces=False, process=True):
        self.name = name
        self.load_faces = load_faces
        self.process = process

    def write(self, points, colors, normals, path, faces=None):
        mesh = trimesh.Trimesh(points, faces) if self.load_faces else trimesh.Trimesh(points)
        mesh.visual.vertex_colors = colors
        mesh.vertex_normals = normals
        mesh.export(path)

    def read(self, path):
        mesh: trimesh.Trimesh = trimesh.load_mesh(path, process=self.process)

        result = dict(
            points=torch.as_tensor(np.asarray(mesh.vertices)),
            colors=torch.as_tensor(np.asarray(mesh.visual.vertex_colors)),
        )
        if self.load_faces:
            result['faces'] = torch.as_tensor(np.asarray(mesh.triangles))

        # normals = torch.as_tensor(np.asarray(mesh.vertex_normals))
        return result


class PlyfileTest:
    def __init__(self, name: str = 'plyfile', load_faces: bool = False):
        self.name = name
        self.load_faces = load_faces

    def write(self, points, colors, normals, path, faces=None):
        vertex_dtype = [
            ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
        ]
        vertex_arr = np.empty(len(points), dtype=vertex_dtype)
        vertex_arr['x'] = points[:, 0]
        vertex_arr['y'] = points[:, 1]
        vertex_arr['z'] = points[:, 2]
        vertex_arr['red'] = colors[:, 0]
        vertex_arr['green'] = colors[:, 1]
        vertex_arr['blue'] = colors[:, 2]
        vertex_arr['nx'] = normals[:, 0]
        vertex_arr['ny'] = normals[:, 1]
        vertex_arr['nz'] = normals[:, 2]

        elements = [pf.PlyElement.describe(vertex_arr, 'vertex')]

        if self.load_faces:
            face_arr = np.empty(len(faces), dtype=[('vertex_index', 'i4', 3)])
            face_arr['vertex_index'] = faces
            elements.append(pf.PlyElement.describe(face_arr, 'face'))

        with open(path, 'wb') as f:
            pf.PlyData(elements).write(f)

    def read(self, path):
        with open(path, 'rb') as f:
            data = pf.PlyData.read(f)
            points = torch.as_tensor(np.stack([data['vertex']['x'], data['vertex']['y'], data['vertex']['z']], axis=1))
            colors = torch.as_tensor(
                np.stack([data['vertex']['red'], data['vertex']['green'], data['vertex']['blue']], axis=1))
            normals = torch.as_tensor(
                np.stack([data['vertex']['nx'], data['vertex']['ny'], data['vertex']['nz']], axis=1))
            faces = torch.as_tensor(np.stack(data['face']['vertex_index']))

        return dict(points=points, colors=colors, normals=normals, faces=faces)


class PlytorchTest:
    def __init__(self, name: str = 'plytorch', load_faces: bool = False):
        self.name = name
        self.load_faces = load_faces

    def write(self, points, colors, normals, path, faces=None):
        if self.load_faces:
            pt.Mesh(points=points, colors=colors, normals=normals, faces=faces).save(path)
        else:
            pt.PointCloud(points=points, colors=colors, normals=normals).save(path)

    def read(self, path):
        return pt.Mesh.load(path) if self.load_faces else pt.PointCloud.load(path)


test_meshes = True
N = 10_000_000
M = 1_000_000
nrepeats = 10

tests = [
    Open3DTest(load_faces=test_meshes),
    TrimeshTest(name='trimesh(process=False)', load_faces=test_meshes, process=False),
    # TrimeshTest(name='trimesh', load_faces=test_meshes, process=True),
    PlyfileTest(load_faces=test_meshes),
    PlytorchTest(load_faces=test_meshes)
]

kwargs = dict(
    points = torch.rand(N, 3),
    normals = torch.rand(N, 3),
    colors = torch.randint(0, 255, (N, 3)).byte(),
)
if test_meshes:
    kwargs['faces'] = torch.randint(0, N, (M, 3), dtype=torch.uint32)

save_path = 'write_test.ply'
read_path = 'read_test.ply'

if test_meshes:
    pt.Mesh(**kwargs).save(read_path)
else:
    pt.PointCloud(**kwargs).save(read_path)

for test in tests:
    s = time.time()
    for i in range(nrepeats):
        result = test.read(read_path)
    t = time.time()
    print('Read {}: {:.2f}ms'.format(test.name, (t-s)*1000.0 / nrepeats))



for test in tests:
    s = time.time()
    for i in range(nrepeats):
        test.write(**kwargs, path=save_path)
    t = time.time()
    print('Write {}: {:.2f}ms'.format(test.name, (t-s)*1000 / nrepeats))
