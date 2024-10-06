# plytorch

Python library for fast loading and writing arbitrary PLY (Polygon File Format) files that directly interfaces with PyTorch. It provides an efficient way to work with 3D point cloud and mesh data in PyTorch-based applications.

## Features

- Fast loading and writing of PLY files
- Direct integration with PyTorch tensors
- Support for point clouds and meshes
- Flexible handling of elements and their properties, allowing for easy extension to new custom elements.
- Easy-to-use API for working with 3D geometry data

## Installation

You can install plytorch using pip:

```bash
pip install git+https://github.com/npatakin/plytorch.git
```

or by cloning the repository and installing the package locally:

```bash
git clone https://github.com/npatakin/plytorch.git
cd plytorch
pip install .
```





## Quickstart

Here's an example of how to use plytorch to load and save a PLY file:


```python
from plytorch import PointCloud, Mesh

# Load a PLY file
pcd = PointCloud.load('path/to/your/point_cloud.ply')
mesh = Mesh.load('path/to/your/mesh.ply')

print(mesh)

# Save a PLY file
pcd.save('new_point_cloud.ply')
mesh.save('new_mesh.ply')
```

will output:

```
Mesh:
  points (shape: torch.Size([2440, 3]), dtype: torch.float32)
  normals (shape: torch.Size([2440, 3]), dtype: torch.float32)
  colors (None)
  uv (shape: torch.Size([2440, 2]), dtype: torch.float32)
  faces (shape: torch.Size([3588, 3]), dtype: torch.int32)
```

Internally, all the data is stored in PyTorch tensors, so you can access them as `mesh.points`, `mesh.normals`, `mesh.colors`, `mesh.uv`, `mesh.faces`. Point clouds and meshes can be transferred to different devices (cpu/cuda) using conventional `.to(device)`, `.cpu()`, `.cuda()` methods. These methods do not modify the underlying data, but rather return a new structure with data on the specified device.

You can also construct `Mesh` and `PointCloud` directly from Pytorch tensors:

```python
from plytorch import Mesh, PointCloud
import torch

mesh = Mesh(points=torch.randn(1000, 3), faces=torch.randint(0, 1000, (1000, 3)))
mesh.save('new_mesh.ply')
```

# Defining new structures

`.ply` file consists of multiple elements, which in its turn consist of multiple properties. For example, a typical `vertex` element has such properties as `x`, `y`, `z` (vertex coordinates), `nx`, `ny`, `nz` (vertex normals), `red`, `green`, `blue` (vertex colors) and so on. While some of these property names are reserved (like `x`, `y`, `z`), the rest are completely arbitrary. Most of existing python libraries for reading and writing geometry in `.ply` format allows you to interact with these reserved properties only. 

`plytorch`, however, allows you to use all of the available properties and elements, providing a flexible way to work with 3D geometry data. Moreover, `plytorch` allows to group multiple properties into custom fields, as typically you don't want to access them separately. For example, user typically doesn't work with `red`, `green`, `blue` colors separately, but rather prefers to access them as a single `colors` property. 


To define a new structure, you need to subclass `BasicGeometry` or one of the existing structures, and specify the fields by specifying elements and properties they consist of. For example, internally, `PointCloud` and `Mesh` are defined in the following way:

```python

from plytorch import BasicGeometry, field, vertex_field

class PointCloud(BasicGeometry):
    points: vertex_field('x', 'y', 'z', required=True)
    normals: vertex_field('nx', 'ny', 'nz')
    colors: vertex_field('red', 'green', 'blue')
    uv: vertex_field('s', 't')

class Mesh(PointCloud):
    faces: field('face', 'vertex_index', list_t=True, required=True)
```

Creating a new child class of `BasicGeometry` automatically generates all the following methods:
- constructor, which allows to create a new structure from specified field tensors. E.g.: 
    ```python
    PointCloud(
        points=torch.randn(1000, 3), 
        normals=torch.randn(1000, 3), 
        colors=torch.randn(1000, 3), 
        uv=torch.randn(1000, 2)
    )
    ```
    Some of them can be ommited, while other are mandatory. `required=True` passed to `field` means that this field is required, and the constructor will throw an error if it's not provided.
- `load()` and `save()` methods, which load and save data to a PLY file.
- `to()`, `.cpu()`, `.cuda()` methods for transferring the data to a different device.
- Meaningful `str` representation of the structure.

Each field that is planned to be stored in the `.ply` file should be decorated with `field`:
```python
field(element: str, props: list[str] | str, list_t: bool = False, required: bool = False, dtype=None)
``` 
is a decorator that specifies a new field in the structure. 
    - `element` is a name of the element that this field will belong to (e.g. `vertex` for `PointCloud`, `face` for `Mesh`).
    - `props` is a name of the properties that this field will consist of. It can be a single property name, or a list of property names.
    - `list_t` is a flag that specifies if the field is a list of values (e.g. `face` in `Mesh` is a list of indices, so `list_t=True` for `face` field).
    - `required` is a flag that specifies if the field is required. If it's `True`, the constructor, `load()` and `save()` methods will throw an error if the field is not provided.

`vertex_field` is a shortcut for `field(element='vertex', ...)`, so that you don't have to specify `element='vertex'` every time.


# Benchmarks

TODO: add benchmark and comparison description

| Library                                              | Arbitrary properties | Properties grouping | Load time | Lines of Code<br>to `load` | Save time | Lines of Code<br>to `save` |
|------------------------------------------------------|----------------------|---------------------|-----------|----------------------------|-----------|----------------------------|
| [open3d](https://github.com/isl-org/Open3D)          | :x:                  | :white_check_mark:  | 3x        | 6                          | 96x (!)   | 6                          |
| [trimesh](https://github.com/mikedh/trimesh)         | :x:                  | :white_check_mark:  | 1.40x     | 4-5                        | 1.25x     | 4-5                        |
| [plyfile](https://github.com/dranjan/python-plyfile) | :white_check_mark:   | :x:                 | 7x        | 6-7                        | 1.26x     | ~20                        |
| **plytorch**                                         | :white_check_mark:   | :white_check_mark:  | **1.0x**  | **1**                      | **1.0x**  | **1**                      |

# Low-level access

If you want to read all the data from the `.ply` file and minimize the overhead, you can use `PLYData` class:

```python
from plytorch import PLYData

data = PLYData.load('mesh_with_several_uv_maps.ply')
print(data)
```

will give you:
```
PLYData (2 elements):
  vertex: PLYElement (12 properties). Properties: nx, ny, nz, s, s1, s2, t, t1, t2, x, y, z
  face: PLYElement (1 properties). Properties: vertex_index
```

now properties of the ply file can be accessed directly:

```python
data.vertex.s  # will give you a tensor of shape [num_vertices, ] with s values
data.vertex.t  # will give you a tensor of shape [num_vertices, ] with t values
data.face.vertex_index  # will give you a tensor of shape [num_faces, face_size] with mesh faces
```

PLYData is an extension of `OrderedDict`, so you can use all the methods of `OrderedDict` to access the data. As shown above, you can also access the properties of the ply file using the dot notation. It is also convenient to access multiple properties of the ply file at once:

```python
data.vertex['s', 't']  # will give you a tensor of shape [num_vertices, 2] with s and t values
data.vertex['x', 'y', 'z']  # will give you a tensor of shape [num_vertices, 3] with x, y and z values
```

If any of the requested properties are not present in element, it will return `None`. If all the properties are present, but have different dtypes, error will be raised.

All the elements and properties are ordered, since the most 3D viewers (like MeshLab) sensitive to the order of elements (e.g. `vertex` should come before `face`).

# Acknowledgements

- This library is internally uses [miniply](https://github.com/vilya/miniply) library for reading PLY files. Thank you, authors!

