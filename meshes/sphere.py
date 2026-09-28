import os
import meshio
import numpy as np
from netgen.csg import CSGeometry, Sphere, Pnt
from ngsolve import Mesh

script_dir = os.path.dirname(os.path.abspath(__file__))

name_vol = "sphere.vol"
name_msh = "sphere.msh"
name_vtu = "sphere.vtu"
h = 0.0133

# Geometry
geo = CSGeometry()
geo.Add(Sphere(Pnt(0, 0, 0), 1.0).bc("sphere_surface"))

# Generate surface mesh only (dim=2)
ngmesh = geo.GenerateMesh(
    maxh=h,
    perfstepsend=3,   # stop after surface meshing (skip volume)
    grading=0.3,
)

# # Save .vol file (Netgen native format)
# oPath_vol = os.path.join(script_dir, name_vol)
# ngmesh.Save(oPath_vol)

# Count using available attributes
pts_list = list(ngmesh.Points())
el2d_list = list(ngmesh.Elements2D())
# print(f"VOL  file written to: {oPath_vol}")
print(f"  Vertices         : {len(pts_list)}")
print(f"  Surface elements : {len(el2d_list)}")
print(f"  Volume elements  : {ngmesh.ne}")

# Extract points
points = np.array([[p[0], p[1], p[2]] for p in pts_list])

# Extract surface triangles (1-indexed → 0-indexed)
triangles = np.array([[v.nr - 1 for v in el.vertices] for el in el2d_list])

cells = [("triangle", triangles)]
mio_mesh = meshio.Mesh(points=points, cells=cells)

# Write .msh (Gmsh 2.2 format)
oPath_msh = os.path.join(script_dir, name_msh)
meshio.write(oPath_msh, mio_mesh, file_format="gmsh22")
print(f"MSH  file written to: {oPath_msh}")

# # Write .vtu for ParaView
# oPath_vtu = os.path.join(script_dir, name_vtu)
# meshio.write(oPath_vtu, mio_mesh)
# print(f"VTU  file written to: {oPath_vtu}")