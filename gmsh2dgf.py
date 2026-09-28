"""
gmsh2DGF: Convert a meshio-style mesh (points + cells dict) to a DGF format string.

Usage:
    from gmsh2dgf import gmsh2DGF
    dgf_str = gmsh2DGF(points, {"triangle": cells_array})           # 2D
    dgf_str = gmsh2DGF(points, {"triangle": cells_array}, dim=3)    # surface in 3D

Then pass to dune:
    from dune.grid import reader
    domain  = (reader.dgfString, dgf_str)
    gridView = leafGridView(domain, dimgrid=2)              # 2D
    gridView = leafGridView(domain, dimgrid=2, dimworld=3)  # surface in 3D
"""

import numpy as np


def gmsh2DGF(points, cells, bndDomain=None, bndSegments=None, periodic=None, dim=None, computeFirstIndex=True):
    """
    Convert a mesh to DGF format string.

    Parameters
    ----------
    points : array-like
        Array of vertex coordinates, shape (npoints, dim).
    cells : dict
        Dictionary mapping element type to vertex index array.
        e.g. {"triangle": array([[0,1,2], ...])}
        Supported keys: "triangle", "tetra", "quad", "hexa"
    bndDomain : dict, optional
        dict id -> [lower, upper] bounding box OR id -> str for named boundaries.
    bndSegments : dict, optional
        dict id -> list of vertex lists for boundary segments.
    periodic : str, optional
        String containing periodic boundary transformation block.
    dim : int, optional
        Spatial dimension for writing vertex coordinates.
        If None, inferred from cell type:
            "tetra" or "hexa" -> 3
            "triangle" or "quad" -> 2
            else -> len(points[0])
        NOTE: for surface meshes (dimworld=3, dimgrid=2) where cells are
        triangles but points are 3D, you MUST pass dim=3 explicitly,
        otherwise only the first 2 coordinates are written.
    computeFirstIndex : bool
        If True, first vertex index is computed from all cell vertices.
        Default is True (handles both 0-based and 1-based numbering).

    Returns
    -------
    str
        Mesh description in DGF format, suitable for passing to
        (reader.dgfString, dgf_str) in dune.
    """
    if dim is None:
        if "tetra" in cells or "hexa" in cells:
            dim = 3
        elif "quad" in cells or "triangle" in cells:
            dim = 2
        else:
            dim = len(points[0])

    simplex = "triangle" if "triangle" in cells else None
    if dim == 3 and "tetra" in cells:
        simplex = "tetra"

    # Compute first vertex index (handles 0-based and 1-based numbering)
    firstVertexIndex = 0
    if computeFirstIndex:
        for key, cellVertices in cells.items():
            firstVertexIndex = min(firstVertexIndex, int(np.min(cellVertices)))

    dgf = "DGF\nVertex\n"

    # If first vertex index is not 0, declare it
    if firstVertexIndex > 0:
        dgf += f"firstindex {firstVertexIndex}\n"

    # Write vertex coordinates (only first `dim` coordinates per point)
    for p in points:
        for i in range(dim):
            dgf += str(p[i]) + " "
        dgf += "\n"
    dgf += "#\n\n"

    # Write simplex elements (triangles or tetrahedra)
    if simplex is not None:
        dgf += "Simplex\n"
        for t in cells[simplex]:
            for v in t:
                dgf += str(v) + " "
            dgf += "\n"
        dgf += "#\n\n"

    # Write quad elements (with gmsh->DGF vertex reordering)
    if "quad" in cells:
        dgf += "Cube\n"
        vxmap = [0, 1, 3, 2]   # gmsh has different reference quad orientation
        for t in cells["quad"]:
            for i in range(4):
                dgf += str(t[vxmap[i]]) + " "
            dgf += "\n"
        dgf += "#\n\n"

    # Write boundary segments
    if bndSegments is not None:
        assert isinstance(bndSegments, dict), \
            "bndSegments must be a dictionary {id: list_of_segments}"
        dgf += "BoundarySegments\n"
        for bndid, bndsegs in bndSegments.items():
            for segment in bndsegs:
                dgf += str(bndid)
                for vx in segment:
                    dgf += " " + str(vx)
                dgf += "\n"
        dgf += "#\n\n"

    # Write boundary domain
    if bndDomain is not None:
        assert isinstance(bndDomain, dict), \
            "bndDomain must be a dictionary"
        dgf += "BoundaryDomain\n"
        for bndid, bnd in bndDomain.items():
            if isinstance(bnd, str):
                if bnd == "default":
                    dgf += bnd + " " + str(bndid) + "\n"
                else:
                    dgf += str(bndid) + " " + bnd + "\n"
            else:
                assert isinstance(bnd, (tuple, list)), \
                    "bndDomain value must be str or [lower, upper] pair"
                assert len(bnd) == 2, \
                    "bndDomain value must be [lower_corner, upper_corner]"
                dgf += str(bndid)
                for coord in bnd:
                    assert len(coord) == dim
                    for c in coord:
                        dgf += " " + str(c)
                dgf += "\n"
        dgf += "#\n"

    # Write periodic boundaries
    if periodic is not None:
        dgf += periodic

    return dgf