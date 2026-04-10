"""Structured quadrilateral mesh generation for the axisymmetric cylinder domain.

Wraps gmsh to produce an orthogonal quad mesh on the r-z half-domain
[0, Re] × [0, H/2].  Elements are graded geometrically so the finest
resolution is concentrated near the lateral boundary (r=Re) and the
loaded top face (z=H/2), where stress and pressure gradients are largest.

Boundary physical groups:
    bottom = 1  (z = 0,   symmetry plane)
    right  = 2  (r = Re,  lateral boundary)
    top    = 3  (z = H/2, loaded face)
    left   = 4  (r = 0,   axis of symmetry)
"""

import numpy as np
import gmsh
from mpi4py import MPI
from dolfinx.io import gmsh as gmshio

from config import CONFIG


# ---------------------------------------------------------------------------
# Mesh
# ---------------------------------------------------------------------------

class CylinderMesh:
    """Axisymmetric (r, z) cylinder mesh using structured quadrilateral elements.

    Generates a graded orthogonal quad mesh via gmsh.  Refinement is applied
    near the lateral boundary (r=Re) and near the loaded face (z=H_mesh),
    where stress gradients are sharpest.

    Boundary tags: bottom=1, right=2, top=3, left=4.
    The mesh spans z ∈ [0, H/2] — only the half-specimen (symmetry at z=0).
    """

    def __init__(self, comm=MPI.COMM_WORLD):
        self.mesh_cfg = CONFIG.mesh
        self.comm     = comm
        self.domain, self.facets = self._create_gmsh_mesh()

    #
    #
    def _create_gmsh_mesh(self):
        """Build orthogonal quad mesh with geometric grading.

        Grading strategy (applied independently in r and z):
          - Refined zone: N//2 elements covering L/6, with geometric spacing
            starting at h_min = L/(N·10) and growing by ratio r each step.
          - Coarse zone: remaining N//2 elements covering 5L/6 uniformly.
          - r is found by bisection so the refined zone sums exactly to L/6.
          - reverse=True flips the array so refinement is at the far end
            (r=Re or z=H_mesh) rather than the origin.
        """
        cfg    = self.mesh_cfg
        Re     = cfg.Re
        H_mesh = cfg.H / 2   # mesh spans half the specimen height
        N_r    = cfg.Nr if cfg.Nr is not None else cfg.N
        N_z    = cfg.Nz if cfg.Nz is not None else cfg.N

        def _geometric_ratio(h_min, n_ref, L_ref, tol=1e-12):
            """Find geometric ratio r so that h_min*(r^n_ref - 1)/(r-1) = L_ref."""
            lo, hi = 1.0 + 1e-10, 1e6
            for _ in range(200):
                mid = (lo + hi) / 2.0
                try:
                    s = h_min * (mid**n_ref - 1.0) / (mid - 1.0)
                except OverflowError:
                    s = float('inf')
                lo, hi = (mid, hi) if s < L_ref else (lo, mid)
                if hi - lo < tol:
                    break
            return (lo + hi) / 2.0

        def _make_coords(length, N, graded=True, reverse=False):
            """Build 1-D coordinate array of N+1 nodes over [0, length].

            If graded=True, applies geometric refinement in the first L/6 of
            the interval.  If reverse=True, the fine zone is at the far end.
            If graded=False, returns uniform spacing regardless of reverse.
            """
            if not graded:
                return np.linspace(0.0, length, N + 1)

            n_ref = N // 2
            n_crs = N - n_ref
            h_min = length / (N * 10)
            L_ref = length / 6.0
            h_crs = (length - L_ref) / n_crs
            r     = _geometric_ratio(h_min, n_ref, L_ref)

            coords = [0.0]
            h = h_min
            for _ in range(n_ref):
                coords.append(coords[-1] + h)
                h *= r
            coords[n_ref] = L_ref          # snap refined zone end exactly
            for _ in range(n_crs):
                coords.append(coords[-1] + h_crs)
            coords[-1] = length            # snap final point exactly

            arr = np.array(coords)
            return length - arr[::-1] if reverse else arr

        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("CylinderMesh")
        geom = gmsh.model.geo

        # r: graded near r=Re by default; z: graded near z=0 or z=H_mesh per config
        x_coords = _make_coords(Re,     N_r, graded=cfg.grade_r,        reverse=True)
        y_coords = _make_coords(H_mesh, N_z, graded=True,               reverse=not cfg.grade_z_bottom)

        n_x = len(x_coords) - 1
        n_y = len(y_coords) - 1

        # Points — indexed by (i, j) on the structured grid
        points = {
            (i, j): geom.add_point(x_coords[i], y_coords[j], 0,
                                   tag=j * (n_x + 1) + i + 1)
            for j in range(n_y + 1)
            for i in range(n_x + 1)
        }

        # Lines — horizontal (h_lines) and vertical (v_lines), keyed by lower-left corner
        tag = 1
        h_lines, v_lines = {}, {}
        for j in range(n_y + 1):
            for i in range(n_x):
                h_lines[(i, j)] = geom.add_line(points[(i, j)], points[(i+1, j)], tag=tag); tag += 1
        for i in range(n_x + 1):
            for j in range(n_y):
                v_lines[(i, j)] = geom.add_line(points[(i, j)], points[(i, j+1)], tag=tag); tag += 1

        # Surfaces — one quad per (i,j) cell; loop order (j, i) matches gmsh's row-major expectation
        surfaces = []
        for surf_tag, (j, i) in enumerate(
                ((j, i) for j in range(n_y) for i in range(n_x)), start=1):
            loop = geom.add_curve_loop(
                [h_lines[(i, j)], v_lines[(i+1, j)], -h_lines[(i, j+1)], -v_lines[(i, j)]],
                tag=surf_tag)
            surfaces.append(geom.add_plane_surface([loop], tag=surf_tag))

        geom.synchronize()

        # Force transfinite (structured) quads on every surface and line
        for surf in surfaces:
            gmsh.model.mesh.setTransfiniteSurface(surf)
            gmsh.model.mesh.setRecombine(2, surf)
        for line in list(h_lines.values()) + list(v_lines.values()):
            gmsh.model.mesh.setTransfiniteCurve(line, 2)
        gmsh.option.setNumber("Mesh.RecombineAll", 1)

        # Physical groups — marker IDs must match MARKERS in formulation.py
        gmsh.model.addPhysicalGroup(2, surfaces, 1)
        gmsh.model.addPhysicalGroup(1, [h_lines[(i, 0)]    for i in range(n_x)], 1, name="bottom")
        gmsh.model.addPhysicalGroup(1, [v_lines[(n_x, j)]  for j in range(n_y)], 2, name="right")
        gmsh.model.addPhysicalGroup(1, [h_lines[(i, n_y)]  for i in range(n_x)], 3, name="top")
        gmsh.model.addPhysicalGroup(1, [v_lines[(0, j)]    for j in range(n_y)], 4, name="left")

        gmsh.model.mesh.generate(2)
        mesh_data = gmshio.model_to_mesh(gmsh.model, self.comm, 0, gdim=2)
        domain    = mesh_data.mesh
        facets    = mesh_data.facet_tags
        gmsh.finalize()

        if self.comm.rank == 0:
            nc = domain.topology.index_map(2).size_global
            print(f"✓ Mesh: {nc} cells  ({n_x}×{n_y})  "
                  f"Δr=[{np.diff(x_coords).min():.2e}, {np.diff(x_coords).max():.2e}] m  "
                  f"Δz=[{np.diff(y_coords).min():.2e}, {np.diff(y_coords).max():.2e}] m")

        return domain, facets
