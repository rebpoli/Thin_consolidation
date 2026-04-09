import gmsh
from mpi4py import MPI
from dolfinx.io import gmsh as gmshio

import config


class CylinderMesh:
    def __init__(self, comm=MPI.COMM_WORLD):
        cfg = config.get()
        
        self.mesh_cfg = cfg.mesh
        self.comm     = comm
        self.domain   = None
        self.facets   = None

        self.generate()
        
    def generate(self):
        self.domain, self.facets = self._create_gmsh_mesh()
        return self.domain, self.facets
    
#     # Homogeneous triangles
#     def _create_gmsh_mesh(self):
#         Re    = self.mesh_cfg.Re
#         H     = self.mesh_cfg.H
#         N     = self.mesh_cfg.N
#         hsize = H / N
#         
#         gmsh.initialize()
#         gmsh.option.setNumber("General.Terminal", 0)
#         
#         gdim       = 2
#         model_rank = 0
#         
#         gmsh.model.add("Cylinder")
#         geom = gmsh.model.geo
#         
#         p1 = geom.add_point(0,  0, 0)
#         p2 = geom.add_point(Re, 0, 0)
#         p3 = geom.add_point(Re, H, 0)
#         p4 = geom.add_point(0,  H, 0)
#         
#         bottom = geom.add_line(p1, p2)
#         right  = geom.add_line(p2, p3)
#         top    = geom.add_line(p3, p4)
#         left   = geom.add_line(p4, p1)
#         
#         boundary = geom.add_curve_loop([bottom, right, top, left])
#         surf     = geom.add_plane_surface([boundary])
#         
#         geom.synchronize()
#         
#         gmsh.option.setNumber("Mesh.CharacteristicLengthMin", hsize)
#         gmsh.option.setNumber("Mesh.CharacteristicLengthMax", hsize)
#         
#         gmsh.model.addPhysicalGroup(gdim, [surf], 1)
#         gmsh.model.addPhysicalGroup(gdim - 1, [bottom], 1, name="bottom")
#         gmsh.model.addPhysicalGroup(gdim - 1, [right],  2, name="right")
#         gmsh.model.addPhysicalGroup(gdim - 1, [top],    3, name="top")
#         gmsh.model.addPhysicalGroup(gdim - 1, [left],   4, name="left")
#         
#         gmsh.model.mesh.generate(gdim)
#         
#         mesh_data = gmshio.model_to_mesh(gmsh.model, self.comm, model_rank, gdim=gdim)
#         domain    = mesh_data.mesh
#         facets    = mesh_data.facet_tags
#         
#         gmsh.finalize()
#         
#         if self.comm.rank == 0:
#             num_cells = domain.topology.index_map(gdim).size_global
#             print(f"✓ Mesh created: {num_cells} cells")
#         
#         return domain, facets

    def _create_gmsh_mesh(self):
        """Create orthogonal quadrilateral mesh with geometric grading.

        Grading rules (applied independently to r and z directions):
          h_min  = L / (N * 10)                     finest element
          n_ref  = N // 2   elements, geometric,    covering L/4  (refined zone)
          n_crs  = N - n_ref elements, uniform,     covering 3L/4 (coarse zone)
          r      = geometric ratio, bisected so sum of refined zone = L/4 exactly
          h_crs  = (3L/4) / n_crs                   uniform coarse size

        Refinement direction:
          r (x): fine near r=Re (right/lateral boundary) — from_end=True
          z (y): fine near z=0  (bottom symmetry plane)  — from_end=False
        """
        import numpy as np

        Re = self.mesh_cfg.Re
        H  = self.mesh_cfg.H
        N  = self.mesh_cfg.N

        def _geometric_ratio(h_min, n_ref, L_ref, tol=1e-12):
            """Bisect for r: h_min*(r^n_ref - 1)/(r-1) = L_ref."""
            lo, hi = 1.0 + 1e-10, 1e6
            for _ in range(200):
                mid = (lo + hi) / 2.0
                s   = h_min * (mid**n_ref - 1.0) / (mid - 1.0)
                if s < L_ref:
                    lo = mid
                else:
                    hi = mid
                if hi - lo < tol:
                    break
            return (lo + hi) / 2.0

        def _make_coords(length, N, from_end=False):
            """Build node coordinates for one direction."""
            n_ref  = N // 2
            n_crs  = N - n_ref
            h_min  = length / (N * 10)
            L_ref  = length / 6.0
            L_crs  = length - L_ref
            h_crs  = L_crs / n_crs

            r = _geometric_ratio(h_min, n_ref, L_ref)

            coords = [0.0]
            h = h_min
            for i in range(n_ref):
                coords.append(coords[-1] + h)
                h *= r
            # Snap end of refined zone exactly to L/4
            coords[n_ref] = L_ref

            for _ in range(n_crs):
                coords.append(coords[-1] + h_crs)
            # Snap final point exactly to length
            coords[-1] = length

            arr = np.array(coords)
            if from_end:
                arr = length - arr[::-1]   # mirror: fine at far end
            return arr

        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)

        gdim       = 2
        model_rank = 0

        gmsh.model.add("OrthogonalQuadMesh")

        H_mesh = H / 2   # model only the top half; H is the full specimen height

        x_coords = _make_coords(Re,     N, from_end=True)    # fine near r=Re
        y_coords = _make_coords(H_mesh, N, from_end=True)   # fine near z=H_mesh (top/loaded face)

        # ========================================================================
        # Create structured grid of points
        # ========================================================================


        geom = gmsh.model.geo
        points = {}

        # Calculate actual number of cells from coordinate arrays
        n_x = len(x_coords) - 1
        n_y = len(y_coords) - 1

        for j in range(len(y_coords)):
            for i in range(len(x_coords)):
                tag = j * (n_x + 1) + i + 1
                points[(i, j)] = geom.add_point(x_coords[i], y_coords[j], 0, tag=tag)

        # ========================================================================
        # Create horizontal lines
        # ========================================================================

        h_lines = {}
        line_tag = 1

        for j in range(n_y + 1):
            for i in range(n_x):
                h_lines[(i, j)] = geom.add_line(points[(i, j)], points[(i+1, j)], tag=line_tag)
                line_tag += 1

        # ========================================================================
        # Create vertical lines
        # ========================================================================

        v_lines = {}

        for i in range(n_x + 1):
            for j in range(n_y):
                v_lines[(i, j)] = geom.add_line(points[(i, j)], points[(i, j+1)], tag=line_tag)
                line_tag += 1

        # ========================================================================
        # Create quadrilateral surfaces
        # ========================================================================

        surfaces = []
        surf_tag = 1

        for j in range(n_y):
            for i in range(n_x):
                bottom = h_lines[(i, j)]
                right = v_lines[(i+1, j)]
                top = h_lines[(i, j+1)]
                left = v_lines[(i, j)]

                loop = geom.add_curve_loop([bottom, right, -top, -left], tag=surf_tag)
                surf = geom.add_plane_surface([loop], tag=surf_tag)
                surfaces.append(surf)
                surf_tag += 1

        geom.synchronize()

        # ========================================================================
        # Set transfinite mesh (1 quad per surface)
        # ========================================================================

        for surf in surfaces:
            gmsh.model.mesh.setTransfiniteSurface(surf)
            gmsh.model.mesh.setRecombine(gdim, surf)

        # Set all curves as transfinite with 2 points
        for line in list(h_lines.values()) + list(v_lines.values()):
            gmsh.model.mesh.setTransfiniteCurve(line, 2)

        # Global recombination option
        gmsh.option.setNumber("Mesh.RecombineAll", 1)

        # ========================================================================
        # Physical groups
        # ========================================================================

        # Collect boundary lines
        bottom_lines = [h_lines[(i, 0)] for i in range(n_x)]
        right_lines = [v_lines[(n_x, j)] for j in range(n_y)]
        top_lines = [h_lines[(i, n_y)] for i in range(n_x)]
        left_lines = [v_lines[(0, j)] for j in range(n_y)]


        # Add physical groups
        gmsh.model.addPhysicalGroup(gdim, surfaces, 1)
        gmsh.model.addPhysicalGroup(gdim - 1, bottom_lines, 1, name="bottom")
        gmsh.model.addPhysicalGroup(gdim - 1, right_lines, 2, name="right")
        gmsh.model.addPhysicalGroup(gdim - 1, top_lines, 3, name="top")
        gmsh.model.addPhysicalGroup(gdim - 1, left_lines, 4, name="left")

        # ========================================================================
        # Generate mesh
        # ========================================================================

        gmsh.model.mesh.generate(gdim)

        # Convert to dolfinx
        mesh_data = gmshio.model_to_mesh(gmsh.model, self.comm, model_rank, gdim=gdim)
        domain = mesh_data.mesh
        facets = mesh_data.facet_tags

#         gmsh.fltk.run()
        gmsh.finalize()

        # ========================================================================
        # Verification
        # ========================================================================

        if self.comm.rank == 0:
            num_cells = domain.topology.index_map(gdim).size_global
            cell_type = domain.topology.cell_type

            print(f"✓ Mesh created: {num_cells} cells")
            print(f"  Cell type: {cell_type}")
            h_min_x = np.diff(x_coords).min()
            h_max_x = np.diff(x_coords).max()
            h_min_y = np.diff(y_coords).min()
            h_max_y = np.diff(y_coords).max()
            print(f"  Grid: {n_x} × {n_y}  (N={N})")
            print(f"  R spacing: {h_min_x:.3e} … {h_max_x:.3e} m  (ratio {h_max_x/h_min_x:.1f}×)")
            print(f"  Z spacing: {h_min_y:.3e} … {h_max_y:.3e} m  (ratio {h_max_y/h_min_y:.1f}×)")

            from dolfinx.mesh import CellType
            if cell_type == CellType.quadrilateral:
                print(f"  ✓ Quadrilateral mesh")
            else:
                print(f"  ⚠ Warning: {cell_type}")

        return domain, facets
