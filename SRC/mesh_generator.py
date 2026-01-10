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
        """Create orthogonal quadrilateral mesh with refinement near right boundary"""
        import numpy as np

        Re = self.mesh_cfg.Re
        H = self.mesh_cfg.H
        N = self.mesh_cfg.N

        # Mesh parameters
        aspect_ratio = 100.0  # dX:dY ratio
        refinement_factor = 0.6  # >1 = finer near left boundary

        n_y = N
        n_x = max(20, int(N / aspect_ratio))

        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)

        gdim = 2
        model_rank = 0

        gmsh.model.add("OrthogonalQuadMesh")

        # ========================================================================
        # Generate node coordinates with geometric progression in X
        # ========================================================================

        def generate_x_coords(n_x, length, r):
            """Generate X coordinates with geometric progression"""
            if abs(r - 1.0) < 1e-10:
                return np.linspace(0, length, n_x + 1)

            x = np.zeros(n_x + 1)
            x[0] = 0.0
            dx_0 = length * (r - 1) / (r**n_x - 1)

            for i in range(n_x):
                x[i + 1] = x[i] + dx_0 * r**i

            return x * (length / x[-1])  # Normalize

        x_coords = generate_x_coords(n_x, Re, refinement_factor)
        y_coords = np.linspace(0, H, n_y + 1)

        # ========================================================================
        # Create structured grid of points
        # ========================================================================

        geom = gmsh.model.geo
        points = {}

        for j in range(n_y + 1):
            for i in range(n_x + 1):
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
            print(f"  Grid: {n_x} × {n_y}")
            print(f"  X spacing: {np.diff(x_coords).min():.6f} to {np.diff(x_coords).max():.6f}")
            print(f"  Y spacing: {np.diff(y_coords)[0]:.6f}")
            print(f"  Refinement factor: {refinement_factor}")

            from dolfinx.mesh import CellType
            if cell_type == CellType.quadrilateral:
                print(f"  ✓ Quadrilateral mesh")
            else:
                print(f"  ⚠ Warning: {cell_type}")

        return domain, facets
