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
        aspect_ratio = 100.0  # dX:dY ratio of the rectangles
        refinement_factor = 0.6  # >1 = finer near left boundary

        n_x = N # max(20, int(N / aspect_ratio))
        n_y = n_x

        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)

        gdim = 2
        model_rank = 0

        gmsh.model.add("OrthogonalQuadMesh")

        # ========================================================================
        # Generate node coordinates with geometric progression in X
        # ========================================================================

        def generate_x_coords(length, min_cell_length, max_cell_length, progression=1.5):
            max_cells=1000
            coords = [length]  # Start from the right
            current_pos = length
            cell_length = min_cell_length  # Start with min at the right
            cell_count = 0
            
            while current_pos > 0:
                cell_count += 1
                if cell_count > max_cells:
                    raise ValueError(f"Exceeded maximum number of cells ({max_cells}). "
                                   f"Cannot generate mesh for length={length} with current parameters.")
                
                # Use the smaller of: current cell length or remaining distance
                actual_length = min(cell_length, current_pos)
                current_pos -= actual_length
                coords.append(current_pos)
                
                # Grow cell length for next iteration, but cap at max
                cell_length = min(cell_length * progression, max_cell_length)
            
            # Reverse to get coordinates from left to right (0 to length)
            coords.reverse()
            return np.array(coords)        

        #
        def generate_y_coords(length, min_cell_length, max_cell_length, progression=1.5):
            max_cells = 1000
            coords = [0.0]
            current_pos = 0.0
            cell_length = min_cell_length
            cell_count = 0

            while current_pos < length:
                cell_count += 1
                if cell_count > max_cells:
                    raise ValueError(f"Exceeded maximum number of cells ({max_cells}). "
                                   f"Cannot generate mesh for length={length} with current parameters.")

                # Use the smaller of: current cell length or remaining distance
                actual_length = min(cell_length, length - current_pos)
                current_pos += actual_length
                coords.append(current_pos)

                # Grow cell length for next iteration, but cap at max
                cell_length = min(cell_length * progression, max_cell_length)

            return np.array(coords)

        x_coords = generate_x_coords(Re, Re/N, 100*Re/N)
        y_coords = generate_y_coords(H, H/N, 100*H/N)
#         y_coords = np.linspace(0, H, n_y + 1)

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
