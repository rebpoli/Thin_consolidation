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
    
    def generate(self):
        self.domain, self.facets = self._create_gmsh_mesh()
        return self.domain, self.facets
    
    def _create_gmsh_mesh(self):
        Re    = self.mesh_cfg.Re
        H     = self.mesh_cfg.H
        N     = self.mesh_cfg.N
        hsize = H / N
        
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        
        gdim       = 2
        model_rank = 0
        
        gmsh.model.add("Cylinder")
        geom = gmsh.model.geo
        
        p1 = geom.add_point(0,  0, 0)
        p2 = geom.add_point(Re, 0, 0)
        p3 = geom.add_point(Re, H, 0)
        p4 = geom.add_point(0,  H, 0)
        
        bottom = geom.add_line(p1, p2)
        right  = geom.add_line(p2, p3)
        top    = geom.add_line(p3, p4)
        left   = geom.add_line(p4, p1)
        
        boundary = geom.add_curve_loop([bottom, right, top, left])
        surf     = geom.add_plane_surface([boundary])
        
        geom.synchronize()
        
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", hsize)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", hsize)
        
        gmsh.model.addPhysicalGroup(gdim, [surf], 1)
        gmsh.model.addPhysicalGroup(gdim - 1, [bottom], 1, name="bottom")
        gmsh.model.addPhysicalGroup(gdim - 1, [right],  2, name="right")
        gmsh.model.addPhysicalGroup(gdim - 1, [top],    3, name="top")
        gmsh.model.addPhysicalGroup(gdim - 1, [left],   4, name="left")
        
        gmsh.model.mesh.generate(gdim)
        
        mesh_data = gmshio.model_to_mesh(gmsh.model, self.comm, model_rank, gdim=gdim)
        domain    = mesh_data.mesh
        facets    = mesh_data.facet_tags
        
        gmsh.finalize()
        
        if self.comm.rank == 0:
            num_cells = domain.topology.index_map(gdim).size_global
            print(f"✓ Mesh created: {num_cells} cells")
        
        return domain, facets
