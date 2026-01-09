"""
PVD Exporter Module

Organizes VTK output with:
- Each variable in its own subdirectory
- Single master file to load everything in ParaView
"""

import os
from dolfinx import io
from mpi4py import MPI


class PVDExporter:
    """
    Exports multiple fields to organized directory structure
    """
    
    def __init__(self, output_dir, comm=MPI.COMM_WORLD):
        self.output_dir = output_dir
        self.comm = comm
        self.writers = {}
        self.subdirs = {}
        self.functions = {}
        
        # Create main output directory
        if comm.rank == 0 and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        comm.barrier()
    
    def add_field(self, name, function):
        """Add a field to export"""
        subdir = os.path.join(self.output_dir, name)
        if self.comm.rank == 0 and not os.path.exists(subdir):
            os.makedirs(subdir, exist_ok=True)
        self.comm.barrier()
        
        pvd_file = os.path.join(subdir, "data.pvd")
        writer = io.VTKFile(self.comm, pvd_file, "w")
        
        self.writers[name] = writer
        self.functions[name] = function
        self.subdirs[name] = subdir
    
    def write(self, time):
        """Write all fields at given time"""
        for name, writer in self.writers.items():
            function = self.functions[name]
            function.x.scatter_forward()
            writer.write_function(function, time)
    
    def close(self):
        """Close all writers and create ParaView load script"""
        for writer in self.writers.values():
            writer.close()
        
        if self.comm.rank == 0:
            self._create_paraview_script()
            self._create_readme()
    
    def _create_paraview_script(self):
        """Create ParaView macro to load all fields"""
        script_file = os.path.join(self.output_dir, "load_all.py")
        abs_output_dir = os.path.abspath(self.output_dir)
        
        with open(script_file, 'w') as f:
            f.write("from paraview.simple import *\n")
            f.write("import os\n\n")
            
            f.write("# EDIT THIS if macro can't find data\n")
            f.write(f"DEFAULT_DIR = r'{abs_output_dir}'\n\n")
            
            f.write("def find_data():\n")
            f.write("    if os.path.exists(os.path.join(DEFAULT_DIR, 'displacement', 'data.pvd')):\n")
            f.write("        return DEFAULT_DIR\n")
            f.write("    cwd = os.getcwd()\n")
            f.write("    if os.path.exists(os.path.join(cwd, 'displacement', 'data.pvd')):\n")
            f.write("        return cwd\n")
            f.write("    out_dir = os.path.join(cwd, 'outputs')\n")
            f.write("    if os.path.exists(os.path.join(out_dir, 'displacement', 'data.pvd')):\n")
            f.write("        return out_dir\n")
            f.write("    print('ERROR: Data not found!'); return None\n\n")
            
            f.write("base_dir = find_data()\n")
            f.write("if not base_dir: import sys; sys.exit(1)\n\n")
            f.write("print(f'Loading from: {base_dir}\\n')\n\n")
            
            f.write("fields = {\n")
            for name in sorted(self.subdirs.keys()):
                f.write(f"    '{name}': '{self.functions[name].name}',\n")
            f.write("}\n\n")
            
            f.write("for i, (name, var) in enumerate(sorted(fields.items())):\n")
            f.write("    pvd = os.path.join(base_dir, name, 'data.pvd')\n")
            f.write("    if not os.path.exists(pvd): continue\n")
            f.write("    reader = PVDReader(FileName=pvd)\n")
            f.write("    display = Show(reader)\n")
            f.write("    ColorBy(display, ('POINTS', var))\n")
            f.write("    if i > 0: Hide(reader)\n")
            f.write("    print(f'{name} ({'visible' if i==0 else 'hidden'})')\n\n")
            
            f.write("GetActiveView().ResetCamera()\n")
            f.write("Render()\n")
        
        os.chmod(script_file, 0o755)
        print(f"✓ ParaView load script: {script_file}")
    
    def _create_readme(self):
        """Create README"""
        readme_file = os.path.join(self.output_dir, "README.txt")
        
        with open(readme_file, 'w') as f:
            f.write("VTK OUTPUT\n")
            f.write("="*60 + "\n\n")
            f.write("To visualize:\n")
            f.write("  1. Open ParaView\n")
            f.write("  2. Macros -> Add new macro\n")
            f.write("  3. Select load_all.py\n")
            f.write("  4. Macros -> load_all\n\n")
            f.write("Or open individual fields:\n")
            for name in sorted(self.subdirs.keys()):
                f.write(f"  - {name}/data.pvd\n")
    
    def summary(self):
        """Return summary string"""
        lines = [
            f"PVD Export: {self.output_dir}/",
            f"Fields: {', '.join(sorted(self.subdirs.keys()))}",
            f"Load: paraview → Macros → Add → load_all.py"
        ]
        return "\n".join(lines)
