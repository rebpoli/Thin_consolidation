FROM docker.io/dolfinx/dolfinx:stable

ENV DEBIAN_FRONTEND=noninteractive \
    SINGTAG=FENICS

# Install necessary build tools and dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    gfortran \
    git \
    wget \
    python3 \
    python3-dev \
    python3-pip \
    python3-venv \
    gfortran \
    libomp-dev \    
    liblapack-dev \
    zlib1g-dev \
    ffmpeg \
    m4 \
    automake \
    libtool \
    vim \
    rsync \
    eog \
    exuberant-ctags \
    binutils \
    nsight-systems \
    pkg-config \
    bash-completion \
    && rm -rf /var/lib/apt/lists/*

# Create directory for installations

RUN pip install --no-cache-dir matplotlib scipy pandas pyvista h5py xarray meshio vtk pydantic PyYAML
RUN pip install --no-cache-dir netCDF4


WORKDIR /opt/src

CMD ["/bin/bash"]

