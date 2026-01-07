
BUILD_DIR = build
INSTALL_DIR = images
TAG = fenicsx
DOCKERFILE = Dockerfile
TARFILE = $(BUILD_DIR)/$(TAG).tar
SIFFILE = $(INSTALL_DIR)/$(TAG).sif

all: .force build install

build: .force
	rm -rf $(BUILD_DIR)
	mkdir -p $(BUILD_DIR) $(INSTALL_DIR)
	podman build -f $(DOCKERFILE) -t $(TAG) .
	rm -f $(TARFILE)
	podman save $(TAG) -o $(TARFILE)
	singularity build --force $(SIFFILE) docker-archive:$(TARFILE)

install: .force
	cp images/$(TAG).sif /opt/container/
	sudo chmod a+rx /opt/container/bin/*

# download:
#     mkdir -p $(INSTALL_DIR)
#     cd $(INSTALL_DIR) && wget -L --content-disposition "https://utexas.app.box.com/shared/static/o21pihvh0ud6is0h195i7fm596mcgbo3"

.force:
