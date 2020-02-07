# Repulsive Surfaces
Christopher Yu, Henrik Schumacher, Keenan Crane

## Quick setup instructions

First, clone the project and all its dependencies:
```
git clone --recursive https://github.com/icethrush/repulsive-surfaces.git
```

If the recursive flag was not used to clone, then one can also get the dependencies by running:
```
git submodule update --init --recursive
```

From there, the project can be built using CMake.
```
cd repulsive-surfaces
mkdir build
cd build
cmake ..
make -j4
```
We highly recommend using Clang to build the project. Building with GCC/G++ is possible, but will require a different set of warnings to be suppressed.

The code can then be run:
```
./bin/rsurfaces path/to/mesh.obj
```
