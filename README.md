# Repulsive Surfaces
Christopher Yu, Henrik Schumacher, Keenan Crane

## Quick setup instructions

First, clone the project and all its dependencies:
either via https (password login)
```
git clone --recursive https://github.com/icethrush/repulsive-surfaces.git
```
or via ssh (passwordless login -- needs ssh keys to be set up with github)
```
git clone --recursive git@github.com:icethrush/repulsive-surfaces.git
```

If the recursive flag was not used to clone, then one can also get the dependencies by running:
```
git submodule update --init --recursive
```

To set environment variables for the MKL, please run the setvars.[bat|sh] script. For example, with oneMKL and on a unixoid operating system, one can do this with
```
source /opt/intel/oneapi/setvars.sh
```
(This sets the environment variables one for one terminal session. You may want to add the above line to your ~/.profile, ~/.bashrc, or ~/.zshrc.)

From there, the project can be built using CMake.
```
cd repulsive-surfaces
mkdir build
cd build
cmake ..
make -j4
```
We used Clang to compile the codebase during development, but GCC/G++ should also work, though depending on the version it may emit some different warnings.

The code can then be run:
```
./bin/rsurfaces path/to/scene.txt
```
The executable can also be invoked directly on a mesh OBJ, which will initialize an energy with some default settings. But complex scenes such as those shown in the paper should be defined using a scene file. For instructions on how to set up a scene file, see `scenes/FORMAT.txt`. Example scenes can also be found in the subdirectories in `scenes/`.
