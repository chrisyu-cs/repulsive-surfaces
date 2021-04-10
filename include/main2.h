#pragma once

#include "geometrycentral/surface/halfedge_mesh.h"
#include "geometrycentral/surface/meshio.h"
#include "geometrycentral/surface/vertex_position_geometry.h"

#include <boost/program_options.hpp>
namespace po = boost::program_options;

#include <omp.h>
#include <mkl.h>
#include <memory>
#include <Eigen/Core>

#include "rsurface_types.h"
#include "surface_flow.h"


#include "remeshing/dynamic_remesher.h"
#include "remeshing/remeshing.h"

#include "scene_file.h"

#include "bct_kernel_type.h"
#include "optimized_bct.h"
#include "bct_constructors.h"

#include "helpers.h"


#include "energy/all_energies.h"



#define EIGEN_NO_DEBUG
