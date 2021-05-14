#pragma once

#include "geometrycentral/surface/halfedge_mesh.h"
#include "geometrycentral/surface/meshio.h"
#include "geometrycentral/surface/vertex_position_geometry.h"

#include "../deps/polyscope/deps/args/args/args.hxx"

#include <omp.h>
#include <mkl.h>
#include <mkl_spblas.h>

//#include <tbb/task_scheduler_init.h>
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

#include "consistency_helpers.h"


namespace rsurfaces
{
    
} // namespace rsurfaces
