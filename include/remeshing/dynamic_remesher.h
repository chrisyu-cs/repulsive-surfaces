#pragma once

#include "rsurface_types.h"
#include "remeshing/remeshing.h"
#include "vertex_data_wrapper.h"

#include "sobolev/all_constraints.h"
#include "energy/all_energies.h"
#include "profiler.h"

namespace rsurfaces
{
    namespace remeshing
    {
        enum class RemeshingMode
        {
            FlipOnly,
            SmoothOnly,
            SmoothAndFlip,
            SmoothFlipAndCollapse
        };

        enum class SmoothingMode
        {
            Laplacian,
            Circumcenter
        };

        enum class FlippingMode
        {
            Delaunay,
            Degree
        };

        class DynamicRemesher
        {
        public:
            DynamicRemesher(MeshPtr mesh_, GeomPtr geom_, GeomPtr geomOrig_);
            void SetModes(RemeshingMode rMode, SmoothingMode sMode, FlippingMode fMode);
            bool Remesh(int numIters, bool changeTopology);
            void KeepVertexDataUpdated(VertexDataWrapper *data);
            bool curvatureAdaptive;

            SmoothingMode smoothingMode;
            RemeshingMode remeshingMode;
            FlippingMode flippingMode;

        private:
            void flipEdges();
            void smoothVertices();

            MeshPtr mesh;
            GeomPtr geom;
            GeomPtr geomOrig;
            double initialAverageLength;
            double initialHWeightedLength;
            double epsilon;
            std::vector<VertexDataWrapper *> vectorData;
        };
    } // namespace remeshing
} // namespace rsurfaces
