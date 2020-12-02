#pragma once

#include "rsurface_types.h"
#include "remeshing/remeshing.h"
#include "vertex_data_wrapper.h"

#include "sobolev/all_constraints.h"
#include "energy/all_energies.h"

namespace rsurfaces
{
    namespace remeshing
    {
        enum class RemeshingMode
        {
            FlipOnly,
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
            DynamicRemesher(MeshPtr mesh_, GeomPtr geom_);
            void SetModes(RemeshingMode rMode, SmoothingMode sMode, FlippingMode fMode);
            void Remesh(int numIters, bool changeTopology);
            void SmoothAndFlip(int numIters);
            void KeepVertexDataUpdated(VertexDataWrapper *data);

            SmoothingMode smoothingMode;
            RemeshingMode remeshingMode;
            FlippingMode flippingMode;

        private:
            void flipEdges();
            void smoothVertices();

            MeshPtr mesh;
            GeomPtr geom;
            std::vector<VertexDataWrapper *> vectorData;
        };

        template <typename Mode>
        inline std::string StringOfMode(Mode mode)
        {
            throw std::runtime_error("Can't use unspecialized template here.");
        }

        template <>
        inline std::string StringOfMode(RemeshingMode mode)
        {
            switch (mode)
            {
            case RemeshingMode::FlipOnly:
                return "Flip only";
            case RemeshingMode::SmoothAndFlip:
                return "Smooth + flip";
            case RemeshingMode::SmoothFlipAndCollapse:
                return "Smooth + flip + collapse";
            default:
                throw std::runtime_error("Unknown remeshing mode.");
            }
        }

        template <>
        inline std::string StringOfMode(SmoothingMode mode)
        {
            switch (mode)
            {
            case SmoothingMode::Circumcenter:
                return "Circumcenter";
            case SmoothingMode::Laplacian:
                return "Laplacian";
            default:
                throw std::runtime_error("Unknown smoothing mode.");
            }
        }

        template <>
        inline std::string StringOfMode(FlippingMode mode)
        {
            switch (mode)
            {
            case FlippingMode::Delaunay:
                return "Delaunay";
            case FlippingMode::Degree:
                return "Degree";
            default:
                throw std::runtime_error("Unknown flipping mode.");
            }
        }
    } // namespace remeshing
} // namespace rsurfaces