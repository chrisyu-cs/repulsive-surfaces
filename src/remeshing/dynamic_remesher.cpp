#include "remeshing/dynamic_remesher.h"

namespace rsurfaces
{
    namespace remeshing
    {
        DynamicRemesher::DynamicRemesher(MeshPtr mesh_, GeomPtr geom_)
        {
            mesh = mesh_;
            geom = geom_;

            double sumLength = 0;
            for (Edge e : mesh->edges()) {
                sumLength += geom->edgeLength(e);
            }
            initialAverageLength = sumLength / mesh->nEdges();

            std::cout << "Initial average edge length = " << initialAverageLength << std::endl;

            remeshingMode = RemeshingMode::SmoothAndFlip;
            smoothingMode = SmoothingMode::Laplacian;
            flippingMode = FlippingMode::Delaunay;
        }

        void DynamicRemesher::SetModes(RemeshingMode rMode, SmoothingMode sMode, FlippingMode fMode)
        {
            remeshingMode = rMode;
            smoothingMode = sMode;
            flippingMode = fMode;
        }

        void DynamicRemesher::KeepVertexDataUpdated(VertexDataWrapper *data)
        {
            std::cout << "Will attempt to keep vertex data updated through edge splits and collapses." << std::endl;
            vectorData.push_back(data);
        }

        void DynamicRemesher::Remesh(int numIters, bool changeTopology)
        {
            switch (remeshingMode)
            {
            case RemeshingMode::FlipOnly:
            {
                for (int i = 0; i < numIters; i++)
                {
                    flipEdges();
                }
                break;
            }
            case RemeshingMode::SmoothAndFlip:
            {
                for (int i = 0; i < numIters; i++)
                {
                    smoothVertices();
                    flipEdges();
                }
                break;
            }
            case RemeshingMode::SmoothFlipAndCollapse:
            {
                // Only do one edge split/collapse step
                if (changeTopology) adjustEdgeLengths(mesh, geom, initialAverageLength, 1, initialAverageLength * 0.6);

                for (int i = 0; i < numIters; i++)
                {
                    smoothVertices();
                    flipEdges();
                }
                break;
            }
            default:
                throw std::runtime_error("Unknown remeshing mode.");
                break;
            }
            
            geom->refreshQuantities();
        }

        void DynamicRemesher::flipEdges()
        {
            switch (flippingMode)
            {
            case FlippingMode::Delaunay:
                fixDelaunay(mesh, geom);
                break;
            case FlippingMode::Degree:
                adjustVertexDegrees(mesh, geom);
                break;
            default:
                throw std::runtime_error("Unknown flipping mode.");
                break;
            }
        }

        void DynamicRemesher::smoothVertices()
        {
            switch (smoothingMode)
            {
            case SmoothingMode::Laplacian:
                smoothByLaplacian(mesh, geom);
                break;
            case SmoothingMode::Circumcenter:
                smoothByCircumcenter(mesh, geom);
                break;
            default:
                throw std::runtime_error("Unknown smoothing mode.");
                break;
            }
        }

    } // namespace remeshing
} // namespace rsurfaces