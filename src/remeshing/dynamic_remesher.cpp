#include "remeshing/dynamic_remesher.h"

namespace rsurfaces
{
    namespace remeshing
    {
        DynamicRemesher::DynamicRemesher(MeshPtr mesh_, GeomPtr geom_, GeomPtr geomOrig_)
        {
            mesh = mesh_;
            geom = geom_;
            geomOrig = geomOrig_;
            epsilon = 0.1;

            double sumLength = 0;
            for (Edge e : mesh->edges())
            {
                sumLength += geom->edgeLength(e);
            }
            initialAverageLength = sumLength / mesh->nEdges();

            double sumMeanCurv = 0;
            double sumArea = 0;
            for (Vertex v : mesh->vertices())
            {
                sumArea += geom->vertexDualArea(v);
                sumMeanCurv += fabs(geom->vertexMeanCurvature(v));
            }
            sumMeanCurv /= sumArea;

            initialHWeightedLength = initialAverageLength * fmax(1, sumMeanCurv);

            std::cout << "Initial average edge length = " << initialAverageLength << std::endl;
            std::cout << "Average H = " << sumMeanCurv << std::endl;
            std::cout << "Initial H-weighted length = " << initialHWeightedLength << std::endl;

            remeshingMode = RemeshingMode::SmoothFlipAndCollapse;
            smoothingMode = SmoothingMode::Circumcenter;
            flippingMode = FlippingMode::Delaunay;

            curvatureAdaptive = false;
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

        bool DynamicRemesher::Remesh(int numIters, bool changeTopology)
        {
            bool didSplitOrCollapse = false;
            switch (remeshingMode)
            {
            case RemeshingMode::FlipOnly:
            {
                for (int i = 0; i < numIters; i++)
                {
                    flipEdges();
                }
                didSplitOrCollapse = false;
                break;
            }
            case RemeshingMode::SmoothAndFlip:
            {
                for (int i = 0; i < numIters; i++)
                {
                    smoothVertices();
                    flipEdges();
                }
                didSplitOrCollapse = false;
                break;
            }
            case RemeshingMode::SmoothFlipAndCollapse:
            {
                // Only do one edge split/collapse step
                if (changeTopology)
                {
                    double l = (curvatureAdaptive) ? initialHWeightedLength : initialAverageLength;
                    double l_min = (curvatureAdaptive) ? initialAverageLength * 0.9 : initialAverageLength * 0.5;

                    didSplitOrCollapse = adjustEdgeLengths(mesh, geom, geomOrig, l, epsilon, l_min, curvatureAdaptive);
                }
                geom->refreshQuantities();

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
            return didSplitOrCollapse;
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
