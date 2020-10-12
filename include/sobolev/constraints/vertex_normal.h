#pragma once

#include "sobolev/constraints.h"

namespace rsurfaces
{
    namespace Constraints
    {
        class VertexNormalConstraint : public SimpleProjectorConstraint
        {
        public:
            VertexNormalConstraint(MeshPtr &mesh, GeomPtr &geom);
            virtual void addTriplets(std::vector<Triplet> &triplets, MeshPtr &mesh, GeomPtr &geom, int baseRow);
            virtual void addEntries(Eigen::MatrixXd &M, MeshPtr &mesh, GeomPtr &geom, int baseRow);
            virtual size_t nRows();
            virtual void ProjectConstraint(MeshPtr &mesh, GeomPtr &geom);
        
            void pinVertices(MeshPtr &mesh, GeomPtr &geom, std::vector<size_t> &indices_);

        private:
            std::vector<size_t> indices;
            std::vector<Vector3> initNormals;
        };
    } // namespace Constraints
} // namespace rsurfaces