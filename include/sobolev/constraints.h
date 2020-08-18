#pragma once

#include "rsurface_types.h"
#include "matrix_utils.h"

namespace rsurfaces
{
    namespace Constraints
    {
        class ConstraintBase
        {
        public:
            virtual void addTriplets(std::vector<Triplet> &triplets, MeshPtr &mesh, GeomPtr &geom, int baseRow) = 0;
            virtual void addEntries(Eigen::MatrixXd &M, MeshPtr &mesh, GeomPtr &geom, int baseRow) = 0;
            virtual size_t nRows() = 0;
        };

        inline void addEntriesToSymmetric(ConstraintBase &cs, Eigen::MatrixXd &M, MeshPtr &mesh, GeomPtr &geom, int baseRow)
        {
            Eigen::MatrixXd block;
            block.setZero(cs.nRows(), M.cols());
            cs.addEntries(block, mesh, geom, 0);

            for (int i = 0; i < block.rows(); i++)
            {
                for (int j = 0; j < block.cols(); j++)
                {
                    M(baseRow + i, j) = block(i, j);
                    M(j, baseRow + i) = block(i, j);
                }
            }
        }

        inline void addTripletsToSymmetric(ConstraintBase &cs, std::vector<Triplet> &triplets, MeshPtr &mesh, GeomPtr &geom, int baseRow)
        {
            std::vector<Triplet> block;
            cs.addTriplets(triplets, mesh, geom, 0);

            for (Triplet t : block)
            {
                triplets.push_back(Triplet(baseRow + t.row(), t.col(), t.value()));
                triplets.push_back(Triplet(t.col(), baseRow + t.row(), t.value()));
            }
        }

        class BarycenterConstraint : public ConstraintBase
        {
        public:
            virtual void addTriplets(std::vector<Triplet> &triplets, MeshPtr &mesh, GeomPtr &geom, int baseRow);
            virtual void addEntries(Eigen::MatrixXd &M, MeshPtr &mesh, GeomPtr &geom, int baseRow);
            virtual size_t nRows();
        };

        class BarycenterConstraint3X : public ConstraintBase
        {
        public:
            virtual void addTriplets(std::vector<Triplet> &triplets, MeshPtr &mesh, GeomPtr &geom, int baseRow);
            virtual void addEntries(Eigen::MatrixXd &M, MeshPtr &mesh, GeomPtr &geom, int baseRow);
            virtual size_t nRows();
        };

        class ScalingConstraint : public ConstraintBase
        {
        public:
            virtual void addTriplets(std::vector<Triplet> &triplets, MeshPtr &mesh, GeomPtr &geom, int baseRow);
            virtual void addEntries(Eigen::MatrixXd &M, MeshPtr &mesh, GeomPtr &geom, int baseRow);
            virtual size_t nRows();
        };
    } // namespace Constraints
} // namespace rsurfaces