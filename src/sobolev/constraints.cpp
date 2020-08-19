#include "sobolev/constraints.h"

namespace rsurfaces
{
    namespace Constraints
    {

        void addEntriesToSymmetric(ConstraintBase &cs, Eigen::MatrixXd &M, MeshPtr &mesh, GeomPtr &geom, int baseRow)
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

        void addTripletsToSymmetric(ConstraintBase &cs, std::vector<Triplet> &triplets, MeshPtr &mesh, GeomPtr &geom, int baseRow)
        {
            std::vector<Triplet> block;
            cs.addTriplets(block, mesh, geom, 0);

            for (Triplet t : block)
            {
                triplets.push_back(Triplet(baseRow + t.row(), t.col(), t.value()));
                triplets.push_back(Triplet(t.col(), baseRow + t.row(), t.value()));
            }
        }

    } // namespace Constraints
} // namespace rsurfaces