#include "sobolev/h1.h"
#include "sobolev/h1_lbfgs.h"

namespace rsurfaces
{
    H1_LBFGS::H1_LBFGS(size_t memSize_, std::vector<Constraints::SimpleProjectorConstraint *> simpleConstraints_)
    : LBFGSOptimizer(memSize_), simpleConstraints(simpleConstraints_)
    {}

    void H1_LBFGS::SetUpInnerProduct(MeshPtr &mesh, GeomPtr &geom)
    {
        size_t nRows = mesh->nVertices() * 3;
        std::vector<Triplet> triplets, triplets3x;
        
        H1::getTriplets(triplets, mesh, geom, 1e-10);

        MatrixUtils::TripleTriplets(triplets, triplets3x);

        for (Constraints::SimpleProjectorConstraint* spc : simpleConstraints)
        {
            Constraints::addTripletsToSymmetric(*spc, triplets3x, mesh, geom, nRows);
            nRows += spc->nRows();
        }

        L.resize(nRows, nRows);
        L.setFromTriplets(triplets3x.begin(), triplets3x.end());
        
        tempVector.setZero(nRows);
    }

    void H1_LBFGS::ApplyInnerProduct(Eigen::VectorXd &input, Eigen::VectorXd &output)
    {
        tempVector.setZero();
        tempVector.head(input.rows()) = input;
        tempVector = L * tempVector;
        output = tempVector.head(output.rows());
    }

    void H1_LBFGS::ApplyInverseInnerProduct(Eigen::VectorXd &input, Eigen::VectorXd &output)
    {
        tempVector.setZero();
        tempVector.head(input.rows()) = input;
        MatrixUtils::SolveSparseSystem(L, tempVector, tempVector);
        output = tempVector.head(output.rows());
    }
}
