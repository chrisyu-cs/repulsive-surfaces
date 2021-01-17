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
        std::vector<Triplet> triplets;
        
        H1::getTriplets(triplets, mesh, geom, 1e-10);

        for (Constraints::SimpleProjectorConstraint* spc : simpleConstraints)
        {
            Constraints::addTripletsToSymmetric(*spc, triplets, mesh, geom, nRows);
            nRows += spc->nRows();
        }

        L.resize(nRows, nRows);
        L.setFromTriplets(triplets.begin(), triplets.end());

        factorizedL.Compute(L);
        tempVector.setZero(nRows);
    }

    void H1_LBFGS::ApplyInnerProduct(Eigen::VectorXd &input, Eigen::VectorXd &output)
    {
        tempVector.block(0, 0, input.rows(), 1) = input;
        tempVector = L * tempVector;
        output = tempVector.block(0, 0, output.rows(), 1);
    }

    void H1_LBFGS::ApplyInverseInnerProduct(Eigen::VectorXd &input, Eigen::VectorXd &output)
    {
        tempVector.block(0, 0, input.rows(), 1) = input;
        tempVector = factorizedL.Solve(tempVector);
        output = tempVector.block(0, 0, output.rows(), 1);
    }    
}
