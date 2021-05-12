#include "energy/boundary_curvature.h"
#include "matrix_utils.h"

const bool useLength = false;

namespace rsurfaces
{
    BoundaryCurvaturePenalty::BoundaryCurvaturePenalty(MeshPtr mesh_, GeomPtr geom_, double weight_)
    {
        weight = weight_;
        mesh = mesh_;
        geom = geom_;

        if (mesh->nBoundaryLoops() == 0)
        {
            throw std::runtime_error("Boundary curvature penalty was specified, but mesh has no boundary loops.");
        }
    }

    double BoundaryCurvaturePenalty::Value()
    {
       // for convenience, get a reference to the vertex positions
       surface::VertexData<Vector3>& p = geom->inputVertexPositions;

       // The energy is
       // \[
       //    \sum_i L_i (\kappa_i/L_i)^2 = \sum_i \kappa_i^2/L_i
       // \]
       // where $\kappa_i$ is the exterior angle at vertex $i$, and
       // $L_i$ is half the length of the two edges incident on $i$.
       double sum = 0;
       for (surface::BoundaryLoop loop : mesh->boundaryLoops())
       {
          for( GCHalfedge h : loop.adjacentHalfedges() )
          {
             // get three consecutive vertices i, j, k along boundary loop
             GCVertex i = h.vertex();
             GCVertex j = h.next().vertex();
             GCVertex k = h.next().next().vertex();

             // get edge vectors and their lengths
             Vector3 u = p[i] - p[j];
             Vector3 v = p[k] - p[j];
             double a = norm(u);
             double b = norm(v);

             // compute integral of squared curvature over this vertex neighborhood
             double L = useLength ? (a+b)/2. : 1.;
             double theta = acos( std::max( -1., std::min( 1., dot( u, v )/(a*b) )));
             double kappa = M_PI - theta;

             // add to total
             sum += kappa*kappa/L;
          }
       }

       return weight * sum;
    }

    void BoundaryCurvaturePenalty::Differential(Eigen::MatrixXd &output)
    {
       // for convenience, get a reference to the vertex positions
       surface::VertexData<Vector3>& p = geom->inputVertexPositions;

       VertexIndices inds = mesh->getVertexIndices();

       for (surface::BoundaryLoop loop : mesh->boundaryLoops())
       {
          for( GCHalfedge h : loop.adjacentHalfedges() )
          {
             // get three consecutive vertices i, j, k along boundary loop
             GCVertex i = h.vertex();
             GCVertex j = h.next().vertex();
             GCVertex k = h.next().next().vertex();

             // get edge vectors and their lengths
             Vector3 u = p[i] - p[j];
             Vector3 v = p[k] - p[j];
             double a = norm(u);
             double b = norm(v);

             // compute dual length L and exterior angle kappa
             double L = useLength ? (a+b)/2. : 1.;
             double theta = acos( std::max( -1., std::min( 1., dot( u, v )/(a*b) )));
             double kappa = M_PI - theta;
             
             // compute unit normal, being careful about near-zero vectors
             // (note that if the edges are close to parallel, the derivatives
             // will be zero anyway)
             const double eps = 1e-7;
             Vector3 n = cross( u, v );
             double m = norm(n);
             if( m > eps )
             {
                n /= m;
             }

             // compute the derivatives for this term with
             // respect to all three vertices
             Vector3 dThetaI = -cross( n, u )/(a*a);
             Vector3 dThetaK =  cross( n, v )/(b*b);
             Vector3 dThetaJ =  -( dThetaI + dThetaK );
             Vector3 dLI = -u/(2.*a);
             Vector3 dLK =  v/(2.*b);
             Vector3 dLJ = -( dLI + dLK );

             // accumulate these contributions
             if( useLength )
             {
                MatrixUtils::addToRow(output, inds[i], -weight * kappa * (kappa*dLI + 2.*L*dThetaI) / (L*L) );
                MatrixUtils::addToRow(output, inds[j], -weight * kappa * (kappa*dLJ + 2.*L*dThetaJ) / (L*L) );
                MatrixUtils::addToRow(output, inds[k], -weight * kappa * (kappa*dLK + 2.*L*dThetaK) / (L*L) );
             }
             else
             {
                MatrixUtils::addToRow(output, inds[i], -weight * 2.*kappa * dThetaI );
                MatrixUtils::addToRow(output, inds[j], -weight * 2.*kappa * dThetaJ );
                MatrixUtils::addToRow(output, inds[k], -weight * 2.*kappa * dThetaK );
             }
          }
       }
    }

    // Get the exponents of this energy; only applies to tangent-point energies.
    Vector2 BoundaryCurvaturePenalty::GetExponents()
    {
        return Vector2{1, 0};
    }

    // Get a pointer to the current BVH for this energy.
    // Return 0 if the energy doesn't use a BVH.
    OptimizedClusterTree* BoundaryCurvaturePenalty::GetBVH()
    {
        return 0;
    }

    // Return the separation parameter for this energy.
    // Return 0 if this energy doesn't do hierarchical approximation.
    double BoundaryCurvaturePenalty::GetTheta()
    {
        return 0;
    }
}
