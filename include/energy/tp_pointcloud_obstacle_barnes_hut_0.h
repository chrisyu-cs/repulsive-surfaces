#pragma once

#include "rsurface_types.h"
#include "surface_energy.h"
#include "helpers.h"
#include "bct_constructors.h"
#include "optimized_bct_types.h"
#include "optimized_cluster_tree.h"
#include "derivative_assembler.h"

namespace rsurfaces
{

    // 0-th order multipole approximation of tangent-point energy
    class TPPointCloudObstacleBarnesHut0 : public SurfaceEnergy
    {
    public:
        TPPointCloudObstacleBarnesHut0(MeshPtr mesh_, GeomPtr geom_, SurfaceEnergy *bvhSharedFrom_,
                                       Eigen::MatrixXd & pt_positions, Eigen::VectorXd & pt_weights,
                                       mreal alpha_, mreal beta_, mreal theta_, mreal weight_ = 1.)
        {
            mesh = mesh_;
            geom = geom_;
            bvh = 0;
            bvhSharedFrom = bvhSharedFrom_;
            
            if( pt_positions.rows() !=  pt_weights.rows() )
            {
                eprint("TPPointCloudObstacleBarnesHut0: Number of positions vectors and number of weights does not coincide. Aborting.");
                return;
            }
            
            mint primitive_count = pt_positions.rows();
            mint dim = 3;
            mint primitive_length = 1;
            
            mreal * P_far = nullptr;
            mint far_dim = bvhSharedFrom->GetBVH()->far_dim;
            safe_alloc( P_far, primitive_count * far_dim, 0.);
            #pragma omp parallel for
            for( mint i = 0; i < pt_positions.rows(); ++i )
            {
                P_far[ far_dim * i + 0] = pt_weights(i);
                P_far[ far_dim * i + 1] = pt_positions(i,0);
                P_far[ far_dim * i + 2] = pt_positions(i,1);
                P_far[ far_dim * i + 3] = pt_positions(i,2);
            }
            
            mreal * P_near = nullptr;
            mreal * P_coords = nullptr;
            mint near_dim = bvhSharedFrom->GetBVH()->near_dim;
            safe_alloc( P_near, primitive_count * near_dim, 0.);
            safe_alloc( P_coords, primitive_count * dim, 0.);
            #pragma omp parallel for
            for( mint i = 0; i < pt_positions.rows(); ++i )
            {
                P_near[ near_dim * i + 0] = pt_weights(i);
                P_near[ near_dim * i + 1] = P_coords[ dim * i + 0] = pt_positions(i,0);
                P_near[ near_dim * i + 2] = P_coords[ dim * i + 1] = pt_positions(i,1);
                P_near[ near_dim * i + 3] = P_coords[ dim * i + 2] = pt_positions(i,2);
            }
            
            mint * idx = nullptr;
            mint * zeroes = nullptr;
            mreal * ones = nullptr;
            
            safe_iota( idx, primitive_count + 1);
            safe_alloc( zeroes, primitive_count + 1, 0);
            safe_alloc( ones, primitive_count + 1, 1.);
            
            MKLSparseMatrix AvOp = MKLSparseMatrix( primitive_count, primitive_count, idx, idx, ones ); // identity matrix
            MKLSparseMatrix DiffOp = MKLSparseMatrix( primitive_count, primitive_count, zeroes, zeroes, ones ); // zero matrix

            // create a cluster tree
            int split_threshold = 8;
            o_bvh = new OptimizedClusterTree(
                &P_coords[0],      // coordinates used for clustering
                primitive_count,            // number of primitives
                dim,                 // dimension of ambient space
                &P_coords[0], // coordinates of the convex hull of each mesh element
                primitive_length,                 // number of points in the convex hull of each mesh element (3 for triangle meshes, 2 for polylines)
                &P_near[0],        // area, barycenter, and normal of mesh element
                near_dim,                 // number of dofs of P_near per mesh element; it is 7 for polylines and triangle meshes in 3D.
                &P_far[0],        // area, barycenter, and projector of mesh element
                far_dim,                 // number of dofs of P_far per mesh element; it is 10 for polylines and triangle meshes in 3D.
                dim * dim,             // some estimate for the buffer size per vertex and cluster (usually the square of the dimension of the embedding space
                idx,      // some ordering of triangles
                split_threshold,  // create clusters only with at most this many mesh elements in it
                DiffOp,            // the first-order differential operator belonging to the hi order term of the metric
                AvOp               // the zeroth-order differential operator belonging to the lo order term of the metric
            );
            
            safe_free(idx);
            safe_free(ones);
            safe_free(zeroes);
            
            alpha = alpha_;
            beta = beta_;
            theta = theta_;
            weight = weight_;

            mreal intpart;
            use_int = (std::modf(alpha, &intpart) == 0.0) && (std::modf(beta / 2, &intpart) == 0.0);
            
            Update();
        }

        ~TPPointCloudObstacleBarnesHut0()
        {
            if (o_bvh)
            {
                delete o_bvh;
            }
        }

        // Returns the current value of the energy.
        virtual double Value();

        // Returns the current differential of the energy, stored in the given
        // V x 3 matrix, where each row holds the differential (a 3-vector) with
        // respect to the corresponding vertex.
        virtual void Differential(Eigen::MatrixXd &output);

        // Update the energy to reflect the current state of the mesh. This could
        // involve building a new BVH for Barnes-Hut energies, for instance.
        virtual void Update();

        // Get the mesh associated with this energy.
        virtual MeshPtr GetMesh();

        // Get the geometry associated with this geometry.
        virtual GeomPtr GetGeom();

        // Get the exponents of this energy; only applies to tangent-point energies.
        virtual Vector2 GetExponents();

        // Get a pointer to the current BVH for this energy.
        // Return 0 if the energy doesn't use a BVH.
        virtual OptimizedClusterTree *GetBVH();

        // Return the separation parameter for this energy.
        // Return 0 if this energy doesn't do hierarchical approximation.
        virtual double GetTheta();

        bool use_int = false;

    private:
        MeshPtr mesh = nullptr;
        GeomPtr geom = nullptr;
        mreal alpha = 6.;
        mreal beta = 12.;
        mreal weight = 1.;
        mreal theta = 0.5;

        SurfaceEnergy * bvhSharedFrom;
        OptimizedClusterTree * bvh = nullptr;
        OptimizedClusterTree * o_bvh = nullptr;

        template <typename T1, typename T2>
        mreal Energy(T1 alpha, T2 betahalf);

        template <typename T1, typename T2>
        mreal DEnergy(T1 alpha, T2 betahalf);

    }; // TPEnergyBarnesHut0

} // namespace rsurfaces
