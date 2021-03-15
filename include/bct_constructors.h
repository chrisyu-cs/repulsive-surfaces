#pragma once

#include "optimized_bct.h"
#include "optimized_cluster_tree.h"
#include "sobolev/hs_operators.h"

namespace rsurfaces
{

    inline OptimizedClusterTree * CreateOptimizedBVH(MeshPtr &mesh, GeomPtr &geom)
    {
        int vertex_count = mesh->nVertices();
        int primitive_count = mesh->nFaces();
        FaceIndices fInds = mesh->getFaceIndices();
        VertexIndices vInds = mesh->getVertexIndices();

        int dim = 3;
        int near_dim = 1 + dim + dim;
        int far_dim  = 1 + dim + dim;
        
        double athird = 1. / 3.;

        std::vector<int> ordering(primitive_count);
        std::vector<double> P_coords(3 * primitive_count);
        std::vector<double> P_hull_coords(9 * primitive_count);

        MKLSparseMatrix AvOp = MKLSparseMatrix( primitive_count, vertex_count, 3 * primitive_count ); // This is a sparse matrix in CSR format.
        AvOp.outer[primitive_count] = 3 * primitive_count;
        
        std::vector<double> P_near(near_dim * primitive_count);
        std::vector<double> P_far (far_dim * primitive_count);
        
        for (auto face : mesh->faces())
        {
            int i = fInds[face];

            ordering[i] = i; // unless we know anything better, let's use the identity permutation.
            if( static_cast<long long>(i) >= static_cast<long long>(primitive_count) )
            {
                eprint("mesh->getFaceIndices() must be corrupted.");
            }
            
            GCHalfedge he = face.halfedge();

            int i0 = vInds[he.vertex()];
            int i1 = vInds[he.next().vertex()];
            int i2 = vInds[he.next().next().vertex()];
            Vector3 p1 = geom->inputVertexPositions[i0];
            Vector3 p2 = geom->inputVertexPositions[i1];
            Vector3 p3 = geom->inputVertexPositions[i2];

            P_coords[3 * i + 0] = athird * (p1.x + p2.x + p3.x);
            P_coords[3 * i + 1] = athird * (p1.y + p2.y + p3.y);
            P_coords[3 * i + 2] = athird * (p1.z + p2.z + p3.z);

            P_far[far_dim * i + 0] = P_near[near_dim * i + 0] = geom->faceAreas[face];
            P_far[far_dim * i + 1] = P_near[near_dim * i + 1] = P_coords[3 * i + 0];
            P_far[far_dim * i + 2] = P_near[near_dim * i + 2] = P_coords[3 * i + 1];
            P_far[far_dim * i + 3] = P_near[near_dim * i + 3] = P_coords[3 * i + 2];
            
            mreal n1 = geom->faceNormals[face].x;
            mreal n2 = geom->faceNormals[face].y;
            mreal n3 = geom->faceNormals[face].z;
            
            P_far[far_dim * i + 4] = P_near[near_dim * i + 4] = n1;
            P_far[far_dim * i + 5] = P_near[near_dim * i + 5] = n2;
            P_far[far_dim * i + 6] = P_near[near_dim * i + 6] = n3;

            P_hull_coords[9 * i + 0] = p1.x;
            P_hull_coords[9 * i + 1] = p1.y;
            P_hull_coords[9 * i + 2] = p1.z;
            P_hull_coords[9 * i + 3] = p2.x;
            P_hull_coords[9 * i + 4] = p2.y;
            P_hull_coords[9 * i + 5] = p2.z;
            P_hull_coords[9 * i + 6] = p3.x;
            P_hull_coords[9 * i + 7] = p3.y;
            P_hull_coords[9 * i + 8] = p3.z;

            AvOp.outer[i] = 3 * i;

            AvOp.inner[3 * i + 0] = i0;
            AvOp.inner[3 * i + 1] = i1;
            AvOp.inner[3 * i + 2] = i2;
        
            AvOp.values[3 * i + 0] = athird;
            AvOp.values[3 * i + 1] = athird;
            AvOp.values[3 * i + 2] = athird;

            std::sort( AvOp.inner + 3 * i, AvOp.inner + 3 * (i + 1) );
        }
        
        EigenMatrixCSR DiffOp0 = Hs::BuildDfOperator(mesh, geom); // This is a sparse matrix in CSC!!! format.
        
        DiffOp0.makeCompressed();
        
        MKLSparseMatrix DiffOp = MKLSparseMatrix( DiffOp0.rows(), DiffOp0.cols(), DiffOp0.outerIndexPtr(), DiffOp0.innerIndexPtr(), DiffOp0.valuePtr() ); // This is a sparse matrix in CSR format.

        // create a cluster tree
        int split_threashold = 8;
        return new OptimizedClusterTree(
            &P_coords[0],      // coordinates used for clustering
            primitive_count,            // number of primitives
            dim,                 // dimension of ambient space
            &P_hull_coords[0], // coordinates of the convex hull of each mesh element
            3,                 // number of points in the convex hull of each mesh element (3 for triangle meshes, 2 for polylines)
            &P_near[0],        // area, barycenter, and normal of mesh element
            near_dim,                 // number of dofs of P_near per mesh element; it is 7 for polylines and triangle meshes in 3D.
            &P_far[0],        // area, barycenter, and projector of mesh element
            far_dim,                 // number of dofs of P_far per mesh element; it is 10 for polylines and triangle meshes in 3D.
            dim * dim,             // some estimate for the buffer size per vertex and cluster (usually the square of the dimension of the embedding space
            &ordering[0],      // some ordering of triangles
            split_threashold,  // create clusters only with at most this many mesh elements in it
            DiffOp,            // the first-order differential operator belonging to the hi order term of the metric
            AvOp               // the zeroth-order differential operator belonging to the lo order term of the metric
        );
    } // CreateOptimizedBVH

    inline void UpdateOptimizedBVH(MeshPtr &mesh, GeomPtr &geom, OptimizedClusterTree * bvh)
    {
        bvh->UpdateWithNewPositions(mesh, geom);
    } // UpdateOptimizedBVH
    
    inline OptimizedClusterTree * CreateOptimizedBVH_Hybrid(MeshPtr &mesh, GeomPtr &geom)
    {
        int vertex_count = mesh->nVertices();
        int primitive_count = mesh->nFaces();
        FaceIndices fInds = mesh->getFaceIndices();
        VertexIndices vInds = mesh->getVertexIndices();

        int dim = 3;
        int near_dim = 1 + dim + dim;
        int far_dim = 1 + dim + dim * (dim + 1)/2;
        
        double athird = 1. / 3.;

        std::vector<int> ordering(primitive_count);
        std::vector<double> P_coords(3 * primitive_count);
        std::vector<double> P_hull_coords(9 * primitive_count);

        MKLSparseMatrix AvOp = MKLSparseMatrix( primitive_count, vertex_count, 3 * primitive_count ); // This is a sparse matrix in CSR format.
        AvOp.outer[primitive_count] = 3 * primitive_count;
        
        std::vector<double> P_near(near_dim * primitive_count);
        std::vector<double> P_far (far_dim * primitive_count);
        
        for (auto face : mesh->faces())
        {
            int i = fInds[face];

            ordering[i] = i; // unless we know anything better, let's use the identity permutation.
            if( static_cast<long long>(i) >= static_cast<long long>(primitive_count) )
            {
                eprint("mesh->getFaceIndices() must be corrupted.");
            }
            
            GCHalfedge he = face.halfedge();

            int i0 = vInds[he.vertex()];
            int i1 = vInds[he.next().vertex()];
            int i2 = vInds[he.next().next().vertex()];
            Vector3 p1 = geom->inputVertexPositions[i0];
            Vector3 p2 = geom->inputVertexPositions[i1];
            Vector3 p3 = geom->inputVertexPositions[i2];

            P_coords[3 * i + 0] = athird * (p1.x + p2.x + p3.x);
            P_coords[3 * i + 1] = athird * (p1.y + p2.y + p3.y);
            P_coords[3 * i + 2] = athird * (p1.z + p2.z + p3.z);

            P_far[far_dim * i + 0] = P_near[near_dim * i + 0] = geom->faceAreas[face];
            P_far[far_dim * i + 1] = P_near[near_dim * i + 1] = P_coords[3 * i + 0];
            P_far[far_dim * i + 2] = P_near[near_dim * i + 2] = P_coords[3 * i + 1];
            P_far[far_dim * i + 3] = P_near[near_dim * i + 3] = P_coords[3 * i + 2];
            
            mreal n1 = geom->faceNormals[face].x;
            mreal n2 = geom->faceNormals[face].y;
            mreal n3 = geom->faceNormals[face].z;
            
            P_near[near_dim * i + 4] = n1;
            P_near[near_dim * i + 5] = n2;
            P_near[near_dim * i + 6] = n3;
            
            P_far[far_dim * i + 4] = n1 * n1;
            P_far[far_dim * i + 5] = n1 * n2;
            P_far[far_dim * i + 6] = n1 * n3;
            P_far[far_dim * i + 7] = n2 * n2;
            P_far[far_dim * i + 8] = n2 * n3;
            P_far[far_dim * i + 9] = n3 * n3;

            P_hull_coords[9 * i + 0] = p1.x;
            P_hull_coords[9 * i + 1] = p1.y;
            P_hull_coords[9 * i + 2] = p1.z;
            P_hull_coords[9 * i + 3] = p2.x;
            P_hull_coords[9 * i + 4] = p2.y;
            P_hull_coords[9 * i + 5] = p2.z;
            P_hull_coords[9 * i + 6] = p3.x;
            P_hull_coords[9 * i + 7] = p3.y;
            P_hull_coords[9 * i + 8] = p3.z;

            AvOp.outer[i] = 3 * i;

            AvOp.inner[3 * i + 0] = i0;
            AvOp.inner[3 * i + 1] = i1;
            AvOp.inner[3 * i + 2] = i2;
        
            AvOp.values[3 * i + 0] = athird;
            AvOp.values[3 * i + 1] = athird;
            AvOp.values[3 * i + 2] = athird;

            std::sort( AvOp.inner + 3 * i, AvOp.inner + 3 * (i + 1) );
        }
        
        EigenMatrixCSR DiffOp0 = Hs::BuildDfOperator(mesh, geom); // This is a sparse matrix in CSC!!! format.
        
        DiffOp0.makeCompressed();
        
        MKLSparseMatrix DiffOp = MKLSparseMatrix( DiffOp0.rows(), DiffOp0.cols(), DiffOp0.outerIndexPtr(), DiffOp0.innerIndexPtr(), DiffOp0.valuePtr() ); // This is a sparse matrix in CSR format.

        // create a cluster tree
        int split_threashold = 8;
        return new OptimizedClusterTree(
            &P_coords[0],      // coordinates used for clustering
            primitive_count,            // number of primitives
            dim,                 // dimension of ambient space
            &P_hull_coords[0], // coordinates of the convex hull of each mesh element
            3,                 // number of points in the convex hull of each mesh element (3 for triangle meshes, 2 for polylines)
            &P_near[0],        // area, barycenter, and normal of mesh element
            near_dim,                 // number of dofs of P_near per mesh element; it is 7 for polylines and triangle meshes in 3D.
            &P_far[0],        // area, barycenter, and projector of mesh element
            far_dim,                 // number of dofs of P_far per mesh element; it is 10 for polylines and triangle meshes in 3D.
                                        dim * dim,             // some estimate for the buffer size per vertex and cluster (usually the square of the dimension of the embedding space
            &ordering[0],      // some ordering of triangles
            split_threashold,  // create clusters only with at most this many mesh elements in it
            DiffOp,            // the first-order differential operator belonging to the hi order term of the metric
            AvOp               // the zeroth-order differential operator belonging to the lo order term of the metric
        );
    } // CreateOptimizedBVH

    inline OptimizedClusterTree * CreateOptimizedBVH_Projectors(MeshPtr &mesh, GeomPtr &geom)
    {
        int vertex_count = mesh->nVertices();
        int primitive_count = mesh->nFaces();
        int dim = 3;
        int near_dim = 1 + dim + dim * (dim + 1)/2;
        int far_dim = 1 + dim + dim * (dim + 1)/2;
        int primitive_length = 3;
        
        FaceIndices fInds = mesh->getFaceIndices();
        VertexIndices vInds = mesh->getVertexIndices();

        double athird = 1. / 3.;

        std::vector<int> ordering(primitive_count);
        std::vector<double> P_coords(3 * primitive_count);
        std::vector<double> P_hull_coords(9 * primitive_count);

        MKLSparseMatrix AvOp = MKLSparseMatrix( primitive_count, vertex_count, 3 * primitive_count ); // This is a sparse matrix in CSR format.
        AvOp.outer[primitive_count] = 3 * primitive_count;
        
        std::vector<double> P_near(near_dim * primitive_count);
        std::vector<double> P_far (far_dim * primitive_count);
        
        for (auto face : mesh->faces())
        {
            int i = fInds[face];

            ordering[i] = i; // unless we know anything better, let's use the identity permutation.
            if( static_cast<long long>(i) >= static_cast<long long>(primitive_count) )
            {
                eprint("mesh->getFaceIndices() must be corrupted.");
            }

            GCHalfedge he = face.halfedge();

            int i0 = vInds[he.vertex()];
            int i1 = vInds[he.next().vertex()];
            int i2 = vInds[he.next().next().vertex()];
            Vector3 p1 = geom->inputVertexPositions[i0];
            Vector3 p2 = geom->inputVertexPositions[i1];
            Vector3 p3 = geom->inputVertexPositions[i2];

            P_coords[3 * i + 0] = athird * (p1.x + p2.x + p3.x);
            P_coords[3 * i + 1] = athird * (p1.y + p2.y + p3.y);
            P_coords[3 * i + 2] = athird * (p1.z + p2.z + p3.z);

            P_far[far_dim * i + 0] = P_near[near_dim * i + 0] = geom->faceAreas[face];
            P_far[far_dim * i + 1] = P_near[near_dim * i + 1] = P_coords[3 * i + 0];
            P_far[far_dim * i + 2] = P_near[near_dim * i + 2] = P_coords[3 * i + 1];
            P_far[far_dim * i + 3] = P_near[near_dim * i + 3] = P_coords[3 * i + 2];
            
            mreal n1 = geom->faceNormals[face].x;
            mreal n2 = geom->faceNormals[face].y;
            mreal n3 = geom->faceNormals[face].z;
            P_far[far_dim * i + 4] = P_near[near_dim * i + 4] = n1 * n1;
            P_far[far_dim * i + 5] = P_near[near_dim * i + 5] = n1 * n2;
            P_far[far_dim * i + 6] = P_near[near_dim * i + 6] = n1 * n3;
            P_far[far_dim * i + 7] = P_near[near_dim * i + 7] = n2 * n2;
            P_far[far_dim * i + 8] = P_near[near_dim * i + 8] = n2 * n3;
            P_far[far_dim * i + 9] = P_near[near_dim * i + 9] = n3 * n3;

            P_hull_coords[9 * i + 0] = p1.x;
            P_hull_coords[9 * i + 1] = p1.y;
            P_hull_coords[9 * i + 2] = p1.z;
            P_hull_coords[9 * i + 3] = p2.x;
            P_hull_coords[9 * i + 4] = p2.y;
            P_hull_coords[9 * i + 5] = p2.z;
            P_hull_coords[9 * i + 6] = p3.x;
            P_hull_coords[9 * i + 7] = p3.y;
            P_hull_coords[9 * i + 8] = p3.z;

            AvOp.outer[i] = 3 * i;
            
            AvOp.inner[3 * i + 0] = i0;
            AvOp.inner[3 * i + 1] = i1;
            AvOp.inner[3 * i + 2] = i2;
        
            AvOp.values[3 * i + 0] = athird;
            AvOp.values[3 * i + 1] = athird;
            AvOp.values[3 * i + 2] = athird;

            std::sort( AvOp.inner + 3 * i, AvOp.inner + 3 * (i + 1) );
        }
        
        EigenMatrixCSR DiffOp0 = Hs::BuildDfOperator(mesh, geom); // This is a sparse matrix in CSC!!! format.
        
        DiffOp0.makeCompressed();
        
        MKLSparseMatrix DiffOp = MKLSparseMatrix( DiffOp0.rows(), DiffOp0.cols(), DiffOp0.outerIndexPtr(), DiffOp0.innerIndexPtr(), DiffOp0.valuePtr() ); // This is a sparse matrix in CSR format.

        // create a cluster tree
        int split_threashold = 8;
        return new OptimizedClusterTree(
            &P_coords[0],      // coordinates used for clustering
            primitive_count,            // number of primitives
            dim,                 // dimension of ambient space
            &P_hull_coords[0], // coordinates of the convex hull of each mesh element
            primitive_length,                 // number of points in the convex hull of each mesh element (3 for triangle meshes, 2 for polylines)
            &P_near[0],        // area, barycenter, and normal of mesh element
            near_dim,                 // number of dofs of P_data per mesh element; it is 7 for polylines and triangle meshes in 3D.
            &P_far[0],        // area, barycenter, and projector of mesh element
            far_dim,                 // number of dofs of P_data per mesh element; it is 10 for polylines and triangle meshes in 3D.
            std::max( dim * dim, far_dim ),             // some estimate for the buffer size per vertex and cluster (usually the square of the dimension of the embedding space
            &ordering[0],      // some ordering of triangles
            split_threashold,  // create clusters only with at most this many mesh elements in it
            DiffOp,            // the first-order differential operator belonging to the hi order term of the metric
            AvOp               // the zeroth-order differential operator belonging to the lo order term of the metric
        );
    } // CreateOptimizedBVH_Projectors

    
    inline OptimizedBlockClusterTree *CreateOptimizedBCTFromBVH(OptimizedClusterTree* bvh, double alpha, double beta, double theta, bool exploit_symmetry_ = true, bool upper_triangular_ = false)
    {
        OptimizedBlockClusterTree *bct = new OptimizedBlockClusterTree(
            bvh,   // gets handed two pointers to instances of OptimizedClusterTree
            bvh,   // no problem with handing the same pointer twice; this is actually intended
            alpha, // first parameter of the energy (for the numerator)
            beta,  // second parameter of the energy (for the denominator)
            theta,  // separation parameter; different gauge for thetas as before are the block clustering is performed slightly differently from before
            exploit_symmetry_,
            upper_triangular_
        );

        return bct;
    } // CreateOptimizedBCTFromBVH
} // namespace rsurfaces
