#pragma once

// This file is a compatibility layer between MeshPtr+GeomPtr and OptimizedClusterTree+OptimizedBlockClusterTree.
// The latter two classes are agnostic to what kind of meshes they are used with, so they can be applied also to polyline mesh, splines, higher order FEM meshes, NURBS,...


// To create a OptimizedClusterTree, essentially one has to  hand over the following data for each primitive (triangle, edge, finite element, NURBS patch...)
//  - clustering coordinates in "clustering space" (which need not be the embedding space)
//  - convex hulls for the primitive (in terms of a points in "clustering space" that span the hull )
//  - each a list for far field and near field data
//  - a derivative operator AvOp and an averaging operator DiffOp that handle the pre- and postprocessing for matrix-vector multiplication
// To create a OptimizedBlockClusterTree, essentially one has to hand over only two pointers to instances of OptimizedClusterTree.

// This header file is supposed to create this data for MeshPtr+GeomPtr embedded in 3D space.

#include "optimized_bct.h"
//#include "optimized_cluster_tree.h"
#include "sobolev/hs_operators.h"

namespace rsurfaces
{

    inline OptimizedClusterTree * CreateOptimizedBVH_Hybrid(MeshPtr &mesh, GeomPtr &geom)
    {
        geom->requireFaceAreas();
        geom->requireFaceNormals();
        
        int vertex_count = mesh->nVertices();
        int primitive_count = mesh->nFaces();
        int primitive_length = 3;
        FaceIndices fInds = mesh->getFaceIndices();
        VertexIndices vInds = mesh->getVertexIndices();

        int dim = 3;
        int near_dim = 1 + dim + dim;
        int far_dim = 1 + dim + dim * (dim + 1)/2;
        
        double athird = 1. / primitive_length;

        std::vector<int> ordering(primitive_count);
        std::vector<double> P_coords(primitive_count * dim );
        std::vector<double> P_hull_coords(primitive_count * primitive_length * dim);

        MKLSparseMatrix AvOp = MKLSparseMatrix( primitive_count, vertex_count, 3 * primitive_count ); // This is a sparse matrix in CSR format.
        AvOp.outer[primitive_count] = primitive_length * primitive_count;
        
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

            P_coords[dim * i + 0] = athird * (p1.x + p2.x + p3.x);
            P_coords[dim * i + 1] = athird * (p1.y + p2.y + p3.y);
            P_coords[dim * i + 2] = athird * (p1.z + p2.z + p3.z);

            P_far[far_dim * i + 0] = P_near[near_dim * i + 0] = geom->faceAreas[face];
            P_far[far_dim * i + 1] = P_near[near_dim * i + 1] = P_coords[dim * i + 0];
            P_far[far_dim * i + 2] = P_near[near_dim * i + 2] = P_coords[dim * i + 1];
            P_far[far_dim * i + 3] = P_near[near_dim * i + 3] = P_coords[dim * i + 2];
            
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

            P_hull_coords[primitive_length * dim * i + 0] = p1.x;
            P_hull_coords[primitive_length * dim * i + 1] = p1.y;
            P_hull_coords[primitive_length * dim * i + 2] = p1.z;
            P_hull_coords[primitive_length * dim * i + 3] = p2.x;
            P_hull_coords[primitive_length * dim * i + 4] = p2.y;
            P_hull_coords[primitive_length * dim * i + 5] = p2.z;
            P_hull_coords[primitive_length * dim * i + 6] = p3.x;
            P_hull_coords[primitive_length * dim * i + 7] = p3.y;
            P_hull_coords[primitive_length * dim * i + 8] = p3.z;

            AvOp.outer[i] = primitive_length * i;

            AvOp.inner[primitive_length * i + 0] = i0;
            AvOp.inner[primitive_length * i + 1] = i1;
            AvOp.inner[primitive_length * i + 2] = i2;
        
            AvOp.values[primitive_length * i + 0] = athird;
            AvOp.values[primitive_length * i + 1] = athird;
            AvOp.values[primitive_length * i + 2] = athird;

            std::sort( AvOp.inner + primitive_length * i, AvOp.inner + primitive_length * (i + 1) );
        }
        
        EigenMatrixCSR DiffOp0 = Hs::BuildDfOperator(mesh, geom); // This is a sparse matrix in CSC!!! format.
        
        DiffOp0.makeCompressed();
        
        MKLSparseMatrix DiffOp = MKLSparseMatrix( DiffOp0.rows(), DiffOp0.cols(), DiffOp0.outerIndexPtr(), DiffOp0.innerIndexPtr(), DiffOp0.valuePtr() ); // This is a sparse matrix in CSR format.

        // create a cluster tree
        return new OptimizedClusterTree(
            &P_coords[0],      // coordinates used for clustering
            primitive_count,            // number of primitives
            dim,                 // dimension of ambient space
            &P_hull_coords[0], // coordinates of the convex hull of each mesh element
            primitive_length,                 // number of points in the convex hull of each mesh element (3 for triangle meshes, 2 for polylines)
            &P_near[0],        // area, barycenter, and normal of mesh element
            near_dim,                 // number of dofs of P_near per mesh element; it is 7 for polylines and triangle meshes in 3D.
            &P_far[0],        // area, barycenter, and projector of mesh element
            far_dim,                 // number of dofs of P_far per mesh element; it is 10 for polylines and triangle meshes in 3D.
            &ordering[0],      // some ordering of triangles
            DiffOp,            // the first-order differential operator belonging to the hi order term of the metric
            AvOp               // the zeroth-order differential operator belonging to the lo order term of the metric
        );
    } // CreateOptimizedBVH
    
    inline OptimizedClusterTree * CreateOptimizedBVH_Normals(MeshPtr &mesh, GeomPtr &geom)
    {
        geom->requireFaceAreas();
        geom->requireFaceNormals();

        
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
            &ordering[0],      // some ordering of triangles
            DiffOp,            // the first-order differential operator belonging to the hi order term of the metric
            AvOp               // the zeroth-order differential operator belonging to the lo order term of the metric
        );
    } // CreateOptimizedBVH_Normls

    inline void UpdateOptimizedBVH(OptimizedClusterTree * bvh, MeshPtr &mesh, GeomPtr &geom)
    {
//        bvh->UpdateWithNewPositions(mesh, geom);
        
        geom->requireFaceAreas();
        geom->requireFaceNormals();
        
        mint nVertices = mesh->nVertices();
        mint nFaces = mesh->nFaces();
        FaceIndices fInds = mesh->getFaceIndices();
        VertexIndices vInds = mesh->getVertexIndices();

        mreal athird = 1. / 3.;

        if( bvh->far_dim == 7 && bvh->near_dim == 7)
        {
            std::vector<mreal> P_near_(7 * nFaces);
            std::vector<mreal> P_far_(7 * nFaces);
            
            for (auto face : mesh->faces())
            {
                mint i = fInds[face];
                
                GCHalfedge he = face.halfedge();
                
                mint i0 = vInds[he.vertex()];
                mint i1 = vInds[he.next().vertex()];
                mint i2 = vInds[he.next().next().vertex()];
                Vector3 p1 = geom->inputVertexPositions[i0];
                Vector3 p2 = geom->inputVertexPositions[i1];
                Vector3 p3 = geom->inputVertexPositions[i2];
                
                P_far_[7 * i + 0] = P_near_[7 * i + 0] = geom->faceAreas[face];
                P_far_[7 * i + 0] = P_near_[7 * i + 1] = athird * (p1.x + p2.x + p3.x);
                P_far_[7 * i + 0] = P_near_[7 * i + 2] = athird * (p1.y + p2.y + p3.y);
                P_far_[7 * i + 0] = P_near_[7 * i + 3] = athird * (p1.z + p2.z + p3.z);
                P_far_[7 * i + 0] = P_near_[7 * i + 4] = geom->faceNormals[face].x;
                P_far_[7 * i + 0] = P_near_[7 * i + 5] = geom->faceNormals[face].y;
                P_far_[7 * i + 0] = P_near_[7 * i + 6] = geom->faceNormals[face].z;
            }
            bvh->SemiStaticUpdate( &P_near_[0], &P_far_[0] );
        }
        else
        {
            if( bvh->far_dim == 10 && bvh->near_dim == 10)
            {
                std::vector<mreal> P_near_(10 * nFaces);
                std::vector<mreal> P_far_(10 * nFaces);
                
                for (auto face : mesh->faces())
                {
                    mint i = fInds[face];
                    
                    GCHalfedge he = face.halfedge();
                    
                    mint i0 = vInds[he.vertex()];
                    mint i1 = vInds[he.next().vertex()];
                    mint i2 = vInds[he.next().next().vertex()];
                    Vector3 p1 = geom->inputVertexPositions[i0];
                    Vector3 p2 = geom->inputVertexPositions[i1];
                    Vector3 p3 = geom->inputVertexPositions[i2];
                    
                    P_far_[10 * i + 0] = P_near_[10 * i + 0] = geom->faceAreas[face];
                    P_far_[10 * i + 0] = P_near_[10 * i + 0] = athird * (p1.x + p2.x + p3.x);
                    P_far_[10 * i + 0] = P_near_[10 * i + 0] = athird * (p1.y + p2.y + p3.y);
                    P_far_[10 * i + 0] = P_near_[10 * i + 0] = athird * (p1.z + p2.z + p3.z);
                    
                    mreal n1 = geom->faceNormals[face].x;
                    mreal n2 = geom->faceNormals[face].y;
                    mreal n3 = geom->faceNormals[face].z;
                    
                    P_near_[10 * i + 4] = P_far_[10 * i + 4] = n1 * n1;
                    P_near_[10 * i + 4] = P_far_[10 * i + 5] = n1 * n2;
                    P_near_[10 * i + 4] = P_far_[10 * i + 6] = n1 * n3;
                    P_near_[10 * i + 4] = P_far_[10 * i + 7] = n2 * n2;
                    P_near_[10 * i + 4] = P_far_[10 * i + 8] = n2 * n3;
                    P_near_[10 * i + 4] = P_far_[10 * i + 9] = n3 * n3;
                }
            }
            else
            {
                std::vector<mreal> P_near_(7 * nFaces);
                std::vector<mreal> P_far_(10 * nFaces);
                
                for (auto face : mesh->faces())
                {
                    mint i = fInds[face];
                    
                    GCHalfedge he = face.halfedge();
                    
                    mint i0 = vInds[he.vertex()];
                    mint i1 = vInds[he.next().vertex()];
                    mint i2 = vInds[he.next().next().vertex()];
                    Vector3 p1 = geom->inputVertexPositions[i0];
                    Vector3 p2 = geom->inputVertexPositions[i1];
                    Vector3 p3 = geom->inputVertexPositions[i2];
                    
                    P_far_[10 * i + 0] = P_near_[7 * i + 0] = geom->faceAreas[face];
                    P_far_[10 * i + 0] = P_near_[7 * i + 0] = athird * (p1.x + p2.x + p3.x);
                    P_far_[10 * i + 0] = P_near_[7 * i + 0] = athird * (p1.y + p2.y + p3.y);
                    P_far_[10 * i + 0] = P_near_[7 * i + 0] = athird * (p1.z + p2.z + p3.z);
                    
                    mreal n1 = geom->faceNormals[face].x;
                    mreal n2 = geom->faceNormals[face].y;
                    mreal n3 = geom->faceNormals[face].z;
                    
                    P_near_[7 * i + 4] = n1;
                    P_near_[7 * i + 5] = n2;
                    P_near_[7 * i + 6] = n3;
                    
                    P_far_[10 * i + 4] = n1 * n1;
                    P_far_[10 * i + 5] = n1 * n2;
                    P_far_[10 * i + 6] = n1 * n3;
                    P_far_[10 * i + 7] = n2 * n2;
                    P_far_[10 * i + 8] = n2 * n3;
                    P_far_[10 * i + 9] = n3 * n3;

                }
                bvh->SemiStaticUpdate( &P_near_[0], &P_far_[0] );
            }
        }
    } // UpdateOptimizedBVH

    inline OptimizedClusterTree * CreateOptimizedBVH_Projectors(MeshPtr &mesh, GeomPtr &geom)
    {
        geom->requireFaceAreas();
        geom->requireFaceNormals();
        
        int vertex_count = mesh->nVertices();
        int primitive_count = mesh->nFaces();
        int dim = 3;
        int near_dim = 1 + dim + dim * (dim + 1)/2;
        int far_dim = 1 + dim + dim * (dim + 1)/2;
        int primitive_length = 3;
        
        FaceIndices fInds = mesh->getFaceIndices();
        VertexIndices vInds = mesh->getVertexIndices();

        double athird = 1. / primitive_length;

        std::vector<int> ordering(primitive_count);
        std::vector<double> P_coords(primitive_count * dim);
        std::vector<double> P_hull_coords(primitive_count * primitive_length * dim);

        MKLSparseMatrix AvOp = MKLSparseMatrix( primitive_count, vertex_count, primitive_length * primitive_count ); // This is a sparse matrix in CSR format.
        AvOp.outer[primitive_count] = primitive_length * primitive_count;
        
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

            P_coords[dim * i + 0] = athird * (p1.x + p2.x + p3.x);
            P_coords[dim * i + 1] = athird * (p1.y + p2.y + p3.y);
            P_coords[dim * i + 2] = athird * (p1.z + p2.z + p3.z);

            P_far[far_dim * i + 0] = P_near[near_dim * i + 0] = geom->faceAreas[face];
            P_far[far_dim * i + 1] = P_near[near_dim * i + 1] = P_coords[dim * i + 0];
            P_far[far_dim * i + 2] = P_near[near_dim * i + 2] = P_coords[dim * i + 1];
            P_far[far_dim * i + 3] = P_near[near_dim * i + 3] = P_coords[dim * i + 2];
            
            mreal n1 = geom->faceNormals[face].x;
            mreal n2 = geom->faceNormals[face].y;
            mreal n3 = geom->faceNormals[face].z;
            
            P_far[far_dim * i + 4] = P_near[near_dim * i + 4] = n1 * n1;
            P_far[far_dim * i + 5] = P_near[near_dim * i + 5] = n1 * n2;
            P_far[far_dim * i + 6] = P_near[near_dim * i + 6] = n1 * n3;
            P_far[far_dim * i + 7] = P_near[near_dim * i + 7] = n2 * n2;
            P_far[far_dim * i + 8] = P_near[near_dim * i + 8] = n2 * n3;
            P_far[far_dim * i + 9] = P_near[near_dim * i + 9] = n3 * n3;

            P_hull_coords[primitive_length * dim * i + 0] = p1.x;
            P_hull_coords[primitive_length * dim * i + 1] = p1.y;
            P_hull_coords[primitive_length * dim * i + 2] = p1.z;
            P_hull_coords[primitive_length * dim * i + 3] = p2.x;
            P_hull_coords[primitive_length * dim * i + 4] = p2.y;
            P_hull_coords[primitive_length * dim * i + 5] = p2.z;
            P_hull_coords[primitive_length * dim * i + 6] = p3.x;
            P_hull_coords[primitive_length * dim * i + 7] = p3.y;
            P_hull_coords[primitive_length * dim * i + 8] = p3.z;

            AvOp.outer[i] = primitive_length * i;
            
            AvOp.inner[primitive_length * i + 0] = i0;
            AvOp.inner[primitive_length * i + 1] = i1;
            AvOp.inner[primitive_length * i + 2] = i2;
        
            AvOp.values[primitive_length * i + 0] = athird;
            AvOp.values[primitive_length * i + 1] = athird;
            AvOp.values[primitive_length * i + 2] = athird;

            std::sort( AvOp.inner + primitive_length * i, AvOp.inner + primitive_length * (i + 1) );
        }
        
        EigenMatrixCSR DiffOp0 = Hs::BuildDfOperator(mesh, geom); // This is a sparse matrix in CSC!!! format.
        
        DiffOp0.makeCompressed();
        
        MKLSparseMatrix DiffOp = MKLSparseMatrix( DiffOp0.rows(), DiffOp0.cols(), DiffOp0.outerIndexPtr(), DiffOp0.innerIndexPtr(), DiffOp0.valuePtr() ); // This is a sparse matrix in CSR format.

        // create a cluster tree
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
            &ordering[0],      // some ordering of triangles
            DiffOp,            // the first-order differential operator belonging to the hi order term of the metric
            AvOp               // the zeroth-order differential operator belonging to the lo order term of the metric
        );
    } // CreateOptimizedBVH_Projectors

    inline OptimizedClusterTree * CreateOptimizedBVH(MeshPtr &mesh, GeomPtr &geom)
    {
//        return CreateOptimizedBVH_Hybrid(mesh, geom);
        return CreateOptimizedBVH_Normals(mesh, geom);
    }
    
    inline OptimizedBlockClusterTree * CreateOptimizedBCTFromBVH(OptimizedClusterTree* bvh, double alpha, double beta, double chi)
    {
        OptimizedBlockClusterTree *bct = new OptimizedBlockClusterTree(
            bvh,   // gets handed two pointers to instances of OptimizedClusterTree
            bvh,   // no problem with handing the same pointer twice; this is actually intended
            alpha, // first parameter of the energy (for the numerator)
            beta,  // second parameter of the energy (for the denominator)
            chi  // separation parameter; different gauge for thetas as before are the block clustering is performed slightly differently from before
        );

        return bct;
    } // CreateOptimizedBCTFromBVH
} // namespace rsurfaces
