#pragma once

#include "rsurface_types.h"
#include "optimized_bct_types.h"
#include <Eigen/Sparse>
#include <Eigen/Dense>


namespace rsurfaces
{
    Eigen::Matrix<mreal, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> getVertexPositions( MeshPtr mesh, GeomPtr geom );
    
    Eigen::Matrix<mint,  Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> getPrimitiveIndices( MeshPtr mesh, GeomPtr geom );
    
    void AssembleDerivative( Eigen::Matrix<mint, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> const & primitives, Eigen::MatrixXd const & buffer, Eigen::MatrixXd & output, mreal weight = 1. );
    
    void AssembleDerivativeFromACNData( MeshPtr mesh, GeomPtr geom, EigenMatrixRM const & P_D_data, Eigen::MatrixXd & output, mreal weight = 1.);
    
    void AssembleDerivativeFromACPData( MeshPtr mesh, GeomPtr geom, EigenMatrixRM const & P_D_data, Eigen::MatrixXd & output, mreal weight = 1.);
    
    EigenMatrixCSC DerivativeAssembler( MeshPtr mesh, GeomPtr geom, mreal weight = 1. );
    
} // namespace rsurfaces
