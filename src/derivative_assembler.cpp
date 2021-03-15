
#include "derivative_assembler.h"

namespace rsurfaces
{
    Eigen::Matrix<mreal, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> getVertexPositions( MeshPtr mesh, GeomPtr geom )
    {
//        tic("getVertexPositions");
        geom->requireVertexPositions();
        mint n = mesh->nVertices();
        Eigen::Matrix<mreal, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> result ( n, 3 );
        #pragma omp parallel for
        for( mint i = 0; i < n; ++i )
        {
            auto x = geom->inputVertexPositions[i];
            result( i, 0 ) = x[0];
            result( i, 1 ) = x[1];
            result( i, 2 ) = x[2];
            
        }
//        toc("getVertexPositions");
        return result;
    }
    
    Eigen::Matrix<mint, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>  getPrimitiveIndices( MeshPtr mesh, GeomPtr geom )
    {
//        tic("getPrimitiveIndices");
        mint n = mesh->nFaces();
        VertexIndices vInds = mesh->getVertexIndices();
        FaceIndices fInds = mesh->getFaceIndices();
        Eigen::Matrix<mint, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> result ( n, 3 );
        
//        #pragma omp parallel for
        for( auto face : mesh->faces() )
        {
            mint i = fInds[face];
            GCHalfedge he = face.halfedge();
            
            result( i, 0 ) = vInds[he.vertex()];
            result( i, 1 ) = vInds[he.next().vertex()];
            result( i, 2 ) = vInds[he.next().next().vertex()];
            
        }
//        toc("getPrimitiveIndices");
        return result;
    }
    
    EigenMatrixCSC DerivativeAssembler(MeshPtr mesh, GeomPtr geom, mreal weight )
    {
        FaceIndices fInds = mesh->getFaceIndices();
        VertexIndices vInds = mesh->getVertexIndices();

        mint vertex_count = mesh->nVertices();
        mint primitive_count = mesh->nFaces();
        mint primitive_length = 3;

        auto outer  = std::vector<mint> ( primitive_count * primitive_length + 1 , 0);
        auto inner  = std::vector<mint> ( primitive_count * primitive_length, 0);
        auto values = std::vector<mreal>( primitive_count * primitive_length, weight);

        for( auto face : mesh->faces() )
        {
            mint i = fInds[face];

            GCHalfedge he = face.halfedge();

            mint i0 = vInds[he.vertex()];
            mint i1 = vInds[he.next().vertex()];
            mint i2 = vInds[he.next().next().vertex()];
            
            outer[primitive_length * i + 1] = primitive_length * i + 1;
            outer[primitive_length * i + 2] = primitive_length * i + 2;
            outer[primitive_length * i + 3] = primitive_length * i + 3;
            
            inner[primitive_length * i + 0] = i0;
            inner[primitive_length * i + 1] = i1;
            inner[primitive_length * i + 2] = i2;
        }
        return Eigen::Map<Eigen::SparseMatrix<mreal>> ( vertex_count, primitive_count * primitive_length, primitive_count * primitive_length, &outer[0], &inner[0], &values[0] );
    }

    void AssembleDerivative( Eigen::Matrix<mint, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> const & primitives, Eigen::MatrixXd const & buffer, Eigen::MatrixXd & output, mreal weight )
    {
        mint vertex_count = output.rows();
        mint primitive_count = primitives.rows();
        mint primitive_length = primitives.cols();

        mint * outer = mint_iota( primitive_count * primitive_length + 1 );
        mint * inner = mint_alloc( primitive_count * primitive_length );
        mreal * values = mreal_alloc( primitive_count * primitive_length, weight );

        #pragma omp parallel for simd aligned( outer, inner, values : ALIGN)
        for( mint i = 0; i < primitive_count; ++i )
        {
            for( mint j = 0; j < primitive_length; ++j )
            {
                inner[ primitive_length * i + j ] = primitives(i,j);
            }
        }

        // It's crucial here that primitives is row major.
        Eigen::Map<Eigen::SparseMatrix<mreal>> A ( vertex_count, primitive_count * primitive_length, primitive_count * primitive_length, outer, inner, values );

        output += A * buffer;

        mint_free(outer);
        mint_free(inner);
        mreal_free(values);
    }
    
    void AssembleDerivativeFromACNData( MeshPtr mesh, GeomPtr geom, EigenMatrixRM const & P_D_data, Eigen::MatrixXd & output, mreal weight )
    {
        auto V_coords = getVertexPositions( mesh, geom );
        auto primitives = getPrimitiveIndices( mesh, geom );
        
        mint vertex_count = V_coords.rows();
        mint dim = V_coords.cols();
        mint primitive_count = primitives.rows();
        mint primitive_length = primitives.cols();
        
        if( P_D_data.cols() != 7 )
        {
            eprint("in DerivativeAssemblerFromACNData: P_D_data.cols() != 7");
        }
        
        Eigen::MatrixXd buffer ( primitive_count * primitive_length, dim );
        
        for( mint i = 0;  i < primitive_count; ++i )
        {
            
            mint i0 = primitives(i,0);
            mint i1 = primitives(i,1);
            mint i2 = primitives(i,2);
            
            mreal x00 = V_coords(i0,0);
            mreal x01 = V_coords(i0,1);
            mreal x02 = V_coords(i0,2);
            
            mreal x10 = V_coords(i1,0);
            mreal x11 = V_coords(i1,1);
            mreal x12 = V_coords(i1,2);
            
            mreal x20 = V_coords(i2,0);
            mreal x21 = V_coords(i2,1);
            mreal x22 = V_coords(i2,2);
            
            mreal s0 = x11;
            mreal s1 = x20;
            mreal s2 = x10;
            mreal s3 = x21;
            mreal s4 = -s0;
            mreal s5 = s3 + s4;
            mreal s6 = -s1;
            mreal s7 = s2 + s6;
            mreal s8 = x12;
            mreal s9 = x22;
            mreal s10 = x00;
            mreal s11 = -s8;
            mreal s12 = s11 + s9;
            mreal s13 = x01;
            mreal s14 = s13*s7;
            mreal s15 = s0*s1;
            mreal s16 = -(s2*s3);
            mreal s17 = s10*s5;
            mreal s18 = s14 + s15 + s16 + s17;
            mreal s19 = s18*s18;
            mreal s20 = x02;
            mreal s21 = s20*s7;
            mreal s22 = s1*s8;
            mreal s23 = -(s2*s9);
            mreal s24 = s10*s12;
            mreal s25 = s21 + s22 + s23 + s24;
            mreal s26 = s25*s25;
            mreal s27 = -s3;
            mreal s28 = s0 + s27;
            mreal s29 = s20*s28;
            mreal s30 = s3*s8;
            mreal s31 = -(s0*s9);
            mreal s32 = s12*s13;
            mreal s33 = s29 + s30 + s31 + s32;
            mreal s34 = s33*s33;
            mreal s35 = s19 + s26 + s34;
            mreal s36 = sqrt(s35);
            mreal s37 = 1/s36;
            mreal s38 = 2*s18*s5;
            mreal s39 = 2*s12*s25;
            mreal s40 = s38 + s39;
            mreal s41 = P_D_data( i , 0 );
            mreal s42 = s1 + s10 + s2;
            mreal s43 = 2*s18*s7;
            mreal s44 = 2*s12*s33;
            mreal s45 = s43 + s44;
            mreal s46 = P_D_data( i , 1 );
            mreal s47 = s0 + s13 + s3;
            mreal s48 = s36/6.;
            mreal s49 = P_D_data( i , 2 );
            mreal s50 = s20 + s8 + s9;
            mreal s51 = P_D_data( i , 3 );
            mreal s52 = P_D_data( i , 6 );
            mreal s53 = 2*s25*s7;
            mreal s54 = 2*s28*s33;
            mreal s55 = s53 + s54;
            mreal s56 = P_D_data( i , 4 );
            mreal s57 = P_D_data( i , 5 );
            mreal s58 = -s9;
            mreal s59 = s13 + s27;
            mreal s60 = 2*s18*s59;
            mreal s61 = s20 + s58;
            mreal s62 = 2*s25*s61;
            mreal s63 = s60 + s62;
            mreal s64 = -s10;
            mreal s65 = s1 + s64;
            mreal s66 = 2*s18*s65;
            mreal s67 = 2*s33*s61;
            mreal s68 = s66 + s67;
            mreal s69 = -s13;
            mreal s70 = s3 + s69;
            mreal s71 = 2*s25*s65;
            mreal s72 = 2*s33*s70;
            mreal s73 = s71 + s72;
            mreal s74 = -s20;
            mreal s75 = s0 + s69;
            mreal s76 = 2*s18*s75;
            mreal s77 = s74 + s8;
            mreal s78 = 2*s25*s77;
            mreal s79 = s76 + s78;
            mreal s80 = -s2;
            mreal s81 = s10 + s80;
            mreal s82 = 2*s18*s81;
            mreal s83 = 2*s33*s77;
            mreal s84 = s82 + s83;
            mreal s85 = s13 + s4;
            mreal s86 = 2*s25*s81;
            mreal s87 = 2*s33*s85;
            mreal s88 = s86 + s87;
            buffer( 3 * i + 0, 0 ) = (s37*s40*s41)/4. + s46*((s37*s40*s42)/12. + s48) + (s37*s40*s47*s49)/12. + (s37*s40*s50*s51)/12. + (s28*s52)/2. + (s12*s57)/2.;
            buffer( 3 * i + 0, 1 ) = (s37*s41*s45)/4. + (s37*s42*s45*s46)/12. + ((s37*s45*s47)/12. + s48)*s49 + (s37*s45*s50*s51)/12. + (s56*(s58 + s8))/2. + (s52*(s1 + s80))/2.;
            buffer( 3 * i + 0, 2 ) = (s37*s41*s55)/4. + (s37*s42*s46*s55)/12. + (s37*s47*s49*s55)/12. + s51*(s48 + (s37*s50*s55)/12.) + (s5*s56)/2. + (s57*s7)/2.;
            buffer( 3 * i + 1, 0 ) = (s57*s61)/2. + (s37*s41*s63)/4. + (s37*s47*s49*s63)/12. + (s37*s50*s51*s63)/12. + s46*(s48 + (s37*s42*s63)/12.) + (s52*s70)/2.;
            buffer( 3 * i + 1, 1 ) = (s52*(s10 + s6))/2. + (s37*s41*s68)/4. + (s37*s42*s46*s68)/12. + (s37*s50*s51*s68)/12. + s49*(s48 + (s37*s47*s68)/12.) + (s56*(s74 + s9))/2.;
            buffer( 3 * i + 1, 2 ) = (s56*s59)/2. + (s57*s65)/2. + (s37*s41*s73)/4. + (s37*s42*s46*s73)/12. + (s37*s47*s49*s73)/12. + s51*(s48 + (s37*s50*s73)/12.);
            buffer( 3 * i + 2, 0 ) = (s57*s77)/2. + (s37*s41*s79)/4. + (s37*s47*s49*s79)/12. + (s37*s50*s51*s79)/12. + s46*(s48 + (s37*s42*s79)/12.) + (s52*s85)/2.;
            buffer( 3 * i + 2, 1 ) = ((s11 + s20)*s56)/2. + (s52*(s2 + s64))/2. + (s37*s41*s84)/4. + (s37*s42*s46*s84)/12. + (s37*s50*s51*s84)/12. + s49*(s48 + (s37*s47*s84)/12.);
            buffer( 3 * i + 2, 2 ) = (s56*s75)/2. + (s57*s81)/2. + (s37*s41*s88)/4. + (s37*s42*s46*s88)/12. + (s37*s47*s49*s88)/12. + s51*(s48 + (s37*s50*s88)/12.);
        }
        
        AssembleDerivative( primitives, buffer, output, weight );
    }// AssembleDerivativeFromACNData
    
    
    
    void AssembleDerivativeFromACPData( MeshPtr mesh, GeomPtr geom, EigenMatrixRM const & P_D_data, Eigen::MatrixXd & output, mreal weight )
    {
        auto V_coords = getVertexPositions( mesh, geom );
        auto primitives = getPrimitiveIndices( mesh, geom );
        
        mint vertex_count = V_coords.rows();
        mint dim = V_coords.cols();
        mint primitive_count = primitives.rows();
        mint primitive_length = primitives.cols();
        
        if( P_D_data.cols() != 10 )
        {
            eprint("in AssembleDerivativeFromACPData: P_D_data.cols() != 10");
        }
        
        Eigen::MatrixXd buffer ( primitive_count * primitive_length, dim );
        
        #pragma omp parallel for
        for( mint i = 0; i < primitive_count; ++i )
        {
            mreal s0 = V_coords(primitives(i,1),1);
            mreal s1 = V_coords(primitives(i,2),0);
            mreal s2 = V_coords(primitives(i,1),0);
            mreal s3 = V_coords(primitives(i,2),1);
            mreal s4 = -s0;
            mreal s5 = s3 + s4;
            mreal s6 = -s1;
            mreal s7 = s2 + s6;
            mreal s8 = V_coords(primitives(i,1),2);
            mreal s9 = V_coords(primitives(i,2),2);
            mreal s10 = V_coords(primitives(i,0),0);
            mreal s11 = -s8;
            mreal s12 = s11 + s9;
            mreal s13 = V_coords(primitives(i,0),1);
            mreal s14 = s13*s7;
            mreal s15 = s0*s1;
            mreal s16 = -(s2*s3);
            mreal s17 = s10*s5;
            mreal s18 = s14 + s15 + s16 + s17;
            mreal s19 = s18*s18;
            mreal s20 = V_coords(primitives(i,0),2);
            mreal s21 = s20*s7;
            mreal s22 = s1*s8;
            mreal s23 = -(s2*s9);
            mreal s24 = s10*s12;
            mreal s25 = s21 + s22 + s23 + s24;
            mreal s26 = s25*s25;
            mreal s27 = -s3;
            mreal s28 = s0 + s27;
            mreal s29 = s20*s28;
            mreal s30 = s3*s8;
            mreal s31 = -(s0*s9);
            mreal s32 = s12*s13;
            mreal s33 = s29 + s30 + s31 + s32;
            mreal s34 = s33*s33;
            mreal s35 = s19 + s26 + s34;
            mreal s36 = sqrt(s35);
            mreal s37 = 1/s36;
            mreal s38 = 2*s18*s5;
            mreal s39 = 2*s12*s25;
            mreal s40 = s38 + s39;
            mreal s41 = -(s0*s20);
            mreal s42 = s13*s8;
            mreal s43 = s20*s3;
            mreal s44 = -(s3*s8);
            mreal s45 = -(s13*s9);
            mreal s46 = s0*s9;
            mreal s47 = s41 + s42 + s43 + s44 + s45 + s46;
            mreal s48 = s47*s47;
            mreal s49 = -(s13*s2);
            mreal s50 = s0*s10;
            mreal s51 = s1*s13;
            mreal s52 = -(s0*s1);
            mreal s53 = -(s10*s3);
            mreal s54 = s2*s3;
            mreal s55 = s49 + s50 + s51 + s52 + s53 + s54;
            mreal s56 = s55*s55;
            mreal s57 = s2*s20;
            mreal s58 = -(s10*s8);
            mreal s59 = -(s1*s20);
            mreal s60 = s10*s9;
            mreal s61 = s22 + s23 + s57 + s58 + s59 + s60;
            mreal s62 = s61*s61;
            mreal s63 = s48 + s56 + s62;
            mreal s64 = 1/s63;
            mreal s65 = s63*s63;
            mreal s66 = 1/s65;
            mreal s67 = 2*s28*s55;
            mreal s68 = 2*s12*s61;
            mreal s69 = s67 + s68;
            mreal s70 = P_D_data( i , 0 );
            mreal s71 = s1 + s10 + s2;
            mreal s72 = 2*s18*s7;
            mreal s73 = 2*s12*s33;
            mreal s74 = s72 + s73;
            mreal s75 = P_D_data( i , 1 );
            mreal s76 = s0 + s13 + s3;
            mreal s77 = s36/6.;
            mreal s78 = P_D_data( i , 2 );
            mreal s79 = s20 + s8 + s9;
            mreal s80 = P_D_data( i , 3 );
            mreal s81 = -s9;
            mreal s82 = s8 + s81;
            mreal s83 = P_D_data( i , 4 );
            mreal s84 = -s2;
            mreal s85 = s1 + s84;
            mreal s86 = 2*s55*s85;
            mreal s87 = 2*s47*s82;
            mreal s88 = s86 + s87;
            mreal s89 = P_D_data( i , 5 );
            mreal s90 = P_D_data( i , 6 );
            mreal s91 = P_D_data( i , 7 );
            mreal s92 = P_D_data( i , 8 );
            mreal s93 = P_D_data( i , 9 );
            mreal s94 = 2*s25*s7;
            mreal s95 = 2*s28*s33;
            mreal s96 = s94 + s95;
            mreal s97 = 2*s61*s7;
            mreal s98 = 2*s47*s5;
            mreal s99 = s97 + s98;
            mreal s100 = s13 + s27;
            mreal s101 = 2*s100*s18;
            mreal s102 = s20 + s81;
            mreal s103 = 2*s102*s25;
            mreal s104 = s101 + s103;
            mreal s105 = -s13;
            mreal s106 = s105 + s3;
            mreal s107 = 2*s106*s55;
            mreal s108 = 2*s102*s61;
            mreal s109 = s107 + s108;
            mreal s110 = -s10;
            mreal s111 = s1 + s110;
            mreal s112 = 2*s111*s18;
            mreal s113 = 2*s102*s33;
            mreal s114 = s112 + s113;
            mreal s115 = -s20;
            mreal s116 = s115 + s9;
            mreal s117 = s10 + s6;
            mreal s118 = 2*s117*s55;
            mreal s119 = 2*s116*s47;
            mreal s120 = s118 + s119;
            mreal s121 = 2*s111*s25;
            mreal s122 = 2*s106*s33;
            mreal s123 = s121 + s122;
            mreal s124 = 2*s111*s61;
            mreal s125 = 2*s100*s47;
            mreal s126 = s124 + s125;
            mreal s127 = s0 + s105;
            mreal s128 = 2*s127*s18;
            mreal s129 = s115 + s8;
            mreal s130 = 2*s129*s25;
            mreal s131 = s128 + s130;
            mreal s132 = s13 + s4;
            mreal s133 = 2*s132*s55;
            mreal s134 = 2*s129*s61;
            mreal s135 = s133 + s134;
            mreal s136 = s10 + s84;
            mreal s137 = 2*s136*s18;
            mreal s138 = 2*s129*s33;
            mreal s139 = s137 + s138;
            mreal s140 = s11 + s20;
            mreal s141 = s110 + s2;
            mreal s142 = 2*s141*s55;
            mreal s143 = 2*s140*s47;
            mreal s144 = s142 + s143;
            mreal s145 = 2*s136*s25;
            mreal s146 = 2*s132*s33;
            mreal s147 = s145 + s146;
            mreal s148 = 2*s136*s61;
            mreal s149 = 2*s127*s47;
            mreal s150 = s148 + s149;
            buffer( 3 * i + 0, 0 ) = (s37*s40*s70)/4. + s75*((s37*s40*s71)/12. + s77) + (s37*s40*s76*s78)/12. + (s37*s40*s79*s80)/12. + ((s37*s40*s48*s64)/4. - (s36*s48*s66*s69)/2.)*s83 + ((s12*s36*s47*s64)/2. + (s37*s40*s47*s61*s64)/4. - (s36*s47*s61*s66*s69)/2.)*s89 + ((s28*s36*s47*s64)/2. + (s37*s40*s47*s55*s64)/4. - (s36*s47*s55*s66*s69)/2.)*s90 + (s12*s36*s61*s64 + (s37*s40*s62*s64)/4. - (s36*s62*s66*s69)/2.)*s91 + ((s12*s36*s55*s64)/2. + (s28*s36*s61*s64)/2. + (s37*s40*s55*s61*s64)/4. - (s36*s55*s61*s66*s69)/2.)*s92 + (s28*s36*s55*s64 + (s37*s40*s56*s64)/4. - (s36*s56*s66*s69)/2.)*s93;
            buffer( 3 * i + 0, 1 ) = (s37*s70*s74)/4. + (s37*s71*s74*s75)/12. + ((s37*s74*s76)/12. + s77)*s78 + (s37*s74*s79*s80)/12. + s83*((s37*s48*s64*s74)/4. + s36*s47*s64*s82 - (s36*s48*s66*s88)/2.) + ((s37*s47*s61*s64*s74)/4. + (s36*s61*s64*s82)/2. - (s36*s47*s61*s66*s88)/2.)*s89 + ((s37*s47*s55*s64*s74)/4. + (s36*s55*s64*s82)/2. + (s36*s47*s64*s85)/2. - (s36*s47*s55*s66*s88)/2.)*s90 + ((s37*s62*s64*s74)/4. - (s36*s62*s66*s88)/2.)*s91 + ((s37*s55*s61*s64*s74)/4. + (s36*s61*s64*s85)/2. - (s36*s55*s61*s66*s88)/2.)*s92 + ((s37*s56*s64*s74)/4. + s36*s55*s64*s85 - (s36*s56*s66*s88)/2.)*s93;
            buffer( 3 * i + 0, 2 ) = (s37*s70*s96)/4. + (s37*s71*s75*s96)/12. + (s37*s76*s78*s96)/12. + s80*(s77 + (s37*s79*s96)/12.) + s83*(s36*s47*s5*s64 + (s37*s48*s64*s96)/4. - (s36*s48*s66*s99)/2.) + s90*((s36*s5*s55*s64)/2. + (s37*s47*s55*s64*s96)/4. - (s36*s47*s55*s66*s99)/2.) + s93*((s37*s56*s64*s96)/4. - (s36*s56*s66*s99)/2.) + s89*((s36*s5*s61*s64)/2. + (s36*s47*s64*s7)/2. + (s37*s47*s61*s64*s96)/4. - (s36*s47*s61*s66*s99)/2.) + s92*((s36*s55*s64*s7)/2. + (s37*s55*s61*s64*s96)/4. - (s36*s55*s61*s66*s99)/2.) + s91*(s36*s61*s64*s7 + (s37*s62*s64*s96)/4. - (s36*s62*s66*s99)/2.);
            buffer( 3 * i + 1, 0 ) = (s104*s37*s70)/4. + s75*((s104*s37*s71)/12. + s77) + (s104*s37*s76*s78)/12. + (s104*s37*s79*s80)/12. + ((s104*s37*s48*s64)/4. - (s109*s36*s48*s66)/2.)*s83 + ((s102*s36*s47*s64)/2. + (s104*s37*s47*s61*s64)/4. - (s109*s36*s47*s61*s66)/2.)*s89 + ((s106*s36*s47*s64)/2. + (s104*s37*s47*s55*s64)/4. - (s109*s36*s47*s55*s66)/2.)*s90 + (s102*s36*s61*s64 + (s104*s37*s62*s64)/4. - (s109*s36*s62*s66)/2.)*s91 + ((s102*s36*s55*s64)/2. + (s106*s36*s61*s64)/2. + (s104*s37*s55*s61*s64)/4. - (s109*s36*s55*s61*s66)/2.)*s92 + (s106*s36*s55*s64 + (s104*s37*s56*s64)/4. - (s109*s36*s56*s66)/2.)*s93;
            buffer( 3 * i + 1, 1 ) = (s114*s37*s70)/4. + (s114*s37*s71*s75)/12. + ((s114*s37*s76)/12. + s77)*s78 + (s114*s37*s79*s80)/12. + (s116*s36*s47*s64 + (s114*s37*s48*s64)/4. - (s120*s36*s48*s66)/2.)*s83 + ((s116*s36*s61*s64)/2. + (s114*s37*s47*s61*s64)/4. - (s120*s36*s47*s61*s66)/2.)*s89 + ((s117*s36*s47*s64)/2. + (s116*s36*s55*s64)/2. + (s114*s37*s47*s55*s64)/4. - (s120*s36*s47*s55*s66)/2.)*s90 + ((s114*s37*s62*s64)/4. - (s120*s36*s62*s66)/2.)*s91 + ((s117*s36*s61*s64)/2. + (s114*s37*s55*s61*s64)/4. - (s120*s36*s55*s61*s66)/2.)*s92 + (s117*s36*s55*s64 + (s114*s37*s56*s64)/4. - (s120*s36*s56*s66)/2.)*s93;
            buffer( 3 * i + 1, 2 ) = (s123*s37*s70)/4. + (s123*s37*s71*s75)/12. + (s123*s37*s76*s78)/12. + (s77 + (s123*s37*s79)/12.)*s80 + (s100*s36*s47*s64 + (s123*s37*s48*s64)/4. - (s126*s36*s48*s66)/2.)*s83 + ((s111*s36*s47*s64)/2. + (s100*s36*s61*s64)/2. + (s123*s37*s47*s61*s64)/4. - (s126*s36*s47*s61*s66)/2.)*s89 + ((s100*s36*s55*s64)/2. + (s123*s37*s47*s55*s64)/4. - (s126*s36*s47*s55*s66)/2.)*s90 + (s111*s36*s61*s64 + (s123*s37*s62*s64)/4. - (s126*s36*s62*s66)/2.)*s91 + ((s111*s36*s55*s64)/2. + (s123*s37*s55*s61*s64)/4. - (s126*s36*s55*s61*s66)/2.)*s92 + ((s123*s37*s56*s64)/4. - (s126*s36*s56*s66)/2.)*s93;
            buffer( 3 * i + 2, 0 ) = (s131*s37*s70)/4. + s75*((s131*s37*s71)/12. + s77) + (s131*s37*s76*s78)/12. + (s131*s37*s79*s80)/12. + ((s131*s37*s48*s64)/4. - (s135*s36*s48*s66)/2.)*s83 + ((s129*s36*s47*s64)/2. + (s131*s37*s47*s61*s64)/4. - (s135*s36*s47*s61*s66)/2.)*s89 + ((s132*s36*s47*s64)/2. + (s131*s37*s47*s55*s64)/4. - (s135*s36*s47*s55*s66)/2.)*s90 + (s129*s36*s61*s64 + (s131*s37*s62*s64)/4. - (s135*s36*s62*s66)/2.)*s91 + ((s129*s36*s55*s64)/2. + (s132*s36*s61*s64)/2. + (s131*s37*s55*s61*s64)/4. - (s135*s36*s55*s61*s66)/2.)*s92 + (s132*s36*s55*s64 + (s131*s37*s56*s64)/4. - (s135*s36*s56*s66)/2.)*s93;
            buffer( 3 * i + 2, 1 ) = (s139*s37*s70)/4. + (s139*s37*s71*s75)/12. + ((s139*s37*s76)/12. + s77)*s78 + (s139*s37*s79*s80)/12. + (s140*s36*s47*s64 + (s139*s37*s48*s64)/4. - (s144*s36*s48*s66)/2.)*s83 + ((s140*s36*s61*s64)/2. + (s139*s37*s47*s61*s64)/4. - (s144*s36*s47*s61*s66)/2.)*s89 + ((s141*s36*s47*s64)/2. + (s140*s36*s55*s64)/2. + (s139*s37*s47*s55*s64)/4. - (s144*s36*s47*s55*s66)/2.)*s90 + ((s139*s37*s62*s64)/4. - (s144*s36*s62*s66)/2.)*s91 + ((s141*s36*s61*s64)/2. + (s139*s37*s55*s61*s64)/4. - (s144*s36*s55*s61*s66)/2.)*s92 + (s141*s36*s55*s64 + (s139*s37*s56*s64)/4. - (s144*s36*s56*s66)/2.)*s93;
            buffer( 3 * i + 2, 2 ) = (s147*s37*s70)/4. + (s147*s37*s71*s75)/12. + (s147*s37*s76*s78)/12. + (s77 + (s147*s37*s79)/12.)*s80 + (s127*s36*s47*s64 + (s147*s37*s48*s64)/4. - (s150*s36*s48*s66)/2.)*s83 + ((s136*s36*s47*s64)/2. + (s127*s36*s61*s64)/2. + (s147*s37*s47*s61*s64)/4. - (s150*s36*s47*s61*s66)/2.)*s89 + ((s127*s36*s55*s64)/2. + (s147*s37*s47*s55*s64)/4. - (s150*s36*s47*s55*s66)/2.)*s90 + (s136*s36*s61*s64 + (s147*s37*s62*s64)/4. - (s150*s36*s62*s66)/2.)*s91 + ((s136*s36*s55*s64)/2. + (s147*s37*s55*s61*s64)/4. - (s150*s36*s55*s61*s66)/2.)*s92 + ((s147*s37*s56*s64)/4. - (s150*s36*s56*s66)/2.)*s93;

        }
        
        AssembleDerivative( primitives, buffer, output, weight );
        
    }// AssembleDerivativeFromACPData
    
} // namespace rsurfaces
