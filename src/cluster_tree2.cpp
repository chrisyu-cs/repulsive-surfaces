#include "cluster_tree2.h"

namespace rsurfaces
{

    Cluster2::Cluster2(mint begin_, mint end_, mint depth_)
    {
        begin = begin_;
        end = end_;
        depth = depth_;
        descendant_count = 0;
        descendant_leaf_count = 0;
        left = nullptr;
        right = nullptr;
    }; // struct Cluster2

    // Solving interface problems by using standard types. This way, things are easier to port. For example, I can call this from Mathematica for faster debugging.
    ClusterTree2::ClusterTree2(
        mreal const *const P_coords_, // coordinates per primitive used for clustering; assumed to be of size primitive_count x dim
        mint primitive_count_,
        mint dim_,
        mreal const *const P_hull_coords_, // points that define the convex hulls of primitives; assumed to be array of size primitive_count x hull_count x dim
        mint hull_count_,
        mreal const *const P_data_, // data used actual interaction computation; assumed to be of size primitive_count x data_dim. For a triangle mesh in 3D, we want to feed each triangles i), area ii) barycenter and iii) normal as a 1 + 3 + 3 = 7 vector
        mint data_dim_,
        const mint max_buffer_dim_,
        mint const *const ordering_, // A suggested preordering of primitives; this gets applied before the clustering begins in the hope that this may improve the sorting within a cluster --- at least in the top level(s). This could, e.g., be the ordering obtained by a tree for  similar data set.
        mint split_threshold_,       // split a cluster if has this many or more primitives contained in it
                                     //            EigenMatrixCSC & DiffOp,        // CSC
                                     //            EigenMatrixCSC & AvOp           // CSC
        EigenMatrixCSR &DiffOp,      // CSR
        EigenMatrixCSR &AvOp         // CSR
    )
    {
        // tic("Initializing ClusterTree2");

        //        omp_set_num_threads(1);

        primitive_count = primitive_count_;
        hull_count = hull_count_;
        dim = dim_;
        data_dim_ = data_dim;
        max_buffer_dim = 0;

#pragma omp parallel
        {
            thread_count = omp_get_num_threads();
        }
        // print("thread_count = " + std::to_string(thread_count) );

        mint a = 1;
        split_threshold = std::max(a, split_threshold_);

        P_ext_pos = A_Vector<mint>(&ordering_[0], &ordering_[primitive_count]);

        P_coords = A_Vector<A_Vector<mreal>>(dim, A_Vector<mreal>(primitive_count));

        // print("primitive_count = " + std::to_string(primitive_count));

        //    tic("Copying");
#pragma omp parallel for num_threads(thread_count) default(none) shared(P_coords, P_ext_pos, P_coords_, dim, primitive_count)
        for (mint i = 0; i < primitive_count; ++i)
        {
            mint j = P_ext_pos[i];
            for (mint k = 0, last = dim; k < last; ++k)
            {
                P_coords[k][i] = P_coords_[dim * j + k];
            }
        }
        //    toc("Copying");

        //    tic("SplitCluster");
        std::shared_ptr<Cluster2> root = std::make_shared<Cluster2>(0, primitive_count, 0);

#pragma omp parallel num_threads(thread_count) default(none) shared(root, P_coords, P_ext_pos, thread_count)
        {
#pragma omp single
            {
                SplitCluster(root.get(), thread_count);
            }
        }

        // tic("Many small allocations");
        cluster_count = root->descendant_count;
        leaf_cluster_count = root->descendant_leaf_count;

        // TODO: Create parallel tasks here.
        PrepareBuffers(std::max(dim * dim, max_buffer_dim_));

        C_left = A_Vector<mint>(cluster_count, -1);
        C_right = A_Vector<mint>(cluster_count, -1);
        C_begin = A_Vector<mint>(cluster_count, -1);
        C_end = A_Vector<mint>(cluster_count, -1);
        C_depth = A_Vector<mint>(cluster_count, -1);
        //    C_ID    = A_Vector<mint> ( cluster_count );
        //    C_one   = A_Vector<mreal>( cluster_count, 1. );

        leaf_clusters = A_Vector<mint>(leaf_cluster_count);
        leaf_cluster_lookup = A_Vector<mint>(cluster_count, -1);
        //    toc("SplitCluster");

        // toc("Many small allocations");

        // print( "cluster_count = " + std::to_string(cluster_count) );
        // print( "leaf_cluster_count = " + std::to_string(leaf_cluster_count) );

        //    tic("Serialize");
#pragma omp parallel num_threads(thread_count) default(none) shared(root, thread_count)
        {
#pragma omp single
            {
                Serialize(root.get(), 0, 0, thread_count);
            }
        }
        //    toc("Serialize");

        inverse_ordering = A_Vector<mint>(primitive_count);
#pragma omp parallel for
        for (mint i = 0; i < primitive_count; ++i)
        {
            inverse_ordering[P_ext_pos[i]] = i;
        }

        leaf_cluster_ptr = A_Vector<mint>(leaf_cluster_count + 1);
        P_leaf = A_Vector<mint>(primitive_count);
#pragma omp parallel for
        for (mint i = 0; i < leaf_cluster_count; ++i)
        {
            mint leaf = leaf_clusters[i];
            mint begin = C_begin[leaf];
            mint end = C_end[leaf];
            leaf_cluster_ptr[i + 1] = end;
            for (mint k = begin; k < end; ++k)
            {
                P_leaf[k] = leaf;
            }
        }

        //    tic("ComputePrimitiveData");
        ComputePrimitiveData(P_hull_coords_, P_data_);
        //    toc("ComputePrimitiveData");

        //    tic("ComputeClusterData");
        ComputeClusterData();
        //    toc("ComputeClusterData");

        // TODO: Fuse permutaton and weighting into that!

        //    // Eigen is insanely slow in multiplying pre and post processors.
        //    // Converting to MKL version.

        // TODO: Fuse permutation and weighting into that!

        // tic("Create pre and post");

        A_Vector<mreal> P_one(primitive_count, 1.);
        A_Vector<mreal> P_iota(primitive_count + 1, 0.);
        A_Vector<mint> P_rp(primitive_count + 1, 0);
        A_Vector<mint> C_rp(cluster_count + 1, 0);

#pragma omp parallel for
        for (mint i = 0; i < primitive_count + 1; ++i)
        {
            P_iota[i] = static_cast<mreal>(i);
            P_rp[i] = i;
        }

//    mint counter = 0;
#pragma omp parallel for
        for (mint i = 0; i < leaf_cluster_count; ++i)
        {
            mint C = leaf_clusters[i];
            C_rp[C + 1] = C_end[C] - C_begin[C];
        }

        for (mint i = 0; i < cluster_count; ++i)
        {
            C_rp[i + 1] += C_rp[i];
        }

        P_to_C = MKLSparseMatrix(cluster_count, primitive_count, &C_rp[0], &P_rp[0], &P_one[0]); // Copy!

        C_to_P = MKLSparseMatrix(primitive_count, cluster_count, &P_rp[0], &P_leaf[0], &P_one[0]); // Copy!

        A_Vector<mint> rp(dim * primitive_count + 1, 0);
        A_Vector<mint> ci(dim * primitive_count, 0);
        A_Vector<mreal> vals(dim * primitive_count, 0);
        rp[dim * primitive_count] = dim * primitive_count;
#pragma omp parallel for
        for (mint i = 0; i < primitive_count; ++i)
        {
            mreal a = P_data[0][i];
            for (mint k = 0; k < dim; ++k)
            {
                mint to = dim * i + k;
                rp[to] = dim * i + k;
                ci[to] = dim * P_ext_pos[i] + k;
                vals[to] = a;
            }
        }

        //permutation matrix and weighting for hi  term
        Eigen::Map<EigenMatrixCSR> hi_perm = Eigen::Map<EigenMatrixCSR>(dim * primitive_count, dim * primitive_count, dim * primitive_count, &rp[0], &ci[0], &vals[0]); // Copy!

        EigenMatrixCSR A, AT; // some buffer to copy into

        // I really want CSR format, not CSC or anything home baked by Eigen. Thus:
        A = hi_perm * DiffOp;
        // Now just give me the data; MKL is better in multiplying anyways.
        hi_pre = MKLSparseMatrix(A.rows(), A.cols(), A.outerIndexPtr(), A.innerIndexPtr(), A.valuePtr()); // Copy!

        // TODO: Optimize away these copys.

        AT = A.transpose();
        hi_post = MKLSparseMatrix(AT.rows(), AT.cols(), AT.outerIndexPtr(), AT.innerIndexPtr(), AT.valuePtr());

        //permutation matrix and weighting for lo and frac terms
        Eigen::Map<EigenMatrixCSR> lo_perm = Eigen::Map<EigenMatrixCSR>(primitive_count, primitive_count, primitive_count, &P_rp[0], &P_ext_pos[0], &P_data[0][0]); // Copy!

        A = lo_perm * AvOp;
        lo_pre = MKLSparseMatrix(A.rows(), A.cols(), A.outerIndexPtr(), A.innerIndexPtr(), A.valuePtr());

        AT = A.transpose();
        lo_post = MKLSparseMatrix(AT.rows(), AT.cols(), AT.outerIndexPtr(), AT.innerIndexPtr(), AT.valuePtr());

        // toc("Create pre and post");

        // toc("Initializing ClusterTree2");

    }; //Constructor

    void ClusterTree2::SplitCluster(Cluster2 *const C, const mint free_thread_count)
    {

        mint begin = C->begin;
        mint end = C->end;
        mint cpcount = end - begin;

        //        print ( "split cluster = " + std::to_string(begin) + " -- " + std::to_string(end));

        mint splitdir = -1;
        mreal L, Lmax;
        Lmax = -1.;
        mreal dirmin, dirmax, min, max;
        std::pair<mreal *, mreal *> range;

        // compute splitdir, the longest direction of bounding box
        for (mint k = 0; k < dim; ++k)
        {
            range = std::minmax_element(&P_coords[k][begin], &P_coords[k][end]);
            min = *range.first;
            max = *range.second;
            L = max - min;
            if (L > Lmax)
            {
                Lmax = L;
                splitdir = k;
                dirmin = min;
                dirmax = max;
            }
        }

        if ((cpcount > split_threshold) && (Lmax > 0.))
        {
            mreal mid = 0.5 * (dirmin + dirmax);

            // swapping points left from mid to the left and determining splitindex
            mint splitindex = begin;
            for (mint i = begin; i < end; ++i)
            {
                if (P_coords[splitdir][i] <= mid)
                {
                    std::swap(P_ext_pos[i], P_ext_pos[splitindex]);

                    for (mint k = 0, last = dim; k < last; ++k)
                    {
                        std::swap(P_coords[k][i], P_coords[k][splitindex]);
                    }
                    ++splitindex;
                }
            }

            // create new nodes...
            C->left = std::make_shared<Cluster2>(begin, splitindex, C->depth + 1);
            C->right = std::make_shared<Cluster2>(splitindex, end, C->depth + 1);
// ... and split them in parallel
#pragma omp task final(free_thread_count < 1) default(none) shared(C)
            {
                SplitCluster(C->left.get(), free_thread_count / 2);
            }
#pragma omp task final(free_thread_count < 1) default(none) shared(C)
            {
                SplitCluster(C->right.get(), free_thread_count - free_thread_count / 2);
            }
#pragma omp taskwait

            // collecting statistic for the later serialization
            // counting ourselves as descendant, too!
            C->descendant_count = 1 + C->left->descendant_count + C->right->descendant_count;
            C->descendant_leaf_count = C->left->descendant_leaf_count + C->right->descendant_leaf_count;
        }
        else
        {
            // count cluster as leaf cluster
            // counting ourselves as descendant, too!
            C->descendant_count = 1;
            C->descendant_leaf_count = 1;
        }
    }; //SplitCluster

    void ClusterTree2::Serialize(Cluster2 *C, mint ID, mint leaf_before_count, mint free_thread_count)
    {
        // enumeration in depth-first order

        C_begin[ID] = C->begin;

        C_end[ID] = C->end;

        C_depth[ID] = C->depth;

        if ((C->left != nullptr) && (C->right != nullptr))
        {
            C_left[ID] = ID + 1;
            C_right[ID] = ID + 1 + C->left->descendant_count;
            //
#pragma omp task final(free_thread_count < 1) default(none) shared(C, ID, C_left, C_right) firstprivate(free_thread_count, leaf_before_count)
            {
                Serialize(C->left.get(), C_left[ID], leaf_before_count, free_thread_count / 2);
            }
#pragma omp task final(free_thread_count < 1) default(none) shared(C, ID, C_left, C_right) firstprivate(free_thread_count, leaf_before_count)
            {
                Serialize(C->right.get(), C_right[ID], leaf_before_count + C->left->descendant_leaf_count, free_thread_count - free_thread_count / 2);
            }
#pragma omp taskwait
        }
        else
        {
            // print( "leaf = " +  std::to_string(ID) + " at  position " + std::to_string(leaf_before_count) );
            leaf_clusters[leaf_before_count] = ID;
            leaf_cluster_lookup[ID] = leaf_before_count;
        }
    }; //Serialize

    void ClusterTree2::ComputePrimitiveData(const mreal *const restrict P_hull_coords_, const mreal *const restrict P_data_) // reordering and computing bounding boxes
    {
        A_Vector<mreal> v(primitive_count);
        P_data = A_Vector<A_Vector<mreal>>(data_dim, v);
        P_min = A_Vector<A_Vector<mreal>>(dim, v);
        P_max = A_Vector<A_Vector<mreal>>(dim, v);

        mint hull_size = hull_count * dim;

#pragma omp parallel for default(none) shared(P_data, P_ext_pos, P_min, P_min, P_data_, P_hull_coords_, hull_size)
        for (mint i = 0; i < primitive_count; ++i)
        {
            mreal min, max;
            mint j = P_ext_pos[i];
            for (mint k = 0; k < data_dim; ++k)
            {
                P_data[k][i] = P_data_[data_dim * j + k];
            }

            // computing bounding boxes of primitives; admittedly, it looks awful
            for (mint k = 0; k < dim; ++k)
            {
                min = max = P_hull_coords_[hull_size * j + dim * 0 + k];
                for (mint h = 1; h < hull_count; ++h)
                {
                    mreal x = P_hull_coords_[hull_size * j + dim * h + k];
                    min = std::min(min, x);
                    max = std::max(max, x);
                }
                P_min[k][i] = min;
                P_max[k][i] = max;
            }
        }
    } //ComputePrimitiveData

    void ClusterTree2::ComputeClusterData()
    {

        // tic("Allocation");
        A_Vector<mreal> v(cluster_count);
        C_data = A_Vector<A_Vector<mreal>>(data_dim, v);
        C_coords = A_Vector<A_Vector<mreal>>(dim, v);
        C_min = A_Vector<A_Vector<mreal>>(dim, v);
        C_max = A_Vector<A_Vector<mreal>>(dim, v);
        C_squared_radius = A_Vector<mreal>(cluster_count);
// toc("Allocation");

// using the already serialized cluster tree
#pragma omp parallel default(none) shared(thread_count)
        {
#pragma omp single
            {
                computeClusterData(0, thread_count);
            }
        }
    }; //ComputeClusterData

    void ClusterTree2::computeClusterData(const mint C, const mint free_thread_count) // helper function for ComputeClusterData
    {
        mint L = C_left[C];
        mint R = C_right[C];

        if (L >= 0 && R >= 0)
        {

            //C points to interior node.

#pragma omp task final(free_thread_count < 1) default(none) shared(L, free_thread_count) //firstprivate(free_thread_count)
            {
                computeClusterData(L, free_thread_count / 2);
            }
#pragma omp task final(free_thread_count < 1) default(none) shared(R, free_thread_count) // firstprivate(free_thread_count)
            {
                computeClusterData(R, free_thread_count - free_thread_count / 2);
            }
#pragma omp taskwait

            //weight
            mreal L_weight = C_data[0][L];
            mreal R_weight = C_data[0][R];
            mreal C_mass = L_weight + R_weight;
            C_data[0][C] = C_mass;

            mreal C_invmass = 1. / C_mass;
            L_weight = L_weight * C_invmass;
            R_weight = R_weight * C_invmass;

            for (mint k = 1, last = data_dim; k < last; ++k)
            {
                C_data[k][C] = L_weight * C_data[k][L] + R_weight * C_data[k][R];
            }

            //clustering coordinates and bounding boxes
            for (mint k = 0, last = dim; k < last; ++k)
            {
                C_coords[k][C] = L_weight * C_coords[k][L] + R_weight * C_coords[k][R];
                C_min[k][C] = std::min(C_min[k][L], C_min[k][R]);
                C_max[k][C] = std::max(C_max[k][L], C_max[k][R]);
            }
        }
        else
        {
            //C points to leaf node.
            //compute from primitives

            mint begin = C_begin[C];
            mint end = C_end[C];
            // Mass
            mreal C_mass = 0.;
            for (mint i = begin, last = end; i < last; ++i)
            {
                C_mass += P_data[0][i];
            }
            C_data[0][C] = C_mass;
            mreal C_invmass = 1. / C_mass;

            // weighting the coordinates
            for (mint i = begin, last = end; i < last; ++i)
            {
                mreal P_weight = P_data[0][i] * C_invmass;
                for (mint k = 1, last = data_dim; k < last; ++k)
                {
                    C_data[k][C] += P_weight * P_data[k][i];
                }
                for (mint k = 0, last = dim; k < last; ++k)
                {
                    C_coords[k][C] += P_weight * P_coords[k][i];
                }
            }

            // bounding boxes
            for (mint k = 0; k < dim; ++k)
            {
                C_min[k][C] = *std::min_element(&P_min[k][begin], &P_min[k][end]);
                C_max[k][C] = *std::max_element(&P_max[k][begin], &P_max[k][end]);
            }
        }

        // finally, we compute the square radius of the bounding box, measured from the clusters barycenter C_coords
        mreal r2 = 0.;
        for (mint k = 0, last = dim; k < last; ++k)
        {
            mreal mid = C_coords[k][C];
            mreal delta_max = abs(C_max[k][C] - mid);
            mreal delta_min = abs(mid - C_min[k][C]);
            r2 += (delta_min <= delta_max) ? delta_max * delta_max : delta_min * delta_min;
        }
        C_squared_radius[C] = r2;
    }; //computeClusterData

    void ClusterTree2::PrepareBuffers(const mint cols)
    {
        if (cols > max_buffer_dim)
        {
            // print("Reallocating buffers to max_buffer_dim = " + std::to_string(cols) + "." );
            max_buffer_dim = cols;

            P_in = A_Vector<mreal>(primitive_count * max_buffer_dim, 0.);
            P_out = A_Vector<mreal>(primitive_count * max_buffer_dim, 0.);

            C_in = A_Vector<mreal>(cluster_count * max_buffer_dim, 0.);
            C_out = A_Vector<mreal>(cluster_count * max_buffer_dim, 0.);
        }

        buffer_dim = cols;

        //    print("ClusterTree2::PrepareBuffers set buffer_dim = " + std::to_string(buffer_dim) + "." );

    }; // PrepareBuffers

    void ClusterTree2::CleanseBuffers(){
#pragma omp parallel
        {
#pragma omp for
            for (mint i = 0; i < primitive_count; ++i){
                P_in[i] = 0.;
    P_out[i] = 0.;
} // namespace rsurfaces
#pragma omp for
for (mint i = 0; i < cluster_count; ++i)
{
    C_in[i] = 0.;
    C_out[i] = 0.;
}
}
}
; // CleanseBuffers

void ClusterTree2::PercolateUp(const mint C, const mint free_thread_count)
{
    // C = cluster index

    //    vprint("C", C );

    mint L = C_left[C];
    mint R = C_right[C];

    //    vprint("L", L );
    //    vprint("R", R );

    if ((L >= 0) && (R >= 0))
    {
// If not a leaf, compute the values of the children first.
#pragma omp task final(free_thread_count < 1) default(none) shared(L, free_thread_count)
        PercolateUp(L, free_thread_count / 2);
#pragma omp task final(free_thread_count < 1) default(none) shared(R, free_thread_count)
        PercolateUp(R, free_thread_count - free_thread_count / 2);
#pragma omp taskwait

        // Aftwards, compute the sum of the two children.

        cblas_dcopy(buffer_dim, &C_in[buffer_dim * L], 1, &C_in[buffer_dim * C], 1);
        cblas_daxpy(buffer_dim, 1., &C_in[buffer_dim * R], 1, &C_in[buffer_dim * C], 1);

        //        for( mint k = 0; k < buffer_dim; ++k )
        //        {
        //            // Overwrite, not add-into. Thus cleansing is not needed.
        //            C_in[ buffer_dim * C + k ] = C_in[ buffer_dim * L + k ] + C_in[ buffer_dim * R + k ];
        //        }
    }
    //    else
    //    {
    //        // f a leaf, just do nothing. P_to_C computed the value for us.
    //        print("Reached leaf");
    //        vprint("C",C);
    //        vprint(" buffer_dim * C", buffer_dim * C);
    //        vprint(" C_in[ buffer_dim * C ]",  C_in[ buffer_dim * C ] );
    //        If a leaf, just do nothing. P_to_C computed the value for us.
    //    }

}; // PercolateUp

void ClusterTree2::PercolateDown(const mint C, const mint free_thread_count)
{
    // C = cluster index

    //    vprint("C", C );
    //    vprint("buffer_dim", buffer_dim );
    //    vprint("C_out[ buffer_dim * C]", C_out[ buffer_dim * C] );

    mint L = C_left[C];
    mint R = C_right[C];

    //    vprint("L", L );
    //    vprint("R", R );

    if ((L >= 0) && (R >= 0))
    {

        //        cblas_daxpy( buffer_dim, 1., &C_out[ buffer_dim * C ], 1 , &C_out[ buffer_dim * L ], 1 );
        //        cblas_daxpy( buffer_dim, 1., &C_out[ buffer_dim * C ], 1 , &C_out[ buffer_dim * R ], 1 );
        //

        //        vprint("C_out[ buffer_dim * L]", C_out[ buffer_dim * L] );
        //        vprint("C_out[ buffer_dim * R]", C_out[ buffer_dim * R] );

        for (mint k = 0; k < buffer_dim; ++k)
        {
            mreal buffer = C_out[buffer_dim * C + k];
            C_out[buffer_dim * L + k] += buffer;
            C_out[buffer_dim * R + k] += buffer;
        }

        //        vprint("C_out[ buffer_dim * L]", C_out[ buffer_dim * L] );
        //        vprint("C_out[ buffer_dim * R]", C_out[ buffer_dim * R] );

        //        #pragma omp task final(free_thread_count<1) default(none) shared( L, free_thread_count )
        PercolateDown(L, free_thread_count / 2);
        //        #pragma omp task final(free_thread_count<1) default(none) shared( R, free_thread_count )
        PercolateDown(R, free_thread_count - free_thread_count / 2);
        //        #pragma omp taskwait
    }
}; // PercolateDown

void ClusterTree2::Pre(Eigen::MatrixXd &input, BCTKernelType type)
{

    mint cols = input.cols();
    //    tic("Eigen map + copy");
    EigenMatrixRM input_wrapper = EigenMatrixRM(input);
    //    toc("Eigen map + copy");

    Pre(input_wrapper.data(), cols, type);
}

void ClusterTree2::Pre(mreal *input, const mint cols, BCTKernelType type)
{
    MKLSparseMatrix *pre;

    switch (type)
    {
    case BCTKernelType::FractionalOnly:
    {
        pre = &lo_pre;
        PrepareBuffers(cols);
        break;
    }
    case BCTKernelType::HighOrder:
    {
        pre = &hi_pre;
        PrepareBuffers(dim * cols); // Beware: The derivative operator increases the number of columns!
        break;
    }
    case BCTKernelType::LowOrder:
    {
        pre = &lo_pre;
        PrepareBuffers(cols);
        break;
    }
    default:
    {
        eprint("Unknown kernel. Doing no.");
        return;
    }
    }

    // Caution: Some magic is going on here high order term...
    //    tic("MKL pre");
    // Apply diff/averaging operate, reorder and multiply by weights.
    pre->Multiply(input, &P_in[0], cols);
    //    toc("MKL pre");

    //    tic("P_to_C");
    // Accumulate into leaf clusters.
    P_to_C.Multiply(&P_in[0], &C_in[0], buffer_dim); // Beware: The derivative operator increases the number of columns!
                                                     //    toc("P_to_C");

    //    tic("PercolateUp");
    PercolateUp(0, thread_count);
    //    toc("PercolateUp");
}; // Pre

void ClusterTree2::Post(Eigen::MatrixXd &output, BCTKernelType type, bool addToResult)
{
    mint cols = output.cols();

    EigenMatrixRM output_wrapper(output.rows(), output.cols());

    Post(output_wrapper.data(), cols, type, false);

    if (addToResult)
    {
        output += output_wrapper; // This also converts back to the requested storage type (row/column major).
    }
    else
    {
        output = output_wrapper; // This also converts back to the requested storage type (row/column major).
    }
}

void ClusterTree2::Post(mreal *output, const mint cols, BCTKernelType type, bool addToResult)
{
    MKLSparseMatrix *post;

    mint expected_dim = buffer_dim;

    switch (type)
    {
    case BCTKernelType::FractionalOnly:
    {
        post = &lo_post;
        break;
    }
    case BCTKernelType::HighOrder:
    {
        post = &hi_post;
        expected_dim /= dim; // Beware: The derivative operator increases the number of columns!
        break;
    }
    case BCTKernelType::LowOrder:
    {
        post = &lo_post;
        break;
    }
    default:
    {
        eprint("Unknown kernel. Doing no.");
        return;
    }
    }

    if (expected_dim < cols)
    {
        wprint("Expected number of columns  = " + std::to_string(expected_dim) + " is smaller than requested number of columns " + std::to_string(cols) + ". Result is very likely unexpected.");
    }

    if (expected_dim > cols)
    {
        wprint("Expected number of columns  = " + std::to_string(expected_dim) + " is greater than requested number of columns " + std::to_string(cols) + ". Truncating output. Result is very likely unexpected.");
    }

    //    tic("PercolateDown");
    PercolateDown(0, thread_count);
    //    toc("PercolateDown");

    //    tic("C_to_P");
    // Add data from leaf clusters into data on primitives
    C_to_P.Multiply(&C_out[0], &P_out[0], buffer_dim, true); // Beware: The derivative operator increases the number of columns!
                                                             //    toc("C_to_P");

    // The last two steps could be fused if we only the output were row major.
    //    tic("MKL post");
    // Multiply by weights, restore external ordering, and apply transpose of diff/averaging operator.
    post->Multiply(&P_out[0], output, cols, false);
    //    toc("MKL post");

}; // Post
} // namespace rsurfaces