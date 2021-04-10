#include "main2.h"


using namespace geometrycentral;
using namespace geometrycentral::surface;



int main(int argc, char **argv)
{
    using namespace rsurfaces;

    std::cout << "Using Eigen version " << EIGEN_WORLD_VERSION << "." << EIGEN_MAJOR_VERSION << "." << EIGEN_MINOR_VERSION << std::endl;
    
    int threads;
    #pragma omp parallel
    {
        threads = omp_get_num_threads()/2;
    }
    ClearProfile("./TestObstacle0_" + std::to_string(threads)+ ".tsv");
    
    std::cout << std::setprecision(8);
    std::cout << "\n  =====                   =====  " << std::endl;
    std::cout << "=======   TestObstacle0   =======" << std::endl;
    std::cout << "  =====                   =====  " << std::endl;
    std::cout << "\n"
              << std::endl;

    double alpha = 6.;
    double beta = 12.;
    double weight = 0.5;
    double theta = 0.5;

    MeshUPtr u_mesh;
    GeomUPtr u_geom;

    // Load mesh1
    std::string obj1 = "../scenes/Bunny/bunny.obj";
    std::cout << "Loading " << obj1 << " as variable." << std::endl;
    std::tie(u_mesh, u_geom) = readMesh(obj1);
    MeshPtr mesh1 = std::move(u_mesh);
    GeomPtr geom1 = std::move(u_geom);
    geom1->requireVertexDualAreas();
    geom1->requireVertexNormals();
    
    // Load mesh2
    std::string obj2 = "../scenes/LungGrowing/sphere.obj";
    std::cout << "Loading " << obj2 << " as obstacle." << std::endl;
    std::tie(u_mesh, u_geom) = readMesh(obj2);
    MeshPtr mesh2 = std::move(u_mesh);
    GeomPtr geom2 = std::move(u_geom);
    geom2->requireVertexDualAreas();
    geom2->requireVertexNormals();
    
    
    mint primitive_count1 = mesh1->nVertices();
    mint primitive_count2 = mesh2->nVertices();
    
    tic("Create bvh1");
    OptimizedClusterTree *bvh1 = CreateOptimizedBVH(mesh1, geom1);
    OptimizedClusterTree *bvh1_nl = CreateOptimizedBVH_Normals(mesh1, geom1);
    OptimizedClusterTree *bvh1_pr = CreateOptimizedBVH_Projectors(mesh1, geom1);
    toc("Create bvh1");
    tic("Create bvh2");
    OptimizedClusterTree *bvh2 = CreateOptimizedBVH(mesh2, geom2);
    OptimizedClusterTree *bvh2_nl = CreateOptimizedBVH_Normals(mesh2, geom2);
    OptimizedClusterTree *bvh2_pr = CreateOptimizedBVH_Projectors(mesh2, geom2);
    toc("Create bvh2");

    tic("Create bct11");
    auto bct11 = std::make_shared<OptimizedBlockClusterTree>(bvh1, bvh1, alpha, beta, theta);
    auto bct11_nl = std::make_shared<OptimizedBlockClusterTree>(bvh1_nl, bvh1_nl, alpha, beta, theta);
    auto bct11_pr = std::make_shared<OptimizedBlockClusterTree>(bvh1_pr, bvh1_pr, alpha, beta, theta);
    toc("Create bct11");
    tic("Create bct12");
    auto bct12 = std::make_shared<OptimizedBlockClusterTree>(bvh1, bvh2, alpha, beta, theta);
    auto bct12_nl = std::make_shared<OptimizedBlockClusterTree>(bvh1_nl, bvh2_nl, alpha, beta, theta);
    auto bct12_pr = std::make_shared<OptimizedBlockClusterTree>(bvh1_pr, bvh2_pr, alpha, beta, theta);
    toc("Create bct12");

    // The transpose of bct12 and thus not needed.
    //auto bct21 = std::make_shared<OptimizedBlockClusterTree>(bvh2, bvh1, alpha, beta, theta);
    tic("Create bct22");
    auto bct22 = std::make_shared<OptimizedBlockClusterTree>(bvh2, bvh2, alpha, beta, theta);
    auto bct22_nl = std::make_shared<OptimizedBlockClusterTree>(bvh2_nl, bvh2_nl, alpha, beta, theta);
    auto bct22_pr = std::make_shared<OptimizedBlockClusterTree>(bvh2_pr, bvh2_pr, alpha, beta, theta);
    toc("Create bct22");

    bct11->PrintStats();
    bct12->PrintStats();
    bct22->PrintStats();
    
    // The joint bct of the union of mesh1 and mesh2 can be written in block matrix for as
    //  bct = {
    //            { bct11, bct12 },
    //            { bct21, bct22 }
    //        },
    // where bct11 and bct22 are the instances of OptimizedBlockClusterTree of mesh1 and mesh2, respectively, bct12 is cross interaction OptimizedBlockClusterTree of mesh1 and mesh2, and bct21 is the transpose of bct12.
    // However, the according matrix (on the space of dofs on the primitives) would be
    //  A   = {
    //            { A11 + diag( A12 * one2 ) , A12                      },
    //            { A21                      , A22 + diag( A21 * one1 ) }
    //        },
    // where one1 and one2 are all-1-vectors on the primitives of mesh1 and mesh2, respectively.
    // OptimizedBlockClusterTree::AddObstacleCorrection is supposed to compute diag( A12 * one2 ) and to add it to the diagonal of A11.
    // Afterwards, bct1->Multiply will also multiply with the metric contribution of the obstacle.
    tic("Modifying bct11 to include the terms with respect to the obstacle.");
    bct11->AddObstacleCorrection(bct12.get());
    bct11_nl->AddObstacleCorrection(bct12_nl.get());
    bct11_pr->AddObstacleCorrection(bct12_pr.get());
    toc("Modifying bct11 to include the terms with respect to the obstacle.");

    mint energy_count = 7;
    
    // the self-interaction energy of mesh1
    auto tpe_fm_11 = std::make_shared<TPEnergyMultipole0>(mesh1, geom1, bct11.get(), alpha, beta, weight);
    auto tpe_fm_nl_11 = std::make_shared<TPEnergyMultipole_Normals0>(mesh1, geom1, bct11_nl.get(), alpha, beta, weight);
    auto tpe_fm_pr_11 = std::make_shared<TPEnergyMultipole_Projectors0>(mesh1, geom1, bct11_pr.get(), alpha, beta, weight);
    auto tpe_bh_11 = std::make_shared<TPEnergyBarnesHut0>(mesh1, geom1, alpha, beta, theta, weight);
    auto tpe_bh_pr_11 = std::make_shared<TPEnergyBarnesHut_Projectors0>(mesh1, geom1, alpha, beta, theta, weight);
    auto tpe_ex_11 = std::make_shared<TPEnergyAllPairs>(mesh1, geom1, alpha, beta, weight);
    auto tpe_ex_pr_11 = std::make_shared<TPEnergyAllPairs_Projectors>(mesh1, geom1, alpha, beta, weight);

    // the interaction energy between mesh1 and mesh2
    auto tpe_fm_12 = std::make_shared<TPObstacleMultipole0>(mesh1, geom1, bct12.get(), alpha, beta, weight);
    auto tpe_fm_nl_12 = std::make_shared<TPObstacleMultipole_Normals0>(mesh1, geom1, bct12_nl.get(), alpha, beta, weight);
    auto tpe_fm_pr_12 = std::make_shared<TPObstacleMultipole_Projectors0>(mesh1, geom1, bct12_pr.get(), alpha, beta, weight);
    auto tpe_bh_12 = std::make_shared<TPObstacleBarnesHut0>(mesh1, geom1, tpe_bh_11.get(), mesh2, geom2, alpha, beta, theta, weight);
    auto tpe_bh_pr_12 = std::make_shared<TPObstacleBarnesHut_Projectors0>(mesh1, geom1, tpe_bh_pr_11.get(), mesh2, geom2, alpha, beta, theta, weight);
    auto tpe_ex_12 = std::make_shared<TPObstacleAllPairs>(mesh1, geom1, tpe_bh_11.get(), mesh2, geom2, alpha, beta, weight);
    auto tpe_ex_pr_12 = std::make_shared<TPObstacleAllPairs_Projectors>(mesh1, geom1, tpe_bh_pr_11.get(), mesh2, geom2, alpha, beta, weight);

    // the self-interaction energy of mesh2; since mesh2 is the obstacle here, this is not needed in practice; I used this here only for test purposes and in order to see how much "work" is saved by this approach.
    auto tpe_fm_22 = std::make_shared<TPEnergyMultipole0>(mesh2, geom2, bct22.get(), alpha, beta, weight);
    auto tpe_fm_nl_22 = std::make_shared<TPEnergyMultipole_Normals0>(mesh2, geom2, bct22_nl.get(), alpha, beta, weight);
    auto tpe_fm_pr_22 = std::make_shared<TPEnergyMultipole_Projectors0>(mesh2, geom2, bct22_pr.get(), alpha, beta, weight);
    auto tpe_bh_22 = std::make_shared<TPEnergyBarnesHut0>(mesh2, geom2, alpha, beta, theta, weight);
    auto tpe_bh_pr_22 = std::make_shared<TPEnergyBarnesHut_Projectors0>(mesh2, geom2, alpha, beta, theta, weight);
    auto tpe_ex_22 = std::make_shared<TPEnergyAllPairs>(mesh2, geom2, alpha, beta, weight);
    auto tpe_ex_pr_22 = std::make_shared<TPEnergyAllPairs_Projectors>(mesh2, geom2, alpha, beta, weight);

    // the energies tpe_**_11, tpe_**_12, tpe_**_22 are gauged such that their sum equals the tangent-point energy of the union of mesh1 and mesh2.

    double E_fm_11, E_fm_12, E_fm_22;
    double E_fm_nl_11, E_fm_nl_12, E_fm_nl_22;
    double E_fm_pr_11, E_fm_pr_12, E_fm_pr_22;
    double E_bh_11, E_bh_12, E_bh_22;
    double E_bh_pr_11, E_bh_pr_12, E_bh_pr_22;
    double E_ex_11, E_ex_12, E_ex_22;
    double E_ex_pr_11, E_ex_pr_12, E_ex_pr_22;

    Eigen::MatrixXd DE_fm_11(primitive_count1, 3);
    Eigen::MatrixXd DE_fm_12(primitive_count1, 3);
    Eigen::MatrixXd DE_fm_22(primitive_count2, 3);
    
    Eigen::MatrixXd DE_fm_nl_11(primitive_count1, 3);
    Eigen::MatrixXd DE_fm_nl_12(primitive_count1, 3);
    Eigen::MatrixXd DE_fm_nl_22(primitive_count2, 3);
    
    Eigen::MatrixXd DE_fm_pr_11(primitive_count1, 3);
    Eigen::MatrixXd DE_fm_pr_12(primitive_count1, 3);
    Eigen::MatrixXd DE_fm_pr_22(primitive_count2, 3);
    
    Eigen::MatrixXd DE_bh_11(primitive_count1, 3);
    Eigen::MatrixXd DE_bh_12(primitive_count1, 3);
    Eigen::MatrixXd DE_bh_22(primitive_count2, 3);
    
    Eigen::MatrixXd DE_bh_pr_11(primitive_count1, 3);
    Eigen::MatrixXd DE_bh_pr_12(primitive_count1, 3);
    Eigen::MatrixXd DE_bh_pr_22(primitive_count2, 3);
    
    Eigen::MatrixXd DE_ex_11(primitive_count1, 3);
    Eigen::MatrixXd DE_ex_12(primitive_count1, 3);
    Eigen::MatrixXd DE_ex_22(primitive_count2, 3);
    
    Eigen::MatrixXd DE_ex_pr_11(primitive_count1, 3);
    Eigen::MatrixXd DE_ex_pr_12(primitive_count1, 3);
    Eigen::MatrixXd DE_ex_pr_22(primitive_count2, 3);

    std::cout << "Using integer exponents." << std::endl;

    mint counter = 0;
    mint count = energy_count * 6;
    tic();
    E_ex_11 = tpe_ex_11->Value();
    mreal t_ex_11 = toc();
    std::cout << "done " << ++counter << " / " << count << std::endl;

    tic();
    E_ex_12 = tpe_ex_12->Value();
    mreal t_ex_12 = toc();
    std::cout << "done " << ++counter << " / " << count << std::endl;

    tic();
    E_ex_22 = tpe_ex_22->Value();
    mreal t_ex_22 = toc();
    std::cout << "done " << ++counter << " / " << count << std::endl;

    tic();
    DE_ex_11.setZero();
    tpe_ex_11->Differential(DE_ex_11);
    mreal Dt_ex_11 = toc();
    std::cout << "done " << ++counter << " / " << count << std::endl;

    tic();
    DE_ex_12.setZero();
    tpe_ex_12->Differential(DE_ex_12);
    mreal Dt_ex_12 = toc();
    std::cout << "done " << ++counter << " / " << count << std::endl;

    tic();
    DE_ex_22.setZero();
    tpe_ex_22->Differential(DE_ex_22);
    mreal Dt_ex_22 = toc();
    std::cout << "done " << ++counter << " / " << count << std::endl;

    //######################################
    
    tic();
    E_ex_pr_11 = tpe_ex_pr_11->Value();
    mreal t_ex_pr_11 = toc();
    std::cout << "done " << ++counter << " / " << count << std::endl;

    tic();
//        tpe_ex_pr_12->Update();
    E_ex_pr_12 = tpe_ex_pr_12->Value();
    mreal t_ex_pr_12 = toc();
    std::cout << "done " << ++counter << " / " << count << std::endl;

    tic();
    E_ex_pr_22 = tpe_ex_pr_22->Value();
    mreal t_ex_pr_22 = toc();
    std::cout << "done " << ++counter << " / " << count << std::endl;

    tic();
    DE_ex_pr_11.setZero();
    tpe_ex_pr_11->Differential(DE_ex_pr_11);
    mreal Dt_ex_pr_11 = toc();
    std::cout << "done " << ++counter << " / " << count << std::endl;

    tic();
    DE_ex_pr_12.setZero();
    tpe_ex_pr_12->Differential(DE_ex_pr_12);
    mreal Dt_ex_pr_12 = toc();
    std::cout << "done " << ++counter << " / " << count << std::endl;

    tic();
    DE_ex_pr_22.setZero();
    tpe_ex_pr_22->Differential(DE_ex_pr_22);
    mreal Dt_ex_pr_22 = toc();
    std::cout << "done " << ++counter << " / " << count << std::endl;

    //######################################
    
    
    tic();
    E_bh_11 = tpe_bh_11->Value();
    mreal t_bh_11 = toc();
    std::cout << "done " << ++counter << " / " << count << std::endl;

    tic();
//        tpe_bh_12->Update();
    E_bh_12 = tpe_bh_12->Value();
    mreal t_bh_12 = toc();
    std::cout << "done " << ++counter << " / " << count << std::endl;

    tic();
    E_bh_22 = tpe_bh_22->Value();
    mreal t_bh_22 = toc();
    std::cout << "done " << ++counter << " / " << count << std::endl;

    tic();
    DE_bh_11.setZero();
    tpe_bh_11->Differential(DE_bh_11);
    mreal Dt_bh_11 = toc();
    std::cout << "done " << ++counter << " / " << count << std::endl;

    tic();
    DE_bh_12.setZero();
    tpe_bh_12->Differential(DE_bh_12);
    mreal Dt_bh_12 = toc();
    std::cout << "done " << ++counter << " / " << count << std::endl;

    tic();
    DE_bh_22.setZero();
    tpe_bh_22->Differential(DE_bh_22);
    mreal Dt_bh_22 = toc();
    std::cout << "done " << ++counter << " / " << count << std::endl;
    
    //######################################
    
    
    tic();
    E_bh_pr_11 = tpe_bh_pr_11->Value();
    mreal t_bh_pr_11 = toc();
    std::cout << "done " << ++counter << " / " << count << std::endl;

    tic();
//        tpe_bh_12->Update();
    E_bh_pr_12 = tpe_bh_pr_12->Value();
    mreal t_bh_pr_12 = toc();
    std::cout << "done " << ++counter << " / " << count << std::endl;

    tic();
    E_bh_pr_22 = tpe_bh_pr_22->Value();
    mreal t_bh_pr_22 = toc();
    std::cout << "done " << ++counter << " / " << count << std::endl;

    tic();
    DE_bh_pr_11.setZero();
    tpe_bh_pr_11->Differential(DE_bh_pr_11);
    mreal Dt_bh_pr_11 = toc();
    std::cout << "done " << ++counter << " / " << count << std::endl;

    tic();
    DE_bh_pr_12.setZero();
    tpe_bh_pr_12->Differential(DE_bh_pr_12);
    mreal Dt_bh_pr_12 = toc();
    std::cout << "done " << ++counter << " / " << count << std::endl;

    tic();
    DE_bh_pr_22.setZero();
    tpe_bh_pr_22->Differential(DE_bh_pr_22);
    mreal Dt_bh_pr_22 = toc();
    std::cout << "done " << ++counter << " / " << count << std::endl;

    //######################################
    
    tic();
    E_fm_nl_11 = tpe_fm_nl_11->Value();
    mreal t_fm_nl_11 = toc();
    std::cout << "done " << ++counter << " / " << count << std::endl;

    tic();
    E_fm_nl_12 = tpe_fm_nl_12->Value();
    mreal t_fm_nl_12 = toc();
    std::cout << "done " << ++counter << " / " << count << std::endl;

    tic();
    E_fm_nl_22 = tpe_fm_nl_22->Value();
    mreal t_fm_nl_22 = toc();
    std::cout << "done " << ++counter << " / " << count << std::endl;

    tic();
    DE_fm_nl_11.setZero();
    tpe_fm_nl_11->Differential(DE_fm_nl_11);
    mreal Dt_fm_nl_11 = toc();
    std::cout << "done " << ++counter << " / " << count << std::endl;

    tic();
    DE_fm_nl_12.setZero();
    tpe_fm_nl_12->Differential(DE_fm_nl_12);
    mreal Dt_fm_nl_12 = toc();
    std::cout << "done " << ++counter << " / " << count << std::endl;

    tic();
    DE_fm_nl_22.setZero();
    tpe_fm_nl_22->Differential(DE_fm_nl_22);
    mreal Dt_fm_nl_22 = toc();
    std::cout << "done " << ++counter << " / " << count << std::endl;
    
    //######################################
    
    tic();
    E_fm_11 = tpe_fm_11->Value();
    mreal t_fm_11 = toc();
    std::cout << "done " << ++counter << " / " << count << std::endl;

    tic();
    E_fm_12 = tpe_fm_12->Value();
    mreal t_fm_12 = toc();
    std::cout << "done " << ++counter << " / " << count << std::endl;

    tic();
    E_fm_22 = tpe_fm_22->Value();
    mreal t_fm_22 = toc();
    std::cout << "done " << ++counter << " / " << count << std::endl;

    tic();
    DE_fm_11.setZero();
    tpe_fm_11->Differential(DE_fm_11);
    mreal Dt_fm_11 = toc();
    std::cout << "done " << ++counter << " / " << count << std::endl;

    tic();
    DE_fm_12.setZero();
    tpe_fm_12->Differential(DE_fm_12);
    mreal Dt_fm_12 = toc();
    std::cout << "done " << ++counter << " / " << count << std::endl;
    
    tic();
    DE_fm_22.setZero();
    tpe_fm_22->Differential(DE_fm_22);
    mreal Dt_fm_22 = toc();
    std::cout << "done " << ++counter << " / " << count << std::endl;
    
    
    //######################################
    tic();
    E_fm_pr_11 = tpe_fm_pr_11->Value();
    mreal t_fm_pr_11 = toc();
    std::cout << "done " << ++counter << " / " << count << std::endl;

    tic();
    E_fm_pr_12 = tpe_fm_pr_12->Value();
    mreal t_fm_pr_12 = toc();
    std::cout << "done " << ++counter << " / " << count << std::endl;

    tic();
    E_fm_pr_22 = tpe_fm_pr_22->Value();
    mreal t_fm_pr_22 = toc();
    std::cout << "done " << ++counter << " / " << count << std::endl;

    tic();
    DE_fm_pr_11.setZero();
    tpe_fm_pr_11->Differential(DE_fm_pr_11);
    mreal Dt_fm_pr_11 = toc();
    std::cout << "done " << ++counter << " / " << count << std::endl;

    tic();
    DE_fm_pr_12.setZero();
    tpe_fm_pr_12->Differential(DE_fm_pr_12);
    mreal Dt_fm_pr_12 = toc();
    std::cout << "done " << ++counter << " / " << count << std::endl;

    tic();
    DE_fm_pr_22.setZero();
    tpe_fm_pr_22->Differential(DE_fm_pr_22);
    mreal Dt_fm_pr_22 = toc();
    std::cout << "done " << ++counter << " / " << count << std::endl;

    //######################################

    int w1 = 21;
    int w  = 13;
//
//        std::string line = "--------------------------------------------------------------------------------------------------------------------------------------------------------------------";
    std::string line = std::string( 3 * energy_count + w1 + w * energy_count, '-');
    std::cout   << std::left;
    std::cout   << std::setw(w1) << ""
                << " | " << std::setw(w) << "exact"
                << " | " << std::setw(w) << "pr"
                << " | " << std::setw(w) << "BH"
                << " | " << std::setw(w) << "BH_pr"
                << " | " << std::setw(w) << "FMM"
                << " | " << std::setw(w) << "FMM_pr"
                << " | " << std::setw(w) << "FMM_nl"
                << std::endl;

    std::cout   << line << std::endl;

    std::cout   << std::setw(w1) << "  E_11 "
                << " | " << std::setw(w) << E_ex_11
                << " | " << std::setw(w) << E_ex_pr_11
                << " | " << std::setw(w) << E_bh_11
                << " | " << std::setw(w) << E_bh_pr_11
                << " | " << std::setw(w) << E_fm_11
                << " | " << std::setw(w) << E_fm_pr_11
                << " | " << std::setw(w) << E_fm_nl_11
                << std::endl;

    std::cout   << std::setw(w1) << "  E_12 "
                << " | " << std::setw(w) << E_ex_12
                << " | " << std::setw(w) << E_ex_pr_12
                << " | " << std::setw(w) << E_bh_12
                << " | " << std::setw(w) << E_bh_pr_12
                << " | " << std::setw(w) << E_fm_12
                << " | " << std::setw(w) << E_fm_pr_12
                << " | " << std::setw(w) << E_fm_nl_12
                << std::endl;

    std::cout   << std::setw(w1) << "  E_22 "
                << " | " << std::setw(w) << E_ex_22
                << " | " << std::setw(w) << E_ex_pr_22
                << " | " << std::setw(w) << E_bh_22
                << " | " << std::setw(w) << E_bh_pr_22
                << " | " << std::setw(w) << E_fm_22
                << " | " << std::setw(w) << E_fm_pr_22
                << " | " << std::setw(w) << E_fm_nl_22
                << std::endl;
    
    
    std::cout   << "\n";
    std::cout   << std::setw(w1) << ""
                << " | " << std::setw(w) << "exact"
                << " | " << std::setw(w) << "pr"
                << " | " << std::setw(w) << "BH"
                << " | " << std::setw(w) << "BH_pr"
                << " | " << std::setw(w) << "FMM"
                << " | " << std::setw(w) << "FMM_pr"
                << " | " << std::setw(w) << "FMM_nl"
                << std::endl;

    std::cout   << line << std::endl;

    std::cout   << std::setw(w1) << "  E_11 error (%) "
                << " | " << std::setw(w) << fabs(E_ex_11 / E_ex_11 - 1) * 100
                << " | " << std::setw(w) << fabs(E_ex_pr_11 / E_ex_11 - 1) * 100
                << " | " << std::setw(w) << fabs(E_bh_11 / E_ex_11 - 1) * 100
                << " | " << std::setw(w) << fabs(E_bh_pr_11 / E_ex_11 - 1) * 100
                << " | " << std::setw(w) << fabs(E_fm_11 / E_ex_11 - 1) * 100
                << " | " << std::setw(w) << fabs(E_fm_pr_11 / E_ex_11 - 1) * 100
                << " | " << std::setw(w) << fabs(E_fm_nl_11 / E_ex_11 - 1) * 100
                << std::endl;

    std::cout   << std::setw(w1) << "  E_12 error (%) "
                << " | " << std::setw(w) << fabs(E_ex_12 / E_ex_12 - 1) * 100
                << " | " << std::setw(w) << fabs(E_ex_pr_12 / E_ex_12 - 1) * 100
                << " | " << std::setw(w) << fabs(E_bh_12 / E_ex_12 - 1) * 100
                << " | " << std::setw(w) << fabs(E_bh_pr_12 / E_ex_12 - 1) * 100
                << " | " << std::setw(w) << fabs(E_fm_12 / E_ex_12 - 1) * 100
                << " | " << std::setw(w) << fabs(E_fm_pr_12 / E_ex_12 - 1) * 100
                << " | " << std::setw(w) << fabs(E_fm_nl_12 / E_ex_12 - 1) * 100
                << std::endl;

    std::cout   << std::setw(w1) << "  E_22 error (%) "
                << " | " << std::setw(w) << fabs(E_ex_22 / E_ex_22 - 1) * 100
                << " | " << std::setw(w) << fabs(E_ex_pr_22 / E_ex_22 - 1) * 100
                << " | " << std::setw(w) << fabs(E_bh_22 / E_ex_22 - 1) * 100
                << " | " << std::setw(w) << fabs(E_bh_pr_22 / E_ex_22 - 1) * 100
                << " | " << std::setw(w) << fabs(E_fm_22 / E_ex_22 - 1) * 100
                << " | " << std::setw(w) << fabs(E_fm_pr_22 / E_ex_22 - 1) * 100
                << " | " << std::setw(w) << fabs(E_fm_nl_22 / E_ex_22 - 1) * 100
                << std::endl;

    std::cout   << std::setw(w1) << " DE_11 error (%) "
                << " | " << std::setw(w) << (DE_ex_11 - DE_ex_11).norm() / DE_ex_11.norm() * 100
                << " | " << std::setw(w) << (DE_ex_pr_11 - DE_ex_11).norm() / DE_ex_11.norm() * 100
                << " | " << std::setw(w) << (DE_bh_11 - DE_ex_11).norm() / DE_ex_11.norm() * 100
                << " | " << std::setw(w) << (DE_bh_pr_11 - DE_ex_11).norm() / DE_ex_11.norm() * 100
                << " | " << std::setw(w) << (DE_fm_11 - DE_ex_11).norm() / DE_ex_11.norm() * 100
                << " | " << std::setw(w) << (DE_fm_pr_11 - DE_ex_11).norm() / DE_ex_11.norm() * 100
                << " | " << std::setw(w) << (DE_fm_nl_11 - DE_ex_11).norm() / DE_ex_11.norm() * 100
                << std::endl;

    std::cout   << std::setw(w1) << " DE_12 error (%) "
                << " | " << std::setw(w) << (DE_ex_12 - DE_ex_12).norm() / DE_ex_12.norm() * 100
                << " | " << std::setw(w) << (DE_ex_pr_12 - DE_ex_12).norm() / DE_ex_12.norm() * 100
                << " | " << std::setw(w) << (DE_bh_12 - DE_ex_12).norm() / DE_ex_12.norm() * 100
                << " | " << std::setw(w) << (DE_bh_pr_12 - DE_ex_12).norm() / DE_ex_12.norm() * 100
                << " | " << std::setw(w) << (DE_fm_12 - DE_ex_12).norm() / DE_ex_12.norm() * 100
                << " | " << std::setw(w) << (DE_fm_pr_12 - DE_ex_12).norm() / DE_ex_12.norm() * 100
                << " | " << std::setw(w) << (DE_fm_nl_12 - DE_ex_12).norm() / DE_ex_12.norm() * 100
                << std::endl;

    std::cout   << std::setw(w1) << " DE_22 error (%) "
                << " | " << std::setw(w) << (DE_ex_22 - DE_ex_22).norm() / DE_ex_22.norm() * 100
                << " | " << std::setw(w) << (DE_ex_pr_22 - DE_ex_22).norm() / DE_ex_22.norm() * 100
                << " | " << std::setw(w) << (DE_bh_22 - DE_ex_22).norm() / DE_ex_22.norm() * 100
                << " | " << std::setw(w) << (DE_bh_pr_22 - DE_ex_22).norm() / DE_ex_22.norm() * 100
                << " | " << std::setw(w) << (DE_fm_22 - DE_ex_22).norm() / DE_ex_22.norm() * 100
                << " | " << std::setw(w) << (DE_fm_pr_22 - DE_ex_22).norm() / DE_ex_22.norm() * 100
                << " | " << std::setw(w) << (DE_fm_nl_22 - DE_ex_22).norm() / DE_ex_22.norm() * 100
                << std::endl;
    

    std::cout   << "\n"
                << std::endl;
    std::cout   << std::setw(w1) << ""
                << " | " << std::setw(w) << "exact"
                << " | " << std::setw(w) << "pr"
                << " | " << std::setw(w) << "BH"
                << " | " << std::setw(w) << "BH_pr"
                << " | " << std::setw(w) << "FMM"
                << " | " << std::setw(w) << "FMM_pr"
                << " | " << std::setw(w) << "FMM_nl"
                << std::endl;

    std::cout   << line << std::endl;

    std::cout   << std::setw(w1) << "  E_11 time  (s) "
                << " | " << std::setw(w) << t_ex_11
                << " | " << std::setw(w) << t_ex_pr_11
                << " | " << std::setw(w) << t_bh_11
                << " | " << std::setw(w) << t_bh_pr_11
                << " | " << std::setw(w) << t_fm_11
                << " | " << std::setw(w) << t_fm_pr_11
                << " | " << std::setw(w) << t_fm_nl_11
                << std::endl;
    
    std::cout   << std::setw(w1) << "  E_12 time  (s) "
                << " | " << std::setw(w) << t_ex_12
                << " | " << std::setw(w) << t_ex_pr_12
                << " | " << std::setw(w) << t_bh_12
                << " | " << std::setw(w) << t_bh_pr_12
                << " | " << std::setw(w) << t_fm_12
                << " | " << std::setw(w) << t_fm_pr_12
                << " | " << std::setw(w) << t_fm_nl_12
                << std::endl;
    
    std::cout   << std::setw(w1) << "  E_22 time  (s) "
                << " | " << std::setw(w) << t_ex_22
                << " | " << std::setw(w) << t_ex_pr_22
                << " | " << std::setw(w) << t_bh_22
                << " | " << std::setw(w) << t_bh_pr_22
                << " | " << std::setw(w) << t_fm_22
                << " | " << std::setw(w) << t_fm_pr_22
                << " | " << std::setw(w) << t_fm_nl_22
                << std::endl;
    
    std::cout   << std::setw(w1) << " DE_11 time  (s) "
                << " | " << std::setw(w) << Dt_ex_11
                << " | " << std::setw(w) << Dt_ex_pr_11
                << " | " << std::setw(w) << Dt_bh_11
                << " | " << std::setw(w) << Dt_bh_pr_11
                << " | " << std::setw(w) << Dt_fm_11
                << " | " << std::setw(w) << Dt_fm_pr_11
                << " | " << std::setw(w) << Dt_fm_nl_11
                << std::endl;
    
    std::cout   << std::setw(w1) << " DE_12 time  (s) "
                << " | " << std::setw(w) << Dt_ex_12
                << " | " << std::setw(w) << Dt_ex_pr_12
                << " | " << std::setw(w) << Dt_bh_12
                << " | " << std::setw(w) << Dt_bh_pr_12
                << " | " << std::setw(w) << Dt_fm_12
                << " | " << std::setw(w) << Dt_fm_pr_12
                << " | " << std::setw(w) << Dt_fm_nl_12
                << std::endl;
    
    std::cout   << std::setw(w1) << " DE_22 time  (s) "
                << " | " << std::setw(w) << Dt_ex_22
                << " | " << std::setw(w) << Dt_ex_pr_22
                << " | " << std::setw(w) << Dt_bh_22
                << " | " << std::setw(w) << Dt_bh_pr_22
                << " | " << std::setw(w) << Dt_fm_22
                << " | " << std::setw(w) << Dt_fm_pr_22
                << " | " << std::setw(w) << Dt_fm_nl_22
                << std::endl;

    delete bvh1;
    
    return EXIT_SUCCESS;
}
