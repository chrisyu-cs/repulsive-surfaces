#include "main.h"
#include "main_picking.h"

#include "geometrycentral/surface/halfedge_mesh.h"
#include "geometrycentral/surface/meshio.h"
#include "geometrycentral/surface/vertex_position_geometry.h"

#include "../deps/polyscope/deps/args/args/args.hxx"
#include "imgui.h"
#include "surface_derivatives.h"

#include "energy/tpe_kernel.h"
#include "energy/all_energies.h"
#include "helpers.h"
#include <memory>

#include <Eigen/Sparse>
#include <omp.h>

#include "sobolev/all_constraints.h"
#include "sobolev/hs.h"
#include "sobolev/hs_iterative.h"
#include "sobolev/h1.h"
#include "spatial/convolution.h"
#include "spatial/convolution_kernel.h"
#include "surface_derivatives.h"
#include "obj_writer.h"
#include "dropdown_strings.h"
#include "energy/coulomb.h"
#include "energy/willmore_energy.h"

#include "bct_constructors.h"

#include "remeshing/remeshing.h"

using namespace geometrycentral;
using namespace geometrycentral::surface;

namespace rsurfaces
{

    MainApp *MainApp::instance = 0;

    MainApp::MainApp(MeshPtr mesh_, GeomPtr geom_, SurfaceFlow *flow_, polyscope::SurfaceMesh *psMesh_, std::string meshName_)
        : mesh(std::move(mesh_)), geom(std::move(geom_)), geomOrig(geom->copy()), remesher(mesh, geom, geomOrig)
    {
        flow = flow_;
        psMesh = psMesh_;
        meshName = meshName_;
        vertBVH = 0;
        vertexPotential = 0;
        ctrlMouseDown = false;
        hasPickedVertex = false;
        numSteps = 0;
        methodChoice = GradientMethod::HsProjectedIterative;
        timeSpentSoFar = 0;
        realTimeLimit = 0;
        logPerformance = false;
        referenceEnergy = 0;
        exitWhenDone = false;
        totalObstacleVolume = 0;
    }

    void MainApp::logPerformanceLine()
    {
        if (!referenceEnergy)
        {
            referenceEnergy = new TPEnergyAllPairs(kernel->mesh, kernel->geom, kernel->alpha, kernel->beta);
        }
        referenceEnergy->Update();

        geom->refreshQuantities();
        std::ofstream outfile;
        outfile.open(sceneData.performanceLogFile, std::ios_base::app);
        double currentEnergy = referenceEnergy->Value();
        std::cout << numSteps << ", " << timeSpentSoFar << ", " << currentEnergy << ", " << mesh->nFaces() << std::endl;
        outfile << numSteps << ", " << timeSpentSoFar << ", " << currentEnergy << ", " << mesh->nFaces() << std::endl;
        outfile.close();
    }

    void MainApp::TakeOptimizationStep(bool remeshAfter, bool showAreaRatios)
    {
        ptic("MainApp::TakeOptimizationStep");
        
        if (logPerformance && numSteps == 0)
        {
            logPerformanceLine();
        }

        long beforeStep = currentTimeMilliseconds();
        
        ptic("Switch");
        switch (methodChoice)
        {
        case GradientMethod::HsProjected:
            flow->StepProjectedGradient();
            break;
        case GradientMethod::HsProjectedIterative:
            flow->StepProjectedGradientIterative();
            break;
        case GradientMethod::HsExactProjected:
            flow->StepProjectedGradientExact();
            break;
        case GradientMethod::H1Projected:
            flow->StepH1ProjGrad();
            break;
        case GradientMethod::L2Unconstrained:
            flow->StepL2Unconstrained();
            break;
        case GradientMethod::L2Projected:
            flow->StepL2Projected();
            break;
        case GradientMethod::AQP:
        {
            double kappa = 100;
            flow->StepAQP(1 / kappa);
        }
        break;
        case GradientMethod::H1_LBFGS:
            flow->StepH1LBFGS();
            break;
        case GradientMethod::BQN_LBFGS:
            flow->StepBQN();
            break;
        case GradientMethod::H2Projected:
        case GradientMethod::Willmore:
            flow->StepH2Projected();
            break;
        default:
            throw std::runtime_error("Unknown gradient method type.");
        }
        ptoc("Switch");
        
        if (remeshAfter)
        {
            bool doCollapse = (numSteps % 1 == 0);
            std::cout << "Applying remeshing..." << std::endl;
            flow->verticesMutated = remesher.Remesh(5, doCollapse);
            if (flow->verticesMutated)
            {
                std::cout << "Vertices were mutated this step -- memory vectors are now invalid." << std::endl;
            }
            else
            {
                std::cout << "Vertices were not mutated this step." << std::endl;
            }
            ptic("mesh->compress()");
            mesh->compress();
            ptoc("mesh->compress()");
            ptic("MainApp::instance->reregisterMesh();");
            MainApp::instance->reregisterMesh();
            ptoc("MainApp::instance->reregisterMesh();");
        }
        else
        {
            flow->verticesMutated = false;
            MainApp::instance->updateMeshPositions();
        }
        long afterStep = currentTimeMilliseconds();
        long timeForStep = afterStep - beforeStep;
        timeSpentSoFar += timeForStep;
        numSteps++;
        std::cout << "  Mesh total volume = " << totalVolume(geom, mesh) << std::endl;
        std::cout << "  Mesh total area = " << totalArea(geom, mesh) << std::endl;

        if (logPerformance)
        {
            logPerformanceLine();
        }

        if (showAreaRatios)
        {
            VertexData<double> areaRatio(*mesh);
            for (Vertex v : mesh->vertices())
            {
                areaRatio[v] = geomOrig->vertexDualArea(v) / geom->vertexDualArea(v);
            }

            psMesh->addVertexScalarQuantity("Area ratios", areaRatio);
        }
        ptoc("MainApp::TakeOptimizationStep");
    }

    void MainApp::updateMeshPositions()
    {
        if (normalizeView)
        {
            double scale = 0;
            for (GCVertex v : mesh->vertices())
            {
                scale = fmax(scale, norm(geom->inputVertexPositions[v]));
            }
            std::vector<Vector3> scaled(mesh->nVertices());
            VertexIndices inds = mesh->getVertexIndices();
            for (GCVertex v : mesh->vertices())
            {
                scaled[inds[v]] = geom->inputVertexPositions[v] / scale;
            }
            psMesh->updateVertexPositions(scaled);
        }
        else
        {
            psMesh->updateVertexPositions(geom->inputVertexPositions);
        }
        polyscope::requestRedraw();
    }

    void PlotMatrix(Eigen::MatrixXd &mat, polyscope::SurfaceMesh *psMesh, std::string name)
    {
        std::vector<Vector3> vecs;
        for (int i = 0; i < mat.rows(); i++)
        {
            Vector3 row_i = GetRow(mat, i);
            vecs.push_back(row_i);
        }
        psMesh->addVertexVectorQuantity(name, vecs);
    }

    void PlotVector(Eigen::VectorXd &vec, int nVerts, polyscope::SurfaceMesh *psMesh, std::string name)
    {
        Eigen::MatrixXd M;
        M.setZero(nVerts, 3);
        MatrixUtils::ColumnIntoMatrix(vec, M);
        PlotMatrix(M, psMesh, name);
    }

    void MainApp::PlotGradients()
    {
        Eigen::MatrixXd l2Diff, hsGrad, hsGradExact;
        l2Diff.setZero(mesh->nVertices(), 3);
        hsGrad.setZero(mesh->nVertices(), 3);
        hsGradExact.setZero(mesh->nVertices(), 3);

        flow->UpdateEnergies();

        std::cout << "Assembling L2 differential..." << std::endl;
        long diffTimeStart = currentTimeMilliseconds();
        flow->AssembleGradients(l2Diff);
        long diffTimeEnd = currentTimeMilliseconds();
        std::cout << "Differential took " << (diffTimeEnd - diffTimeStart) << " ms" << std::endl;

        std::unique_ptr<Hs::HsMetric> hs = flow->GetHsMetric();

        std::cout << "Inverting \"sparse\" metric..." << std::endl;
        long sparseTimeStart = currentTimeMilliseconds();
        hs->InvertMetricMat(l2Diff, hsGrad);
        long sparseTimeEnd = currentTimeMilliseconds();
        std::cout << "Sparse metric took " << (sparseTimeEnd - sparseTimeStart) << " ms" << std::endl;

        std::cout << "Inverting dense metric..." << std::endl;
        long timeStart = currentTimeMilliseconds();
        std::vector<ConstraintPack> empty;
        // hs->ProjectGradientExact(l2Diff, hsGradExact, empty);
        hsGradExact = hsGrad;
        long timeEnd = currentTimeMilliseconds();
        std::cout << "Dense metric took " << (timeEnd - timeStart) << " ms" << std::endl;

        PlotMatrix(l2Diff, psMesh, "L2 differential");
        PlotMatrix(hsGrad, psMesh, "Hs sparse gradient");
        PlotMatrix(hsGradExact, psMesh, "Hs dense gradient");
    }

    bool MainApp::pickNearbyVertex(GCVertex &out)
    {
        using namespace polyscope;
        Vector2 screenPos = getMouseScreenPos();

        std::pair<Structure *, size_t> pickVal =
            pick::evaluatePickQuery(screenPos.x, screenPos.y);

        GCVertex pickedVert;
        GCFace pickedFace;
        GCEdge pickedEdge;
        GCHalfedge pickedHalfedge;

        glm::mat4 view = polyscope::view::getCameraViewMatrix();
        glm::mat4 proj = polyscope::view::getCameraPerspectiveMatrix();
        glm::mat4 viewProj = proj * view;

        polyscope::SurfaceMesh *asMesh = dynamic_cast<polyscope::SurfaceMesh *>(pickVal.first);

        if (tryGetPickedVertex(asMesh, pickVal.second, mesh, pickedVert))
        {
            out = pickedVert;
            return true;
        }
        else if (tryGetPickedFace(asMesh, pickVal.second, mesh, pickedFace))
        {
            out = nearestVertexToScreenPos(screenPos, geom, viewProj, pickedFace);
            return true;
        }
        else if (tryGetPickedEdge(asMesh, pickVal.second, mesh, pickedEdge))
        {
            out = nearestVertexToScreenPos(screenPos, geom, viewProj, pickedEdge);
            return true;
        }
        else if (tryGetPickedHalfedge(asMesh, pickVal.second, mesh, pickedHalfedge))
        {
            out = nearestVertexToScreenPos(screenPos, geom, viewProj, pickedHalfedge);
            return true;
        }
        else
        {
            std::cout << "No valid element was picked (index " << pickVal.second << ")" << std::endl;
            return false;
        }
    }

    class PVCompare
    {
    public:
        bool operator()(PriorityVertex v1, PriorityVertex v2)
        {
            return (v1.priority > v2.priority);
        }
    };

    double gaussian(double radius, double dist)
    {
        double radterm = dist / radius;
        double epow = exp(-0.5 * radterm * radterm);
        return epow;
    }

    void MainApp::GetFalloffWindow(GCVertex v, double radius, std::vector<PriorityVertex> &verts)
    {
        // Do a simple Dijkstra search on edges
        VertexData<bool> seen(*mesh, false);
        std::priority_queue<PriorityVertex, std::vector<PriorityVertex>, PVCompare> queue;
        queue.push(PriorityVertex{v, 0, geom->inputVertexPositions[v]});

        while (!queue.empty())
        {
            PriorityVertex next = queue.top();
            queue.pop();

            if (next.priority > radius)
            {
                break;
            }
            else if (seen[next.vertex])
            {
                continue;
            }
            else
            {
                // Mark the next vertex as seen
                seen[next.vertex] = true;
                // Compute the weight
                double weight = gaussian(radius / 3, next.priority);
                verts.push_back(PriorityVertex{next.vertex, weight, geom->inputVertexPositions[next.vertex]});

                // Enqueue all neighbors
                for (GCVertex neighbor : next.vertex.adjacentVertices())
                {
                    if (seen[neighbor])
                    {
                        continue;
                    }
                    // Add the next edge distance
                    Vector3 p1 = geom->inputVertexPositions[next.vertex];
                    Vector3 p2 = geom->inputVertexPositions[neighbor];
                    double neighborDist = next.priority + norm(p1 - p2);

                    queue.push(PriorityVertex{neighbor, neighborDist, geom->inputVertexPositions[neighbor]});
                }
            }
        }

        std::cout << "Got " << verts.size() << " vertices" << std::endl;
    }

    void MainApp::HandlePicking()
    {
        using namespace polyscope;

        auto io = ImGui::GetIO();
        glm::mat4 view = polyscope::view::getCameraViewMatrix();
        glm::mat4 proj = polyscope::view::getCameraPerspectiveMatrix();
        glm::mat4 viewProj = proj * view;

        if (io.KeyCtrl && io.MouseDown[0])
        {
            if (!ctrlMouseDown)
            {
                if (pickNearbyVertex(pickedVertex))
                {
                    hasPickedVertex = true;
                    GetFalloffWindow(pickedVertex, 0.5, dragVertices);

                    Vector3 screen = projectToScreenCoords3(geom->inputVertexPositions[pickedVertex], viewProj);
                    pickDepth = screen.z;

                    Vector3 unprojected = unprojectFromScreenCoords3(Vector2{screen.x, screen.y}, pickDepth, viewProj);
                    initialPickedPosition = geom->inputVertexPositions[pickedVertex];
                }
                ctrlMouseDown = true;
            }
            else
            {
                if (hasPickedVertex)
                {
                    Vector2 mousePos = getMouseScreenPos();
                    Vector3 unprojected = unprojectFromScreenCoords3(mousePos, pickDepth, viewProj);
                    Vector3 displacement = unprojected - initialPickedPosition;

                    for (PriorityVertex &v : dragVertices)
                    {
                        Vector3 newPos = v.position + v.priority * displacement;
                        geom->inputVertexPositions[v.vertex] = newPos;
                    }

                    flow->ResetAllConstraints();
                    flow->ResetAllPotentials();

                    if (vertexPotential)
                    {
                        for (PriorityVertex &v : dragVertices)
                        {
                            vertexPotential->ChangeVertexTarget(v.vertex, geom->inputVertexPositions[v.vertex]);
                        }
                    }

                    updateMeshPositions();
                }
            }
        }
        else
        {
            if (ctrlMouseDown)
            {
                ctrlMouseDown = false;
                hasPickedVertex = false;
                dragVertices.clear();
                // geom->inputVertexPositions[pickedVertex] = initialPickedPosition;
                updateMeshPositions();
            }
        }
    }

    void MainApp::CreateAndDestroyBVH()
    {
        OptimizedClusterTree *bvh = CreateOptimizedBVH(mesh, geom);
        std::cout << "Created BVH" << std::endl;
        delete bvh;
        std::cout << "Deleted BVH" << std::endl;
    }

    void MainApp::Scale2x()
    {
        for (GCVertex v : mesh->vertices())
        {
            geom->inputVertexPositions[v] = 2 * geom->inputVertexPositions[v];
        }
    }

    Jacobian numericalNormalDeriv(GeomPtr &geom, GCVertex vert, GCVertex wrt)
    {
        double h = 1e-4;

        Vector3 origNormal = vertexAreaNormal(geom, vert);
        Vector3 origPos = geom->inputVertexPositions[wrt];
        geom->inputVertexPositions[wrt] = origPos + Vector3{h, 0, 0};
        geom->refreshQuantities();
        Vector3 n_x = vertexAreaNormal(geom, vert);

        geom->inputVertexPositions[wrt] = origPos + Vector3{0, h, 0};
        geom->refreshQuantities();
        Vector3 n_y = vertexAreaNormal(geom, vert);

        geom->inputVertexPositions[wrt] = origPos + Vector3{0, 0, h};
        geom->refreshQuantities();
        Vector3 n_z = vertexAreaNormal(geom, vert);

        geom->inputVertexPositions[wrt] = origPos;
        geom->refreshQuantities();

        Vector3 deriv_y = (n_y - origNormal) / h;
        Vector3 deriv_z = (n_z - origNormal) / h;
        Vector3 deriv_x = (n_x - origNormal) / h;
        Jacobian J_num{deriv_x, deriv_y, deriv_z};
        return J_num;
    }

    void MainApp::TestNormalDeriv()
    {
        GCVertex vert;
        for (GCVertex v : mesh->vertices())
        {
            if (v.isBoundary())
            {
                vert = v;
                break;
            }
        }
        std::cout << "Testing vertex " << vert << std::endl;
        for (GCVertex neighbor : vert.adjacentVertices())
        {
            std::cout << "Derivative of normal of " << vert << " wrt " << neighbor << std::endl;
            Jacobian dWrtNeighbor = SurfaceDerivs::vertexNormalWrtVertex(geom, vert, neighbor);
            dWrtNeighbor.Print();
            std::cout << "Numerical:" << std::endl;
            numericalNormalDeriv(geom, vert, neighbor).Print();
        }
        std::cout << "Derivative of normal of " << vert << " wrt " << vert << std::endl;
        Jacobian dWrtSelf = SurfaceDerivs::vertexNormalWrtVertex(geom, vert, vert);
        dWrtSelf.Print();
        std::cout << "Numerical:" << std::endl;
        numericalNormalDeriv(geom, vert, vert).Print();
    }

    void MainApp::TestObstacle0()
    {
        int threads;
        #pragma omp parallel
        {
            threads = omp_get_num_threads();
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
        double theta = MainApp::instance->bh_theta;

        // mesh1 and geom1 represent the movable surface
        auto mesh1 = rsurfaces::MainApp::instance->mesh;
        auto geom1 = rsurfaces::MainApp::instance->geom;

        // Load obstacle
//        std::string filename = "../scenes/Bunny/bunny-10p.obj";
        std::string filename = "../scenes/Bunny/bunny.obj";
        MeshUPtr umesh;
        GeomUPtr ugeom;
        std::tie(umesh, ugeom) = readMesh(filename);
        ugeom->requireVertexDualAreas();
        ugeom->requireVertexNormals();
        std::string mesh_name = polyscope::guessNiceNameFromPath(filename);
        polyscope::SurfaceMesh *psMesh = polyscope::registerSurfaceMesh(mesh_name, ugeom->inputVertexPositions, umesh->getFaceVertexList(), polyscopePermutations(*umesh));
        // mesh2 and geom2 represent the pinned obstacle
        MeshPtr mesh2 = std::move(umesh);
        GeomPtr geom2 = std::move(ugeom);

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
        
//        std::ofstream file;
//        file.open("./Profile.tsv");
//        WriteProfile(file);
//        file.close();
        
        
    } // TestObstacle0

    void MainApp::TestBarnesHut0()
    {
        std::cout << std::setprecision(16);
        std::cout << "\n  =====                        =====  " << std::endl;
        std::cout << "=======   TPEnergyBarnesHut0   =======" << std::endl;
        std::cout << "  =====                        =====  " << std::endl;
        std::cout << "\n"
                  << std::endl;
        auto mesh = rsurfaces::MainApp::instance->mesh;
        auto geom = rsurfaces::MainApp::instance->geom;

        mreal alpha = 6.;
        mreal beta = 12.;
        mreal theta = 0.25;

        //        tic("Create BVH");
        //        OptimizedClusterTree* bvh = CreateOptimizedBVH( mesh, geom );
        //        tic("Create BVH");

        double E, Ex;
        Eigen::MatrixXd DE(mesh->nVertices(), 3);
        Eigen::MatrixXd DEx(mesh->nVertices(), 3);

        auto tpe = std::make_shared<TPEnergyBarnesHut0>(mesh, geom, alpha, beta, theta);
        auto tpex = std::make_shared<TPEnergyAllPairs>(mesh, geom, alpha, beta);

        tpe->use_int = false;
        std::cout << "Using double exponents." << std::endl;

        tic("Compute Value");
        tpe->Update();
        E = tpe->Value();
        toc("Compute Value");
        std::cout << "  E = " << E << std::endl;
        tic("Compute Differential");
        DE.setZero();
        tpe->Differential(DE);
        toc("Compute Differential");

        std::cout << "  DE = " << DE(0, 0) << " , " << DE(0, 1) << " , " << DE(0, 2) << std::endl;
        std::cout << "       " << DE(1, 0) << " , " << DE(1, 1) << " , " << DE(1, 2) << std::endl;
        std::cout << "       " << DE(2, 0) << " , " << DE(2, 1) << " , " << DE(2, 2) << std::endl;
        std::cout << "       " << DE(3, 0) << " , " << DE(3, 1) << " , " << DE(3, 2) << std::endl;
        std::cout << "       " << DE(4, 0) << " , " << DE(4, 1) << " , " << DE(4, 2) << std::endl;

        tpe->use_int = true;
        std::cout << "Using integer exponents." << std::endl;

        tic("Compute Value");
        tpe->Update();
        E = tpe->Value();
        toc("Compute Value");
        std::cout << "  E = " << E << std::endl;
        tic("Compute Differential");
        DE.setZero();
        tpe->Differential(DE);
        toc("Compute Differential");

        std::cout << "  DE = " << DE(0, 0) << " , " << DE(0, 1) << " , " << DE(0, 2) << std::endl;
        std::cout << "       " << DE(1, 0) << " , " << DE(1, 1) << " , " << DE(1, 2) << std::endl;
        std::cout << "       " << DE(2, 0) << " , " << DE(2, 1) << " , " << DE(2, 2) << std::endl;
        std::cout << "       " << DE(3, 0) << " , " << DE(3, 1) << " , " << DE(3, 2) << std::endl;
        std::cout << "       " << DE(4, 0) << " , " << DE(4, 1) << " , " << DE(4, 2) << std::endl;

        tic("Compute Value (all pairs)");
        tpex->Update();
        Ex = tpex->Value();
        toc("Compute Value (all pairs)");
        std::cout << "  Ex = " << Ex << std::endl;
        tic("Compute Differential (all pairs)");
        DEx.setZero();
        tpex->Differential(DEx);
        toc("Compute Differential (all pairs)");

        std::cout << "Energy value = " << E << std::endl;
        std::cout << "Diff. norm   = " << DE.norm() << std::endl;

        std::cout << "Exact energy value = " << Ex << std::endl;
        std::cout << "Exact diff. value  = " << DEx.norm() << std::endl;

        double energyError = fabs(E - Ex) / Ex * 100;
        double diffError = (DE - DEx).norm() / DEx.norm() * 100;

        std::cout << "Energy relative error = " << energyError << " percent" << std::endl;
        std::cout << "Diff. relative error  = " << diffError << " percent" << std::endl;

        SurfaceEnergy *oldBH = flow->BaseEnergy();
        Eigen::MatrixXd oldDiff(mesh->nVertices(), 3);

        oldDiff.setZero();
        oldBH->Update();
        double oldE = oldBH->Value();
        oldBH->Differential(oldDiff);

        std::cout << "Old BH energy value = " << oldE << std::endl;
        std::cout << "Old BH diff. value  = " << oldDiff.norm() << std::endl;

        double oldEnergyError = fabs(oldE - Ex) / Ex * 100;
        double oldDiffError = (oldDiff - DEx).norm() / DEx.norm() * 100;

        std::cout << "Energy relative error = " << oldEnergyError << " percent" << std::endl;
        std::cout << "Diff. relative error  = " << oldDiffError << " percent" << std::endl;

        std::cout << "\n --- Derivative test --- " << std::endl;

        mreal Et;
        Eigen::MatrixXd x(mesh->nVertices(), 3);
        Eigen::MatrixXd xnew(mesh->nVertices(), 3);

        VertexIndices inds = mesh->getVertexIndices();
        for (GCVertex v : mesh->vertices())
        {
            size_t i = inds[v];
            x(i, 0) = geom->inputVertexPositions[v].x;
            x(i, 1) = geom->inputVertexPositions[v].y;
            x(i, 2) = geom->inputVertexPositions[v].z;
        }

        Eigen::MatrixXd u = Eigen::MatrixXd::Random(mesh->nVertices(), 3);

        mreal t = 1.;
        for (int k = 0; k < 8; ++k)
        {
            t *= 0.1;
            xnew = x + t * u;
            VertexIndices inds = mesh->getVertexIndices();
            for (GCVertex v : mesh->vertices())
            {
                size_t i = inds[v];
                Vector3 corr{xnew(i, 0), xnew(i, 1), xnew(i, 2)};
                geom->inputVertexPositions[v] = corr;
            }

            geom->refreshQuantities();

            //            UpdateOptimizedBVH(mesh, geom, tpe->GetBVH());
            tpe->Update();
            Et = tpe->Value();
            //            std::cout << "Et - E             = " << Et - E << std::endl;
            //            std::cout << "        t * DE * u = " << t * (DE.transpose() * u).trace() << std::endl;
            std::cout << "Et - E -t * DE * u = " << Et - E - t * (DE.transpose() * u).trace() << std::endl;
        }

    } // TestBarnesHut0

    void MainApp::TestWillmore()
    {

        std::cout << "\n=====   WillmoreEnergy   =====" << std::endl;

        std::cout << std::setprecision(16);

        auto mesh = rsurfaces::MainApp::instance->mesh;
        auto geom = rsurfaces::MainApp::instance->geom;

        VertexIndices vInds = mesh->getVertexIndices();

        SurfaceEnergy *willmore = new WillmoreEnergy(mesh, geom);

        tic("Value");
        double E = willmore->Value();
        toc("Value");

        std::cout << "  E = " << E << std::endl;

        Eigen::MatrixXd DE(mesh->nVertices(), 3);

        tic("Differential");
        willmore->Differential(DE);
        toc("Differential");

        std::cout << "  DE = " << DE(0, 0) << " , " << DE(0, 1) << " , " << DE(0, 2) << std::endl;
        std::cout << "       " << DE(1, 0) << " , " << DE(1, 1) << " , " << DE(1, 2) << std::endl;
        std::cout << "       " << DE(2, 0) << " , " << DE(2, 1) << " , " << DE(2, 2) << std::endl;
        std::cout << "       " << DE(3, 0) << " , " << DE(3, 1) << " , " << DE(3, 2) << std::endl;
        std::cout << "       " << DE(4, 0) << " , " << DE(4, 1) << " , " << DE(4, 2) << std::endl;

        delete willmore;
    } // TestWillmore

    class VectorInit
    {
    public:
        static void Init(Vector3 &data, BVHNode6D *node)
        {
            data = Vector3{1, 2, 3};
        }
    };

    void MainApp::AddObstacle(std::string filename, double weight, bool recenter)
    {
        MeshUPtr obstacleMesh;
        GeomUPtr obstacleGeometry;
        // Load mesh
        std::tie(obstacleMesh, obstacleGeometry) = readMesh(filename);

        obstacleGeometry->requireVertexDualAreas();
        obstacleGeometry->requireVertexNormals();

        if (recenter)
        {
            Vector3 obstacleCenter = meshBarycenter(obstacleGeometry, obstacleMesh);
            std::cout << "Recentering obstacle " << filename << " (offset " << obstacleCenter << ")" << std::endl;
            for (GCVertex v : obstacleMesh->vertices())
            {
                obstacleGeometry->inputVertexPositions[v] = obstacleGeometry->inputVertexPositions[v] - obstacleCenter;
            }
        }

        std::string mesh_name = polyscope::guessNiceNameFromPath(filename);
        polyscope::SurfaceMesh *psMesh = polyscope::registerSurfaceMesh(mesh_name, obstacleGeometry->inputVertexPositions,
                                                                        obstacleMesh->getFaceVertexList(), polyscopePermutations(*obstacleMesh));

        MeshPtr sharedObsMesh = std::move(obstacleMesh);
        GeomPtr sharedObsGeom = std::move(obstacleGeometry);

        TPObstacleBarnesHut0 *obstacleEnergy = new TPObstacleBarnesHut0(mesh, geom, flow->BaseEnergy(), sharedObsMesh, sharedObsGeom,
                                                                        kernel->alpha, kernel->beta, bh_theta, weight);
        flow->AddObstacleEnergy(obstacleEnergy);
        std::cout << "Added " << filename << " as obstacle with weight " << weight << std::endl;

        totalObstacleVolume += totalVolume(sharedObsGeom, sharedObsMesh);
    }

    void MainApp::AddImplicitBarrier(scene::ImplicitBarrierData &barrierData)
    {
        ImplicitSurface *implSurface;
        // Create the requested implicit surface
        switch (barrierData.type)
        {
        case scene::ImplicitType::Plane:
        {
            Vector3 point{barrierData.parameters[0], barrierData.parameters[1], barrierData.parameters[2]};
            Vector3 normal{barrierData.parameters[3], barrierData.parameters[4], barrierData.parameters[5]};
            std::cout << "Constructing implicit plane at point " << point << " with normal " << normal << std::endl;
            implSurface = new FlatPlane(point, normal);
        }
        break;
        case scene::ImplicitType::Torus:
        {
            double major = barrierData.parameters[0];
            double minor = barrierData.parameters[1];
            Vector3 center{barrierData.parameters[2], barrierData.parameters[3], barrierData.parameters[4]};
            std::cout << "Constructing implicit torus with major radius " << major << ", minor radius " << minor << ", center " << center << std::endl;
            implSurface = new ImplicitTorus(major, minor, center);
        }
        break;
        case scene::ImplicitType::Sphere:
        {
            double radius = barrierData.parameters[0];
            Vector3 center{barrierData.parameters[1], barrierData.parameters[2], barrierData.parameters[3]};
            std::cout << "Constructing implicit sphere with radius " << radius << ", center " << center << std::endl;
            implSurface = new ImplicitSphere(radius, center);
        }
        break;
        default:
        {
            throw std::runtime_error("Unimplemented implicit surface type.");
        }
        break;
        }

        // Mesh the 0 isosurface so we can see the implicit surface
        MainApp::instance->MeshImplicitSurface(implSurface);

        // Use the implicit surface to setup the energy
        std::unique_ptr<ImplicitSurface> implUnique(implSurface);
        if (barrierData.repel)
        {
            std::cout << "Using implicit surface as obstacle, with power " << barrierData.power << " and weight " << barrierData.weight << std::endl;
            ImplicitObstacle *obstacle = new ImplicitObstacle(mesh, geom, std::move(implUnique), barrierData.power, barrierData.weight);
            flow->AddAdditionalEnergy(obstacle);
        }
        else
        {
            std::cout << "Using implicit surface as attractor, with power " << barrierData.power << " and weight " << barrierData.weight << std::endl;
            ImplicitAttractor *attractor = new ImplicitAttractor(mesh, geom, std::move(implUnique), uvs, barrierData.power, barrierData.weight);
            flow->AddAdditionalEnergy(attractor);
        }
    }

    void MainApp::AddPotential(scene::PotentialType pType, double weight)
    {
        switch (pType)
        {
        case scene::PotentialType::SquaredError:
        {
            SquaredError *errorPotential = new SquaredError(mesh, geom, weight);
            vertexPotential = errorPotential;
            flow->AddAdditionalEnergy(errorPotential);
            remesher.KeepVertexDataUpdated(&errorPotential->originalPositions);
            break;
        }
        case scene::PotentialType::Area:
        {
            TotalAreaPotential *areaPotential = new TotalAreaPotential(mesh, geom, weight);
            flow->AddAdditionalEnergy(areaPotential);
            break;
        }
        case scene::PotentialType::Volume:
        {
            TotalVolumePotential *volumePotential = new TotalVolumePotential(mesh, geom, weight);
            flow->AddAdditionalEnergy(volumePotential);
            break;
        }
        case scene::PotentialType::BoundaryLength:
        {
            BoundaryLengthPenalty *errorPotential = new BoundaryLengthPenalty(mesh, geom, weight);
            flow->AddAdditionalEnergy(errorPotential);
            break;
        }
        case scene::PotentialType::SoftAreaConstraint:
        {
            SoftAreaConstraint *softArea = new SoftAreaConstraint(mesh, geom, weight);
            flow->AddAdditionalEnergy(softArea);
            break;
        }
        case scene::PotentialType::SoftVolumeConstraint:
        {
            SoftVolumeConstraint *softVol = new SoftVolumeConstraint(mesh, geom, weight);
            flow->AddAdditionalEnergy(softVol);
            break;
        }
        case scene::PotentialType::Willmore:
        {
            WillmoreEnergy *willmore = new WillmoreEnergy(mesh, geom, weight);
            flow->AddAdditionalEnergy(willmore);
            break;
        }
        default:
        {
            std::cout << "Unknown potential type." << std::endl;
            break;
        }
        }
    }

    void MainApp::MeshImplicitSurface(ImplicitSurface *surface)
    {
        CIsoSurface<double> *iso = new CIsoSurface<double>();

        std::cout << "Meshing the supplied implicit surface using marching cubes..." << std::endl;

        const int numCells = 50;
        Vector3 center = surface->BoundingCenter();
        double diameter = surface->BoundingDiameter();
        double cellSize = diameter / numCells;
        double radius = diameter / 2;

        Vector3 lowerCorner = center - Vector3{radius, radius, radius};

        int numCorners = numCells + 1;

        double field[numCorners * numCorners * numCorners];

        int nSlice = numCorners * numCorners;
        int nRow = numCorners;

        for (int x = 0; x < numCorners; x++)
        {
            for (int y = 0; y < numCorners; y++)
            {
                for (int z = 0; z < numCorners; z++)
                {
                    Vector3 samplePt = lowerCorner + Vector3{(double)x, (double)y, (double)z} * cellSize;
                    double value = surface->SignedDistance(samplePt);
                    field[nSlice * z + nRow * y + x] = value;
                }
            }
        }

        iso->GenerateSurface(field, 0, numCells, numCells, numCells, cellSize, cellSize, cellSize);

        std::vector<glm::vec3> nodes;
        std::vector<std::array<size_t, 3>> triangles;

        int nVerts = iso->m_nVertices;

        for (int i = 0; i < nVerts; i++)
        {
            double x = iso->m_ppt3dVertices[i][0];
            double y = iso->m_ppt3dVertices[i][1];
            double z = iso->m_ppt3dVertices[i][2];

            Vector3 p = lowerCorner + Vector3{x, y, z};
            nodes.push_back(glm::vec3{p.x, p.y, p.z});
        }

        int nTris = iso->m_nTriangles;

        for (int i = 0; i < nTris; i++)
        {
            int i1 = iso->m_piTriangleIndices[3 * i];
            int i2 = iso->m_piTriangleIndices[3 * i + 1];
            int i3 = iso->m_piTriangleIndices[3 * i + 2];

            triangles.push_back({(size_t)i1, (size_t)i2, (size_t)i3});
        }

        implicitCount++;
        polyscope::registerSurfaceMesh("implicitSurface" + std::to_string(implicitCount), nodes, triangles);
        delete iso;
    }
} // namespace rsurfaces

// UI parameters
bool run = false;
bool takeScreenshots = false;
bool saveOBJs = false;
uint screenshotNum = 0;
uint objNum = 0;
bool uiNormalizeView = false;
bool remesh = true;
bool changeTopo = false;
bool areaRatios = false;

int partIndex = 4475;

void saveScreenshot(uint i)
{
    char buffer[5];
    std::snprintf(buffer, sizeof(buffer), "%04d", i);
    std::string fname = "frames/frame" + std::string(buffer) + ".png";
    polyscope::screenshot(fname, false);
    std::cout << "Saved screenshot to " << fname << std::endl;
}

void saveOBJ(rsurfaces::MeshPtr mesh, rsurfaces::GeomPtr geom, rsurfaces::GeomPtr geomOrig, uint i)
{

    char buffer[5];
    std::snprintf(buffer, sizeof(buffer), "%04d", i);
    std::string fname = "objs/frame" + std::string(buffer) + ".obj";
    rsurfaces::writeMeshToOBJ(mesh, geom, geomOrig, areaRatios, fname);
    std::cout << "Saved OBJ frame to " << fname << std::endl;
}

template <typename ItemType>
void selectFromDropdown(std::string label, const ItemType choices[], size_t nChoices, ItemType &store)
{
    using namespace rsurfaces;

    // Dropdown menu for list of remeshing mode settings
    if (ImGui::BeginCombo(label.c_str(), StringOfMode(store).c_str()))
    {
        for (size_t i = 0; i < nChoices; i++)
        {
            bool is_selected = (store == choices[i]);
            if (ImGui::Selectable(StringOfMode(choices[i]).c_str(), is_selected))
                store = choices[i];
            if (is_selected)
                ImGui::SetItemDefaultFocus();
        }
        ImGui::EndCombo();
    }
}

// A user-defined callback, for creating control panels (etc)
// Use ImGUI commands to build whatever you want here, see
// https://github.com/ocornut/imgui/blob/master/imgui.h
void customCallback()
{
    using namespace rsurfaces;

    const int INDENT = 10;
    const int ITEM_WIDTH = 160;

    ImGui::Text("Flow control");
    ImGui::BeginGroup();
    ImGui::Indent(INDENT);
    ImGui::PushItemWidth(ITEM_WIDTH);
    ImGui::Checkbox("Run flow", &run);
    ImGui::SameLine(ITEM_WIDTH, 2 * INDENT);
    ImGui::Checkbox("Normalize view", &uiNormalizeView);

    ImGui::Checkbox("Take screenshots", &takeScreenshots);
    ImGui::SameLine(ITEM_WIDTH, 2 * INDENT);
    if ((takeScreenshots && screenshotNum == 0) || ImGui::Button("Take screenshot", ImVec2{ITEM_WIDTH, 0}))
    {
        saveScreenshot(screenshotNum++);
    }

    ImGui::Checkbox("Write OBJs", &saveOBJs);
    ImGui::SameLine(ITEM_WIDTH, 2 * INDENT);
    if ((saveOBJs && objNum == 0) || ImGui::Button("Write OBJ", ImVec2{ITEM_WIDTH, 0}))
    {
        saveOBJ(MainApp::instance->mesh, MainApp::instance->geom, MainApp::instance->geomOrig, objNum++);
    }
    ImGui::Checkbox("Log performance", &MainApp::instance->logPerformance);
    ImGui::SameLine(ITEM_WIDTH, 2 * INDENT);
    ImGui::Checkbox("Show area ratios", &areaRatios);

    const GradientMethod methods[] = {GradientMethod::HsProjectedIterative,
                                      GradientMethod::HsProjected,
                                      GradientMethod::HsExactProjected,
                                      GradientMethod::H1Projected,
                                      GradientMethod::L2Unconstrained,
                                      GradientMethod::L2Projected,
                                      GradientMethod::AQP,
                                      GradientMethod::H1_LBFGS,
                                      GradientMethod::BQN_LBFGS,
                                      GradientMethod::H2Projected};

    selectFromDropdown("Method", methods, IM_ARRAYSIZE(methods), MainApp::instance->methodChoice);

    ImGui::Checkbox("Dynamic remeshing", &remesh);

    const remeshing::RemeshingMode rModes[] = {remeshing::RemeshingMode::FlipOnly,
                                               remeshing::RemeshingMode::SmoothAndFlip,
                                               remeshing::RemeshingMode::SmoothFlipAndCollapse};

    const remeshing::SmoothingMode sModes[] = {remeshing::SmoothingMode::Laplacian,
                                               remeshing::SmoothingMode::Circumcenter};

    const remeshing::FlippingMode fModes[] = {remeshing::FlippingMode::Delaunay,
                                              remeshing::FlippingMode::Degree};

    selectFromDropdown("Remeshing mode", rModes, IM_ARRAYSIZE(rModes), MainApp::instance->remesher.remeshingMode);
    selectFromDropdown("Smoothing mode", sModes, IM_ARRAYSIZE(sModes), MainApp::instance->remesher.smoothingMode);
    selectFromDropdown("Flipping mode", fModes, IM_ARRAYSIZE(fModes), MainApp::instance->remesher.flippingMode);

    ImGui::Checkbox("Curvature adaptive remeshing", &MainApp::instance->remesher.curvatureAdaptive);

    rsurfaces::MainApp::instance->HandlePicking();

    ImGui::InputInt("Iteration limit", &MainApp::instance->stepLimit);
    ImGui::InputInt("Real time limit (ms)", &MainApp::instance->realTimeLimit);

    if (uiNormalizeView != MainApp::instance->normalizeView)
    {
        rsurfaces::MainApp::instance->normalizeView = uiNormalizeView;
        rsurfaces::MainApp::instance->updateMeshPositions();
    }
    ImGui::PopItemWidth();
    if (ImGui::Button("Take 1 step", ImVec2{ITEM_WIDTH, 0}) || run)
    {
        MainApp::instance->TakeOptimizationStep(remesh, areaRatios);

        if (takeScreenshots)
        {
            saveScreenshot(screenshotNum++);
        }
        if (saveOBJs)
        {
            saveOBJ(MainApp::instance->mesh, MainApp::instance->geom, MainApp::instance->geomOrig, objNum++);
        }
        if ((MainApp::instance->stepLimit > 0 && MainApp::instance->numSteps >= MainApp::instance->stepLimit) ||
            (MainApp::instance->realTimeLimit > 0 && MainApp::instance->timeSpentSoFar >= MainApp::instance->realTimeLimit))
        {
            run = false;
            if (MainApp::instance->exitWhenDone)
            {
                std::exit(0);
            }
        }
    }

    ImGui::EndGroup();

    ImGui::Text("Accuracy tests");

    ImGui::BeginGroup();
    ImGui::Indent(INDENT);

    if (ImGui::Button("Create/destroy BVH", ImVec2{ITEM_WIDTH, 0}))
    {
        MainApp::instance->CreateAndDestroyBVH();
    }

    if (ImGui::Button("Test Willmore", ImVec2{ITEM_WIDTH, 0}))
    {
        MainApp::instance->TestWillmore();
    }
    ImGui::SameLine(ITEM_WIDTH, 2 * INDENT);
    if (ImGui::Button("Test TPObstacle0", ImVec2{ITEM_WIDTH, 0}))
    {
        MainApp::instance->TestObstacle0();
    }

    if (ImGui::Button("Test BarnesHut0", ImVec2{ITEM_WIDTH, 0}))
    {
        MainApp::instance->TestBarnesHut0();
    }
    ImGui::SameLine(ITEM_WIDTH, 2 * INDENT);
    if (ImGui::Button("Plot gradients", ImVec2{ITEM_WIDTH, 0}))
    {
        MainApp::instance->PlotGradients();
    }
    ImGui::EndGroup();

    ImGui::Text("Remeshing tests");

    ImGui::BeginGroup();
    ImGui::Indent(INDENT);

    // Section for remeshing tests
    if (ImGui::Button("Fix Delaunay"))
    {
        remeshing::fixDelaunay(MainApp::instance->mesh, MainApp::instance->geom);
        MainApp::instance->mesh->validateConnectivity();
        MainApp::instance->reregisterMesh();
    }
    ImGui::SameLine(ITEM_WIDTH, 2 * INDENT);

    if (ImGui::Button("Laplacian smooth"))
    {
        remeshing::smoothByLaplacian(MainApp::instance->mesh, MainApp::instance->geom);
        MainApp::instance->reregisterMesh();
    }

    if (ImGui::Button("Circumcenter smooth"))
    {
        remeshing::smoothByCircumcenter(MainApp::instance->mesh, MainApp::instance->geom);
        MainApp::instance->reregisterMesh();
    }
    ImGui::SameLine(ITEM_WIDTH, 2 * INDENT);

    if (ImGui::Button("Laplacian opt"))
    {
        for (int i = 0; i < 10; i++)
        {
            remeshing::smoothByLaplacian(MainApp::instance->mesh, MainApp::instance->geom);
            remeshing::fixDelaunay(MainApp::instance->mesh, MainApp::instance->geom);
        }
        MainApp::instance->reregisterMesh();
    }

    if (ImGui::Button("Circumcenter opt"))
    {
        for (int i = 0; i < 10; i++)
        {
            remeshing::smoothByCircumcenter(MainApp::instance->mesh, MainApp::instance->geom);
            remeshing::fixDelaunay(MainApp::instance->mesh, MainApp::instance->geom);
        }
        MainApp::instance->reregisterMesh();
    }
    ImGui::SameLine(ITEM_WIDTH, 2 * INDENT);
    if (ImGui::Button("Adjust edge lengths"))
    {
        remeshing::adjustEdgeLengths(MainApp::instance->mesh, MainApp::instance->geom, MainApp::instance->geomOrig, 0.01, 0.1, 0.001);
        MainApp::instance->reregisterMesh();
    }

    if (ImGui::Button("Adjust vert degrees"))
    {
        remeshing::adjustVertexDegrees(MainApp::instance->mesh, MainApp::instance->geom);
        MainApp::instance->reregisterMesh();
    }
    ImGui::SameLine(ITEM_WIDTH, 2 * INDENT);

    if (ImGui::Button("Face Weight Smooth"))
    {
        FaceData<double> faceWeight(*(MainApp::instance->mesh));
        for (int i = 0; i < 100; i++)
        {
            for (Face f : (MainApp::instance->mesh)->faces())
            {
                faceWeight[f] = 10 * (-remeshing::findBarycenter((MainApp::instance->geom), f).z + 1.5) + 0;
            }
            remeshing::smoothByFaceWeight(MainApp::instance->mesh, MainApp::instance->geom, faceWeight);
            //         remeshing::fixDelaunay(MainApp::instance->mesh, MainApp::instance->geom);
        }
        MainApp::instance->reregisterMesh();
    }
    if (ImGui::Button("Remesh"))
    {
        remeshing::remesh(MainApp::instance->mesh, MainApp::instance->geom, MainApp::instance->geomOrig);
        MainApp::instance->reregisterMesh();
    }
    ImGui::EndGroup();

    // testing stuff

    ImGui::Text("Testing stuff");

    ImGui::BeginGroup();
    ImGui::InputInt("partIndex", &partIndex);
    if (ImGui::Button("Test collapse edge"))
    {
        remeshing::testCollapseEdge(MainApp::instance->mesh, MainApp::instance->geom, partIndex);
        MainApp::instance->reregisterMesh();
    }
    if (ImGui::Button("Test stuff"))
    {
        remeshing::testStuff(MainApp::instance->mesh, MainApp::instance->geom, partIndex);
        //      MainApp::instance->mesh->validateConnectivity();
        //      MainApp::instance->mesh->compress();
        MainApp::instance->reregisterMesh();
    }

    if (ImGui::Button("Test stuff 2"))
    {
        remeshing::testStuff2(MainApp::instance->mesh, MainApp::instance->geom, MainApp::instance->geomOrig);
        MainApp::instance->reregisterMesh();
    }

    if (ImGui::Button("Show Vertex"))
    {
        remeshing::showEdge(MainApp::instance->mesh, MainApp::instance->geom, partIndex);
        MainApp::instance->reregisterMesh();
    }
    if (ImGui::Button("Validate"))
    {
        MainApp::instance->mesh->validateConnectivity();
    }

    if (ImGui::Button("Test vertex"))
    {
        remeshing::testVertex(MainApp::instance->mesh, MainApp::instance->geom, partIndex);
        //      MainApp::instance->mesh->validateConnectivity();
        //      MainApp::instance->mesh->compress();
        MainApp::instance->reregisterMesh();
    }
    if (ImGui::Button("Test edge"))
    {
        remeshing::testEdge(MainApp::instance->mesh, MainApp::instance->geom, partIndex);
        //      MainApp::instance->mesh->validateConnectivity();
        //      MainApp::instance->mesh->compress();
        MainApp::instance->reregisterMesh();
    }
    if (ImGui::Button("Test face"))
    {
        remeshing::testFace(MainApp::instance->mesh, MainApp::instance->geom, partIndex);
        //      MainApp::instance->mesh->validateConnectivity();
        //      MainApp::instance->mesh->compress();
        MainApp::instance->reregisterMesh();
    }
    ImGui::EndGroup();
}

struct MeshAndEnergy
{
    rsurfaces::TPEKernel *kernel;
    polyscope::SurfaceMesh *psMesh;
    rsurfaces::MeshPtr mesh;
    rsurfaces::GeomPtr geom;
    rsurfaces::UVDataPtr uvs;
    std::string meshName;
};

MeshAndEnergy initTPEOnMesh(std::string meshFile, double alpha, double beta)
{
    using namespace rsurfaces;
    std::cout << "Initializing tangent-point energy with (" << alpha << ", " << beta << ")" << std::endl;

    MeshUPtr u_mesh;
    std::unique_ptr<VertexPositionGeometry> u_geometry;
    std::unique_ptr<CornerData<Vector2>> uvs;

    // Load mesh
    std::tie(u_mesh, u_geometry, uvs) = readParameterizedMesh(meshFile);
    std::string mesh_name = polyscope::guessNiceNameFromPath(meshFile);

    std::cout << "Read " << uvs->size() << " UV coordinates" << std::endl;
    bool hasUVs = false;

    for (GCVertex v : u_mesh->vertices())
    {
        for (surface::Corner c : v.adjacentCorners())
        {
            Vector2 uv = (*uvs)[c];
            if (uv.x > 0 || uv.y > 0)
            {
                hasUVs = true;
            }
        }
    }

    if (hasUVs)
    {
        std::cout << "Mesh has nonzero UVs; using as flags for attractors" << std::endl;
    }
    else
    {
        std::cout << "Mesh has no UVs or all UVs are 0; not using as flags" << std::endl;
    }

    // Register the mesh with polyscope
    polyscope::SurfaceMesh *psMesh = polyscope::registerSurfaceMesh(mesh_name,
                                                                    u_geometry->inputVertexPositions, u_mesh->getFaceVertexList(),
                                                                    polyscopePermutations(*u_mesh));

    MeshPtr meshShared = std::move(u_mesh);
    GeomPtr geomShared = std::move(u_geometry);
    UVDataPtr uvShared = std::move(uvs);

    geomShared->requireFaceNormals();
    geomShared->requireFaceAreas();
    geomShared->requireVertexNormals();
    geomShared->requireVertexDualAreas();
    geomShared->requireVertexGaussianCurvatures();

    TPEKernel *tpe = new rsurfaces::TPEKernel(meshShared, geomShared, alpha, beta);

    return MeshAndEnergy{tpe, psMesh, meshShared, geomShared, (hasUVs) ? uvShared : 0, mesh_name};
}

enum class EnergyOverride
{
    TangentPoint,
    Coulomb,
    Willmore
};

rsurfaces::SurfaceFlow *setUpFlow(MeshAndEnergy &m, double theta, rsurfaces::scene::SceneData &scene, EnergyOverride eo)
{
    using namespace rsurfaces;

    SurfaceEnergy *energy;

    if (eo == EnergyOverride::Coulomb)
    {
        std::cout << "Using Coulomb energy in place of tangent-point energy" << std::endl;
        energy = new CoulombEnergy(m.kernel, theta);
    }
    else if (eo == EnergyOverride::Willmore)
    {
        std::cout << "Using Willmore energy in place of tangent-point energy" << std::endl;
        energy = new WillmoreEnergy(m.mesh, m.geom);
    }
    else
    {
        if (theta <= 0)
        {
            std::cout << "Theta was zero (or negative); using exact all-pairs energy." << std::endl;
            energy = new TPEnergyAllPairs(m.kernel->mesh, m.kernel->geom, m.kernel->alpha, m.kernel->beta);
            ;
        }
        else
        {
            std::cout << "Using Barnes-Hut energy with theta = " << theta << "." << std::endl;
            TPEnergyBarnesHut0 *bh = new TPEnergyBarnesHut0(m.kernel->mesh, m.kernel->geom, m.kernel->alpha, m.kernel->beta, theta);
            if (scene.disableNearField)
            {
                throw std::runtime_error("disable_near_field has not yet been ported to the new energy.");
            }
            energy = bh;
        }
    }

    SurfaceFlow *flow = new SurfaceFlow(energy);
    bool kernelRemoved = false;
    flow->allowBarycenterShift = scene.allowBarycenterShift;
    // Set these up here, so that we can aggregate all vertex pins into the same constraint
    Constraints::VertexPinConstraint *pinC = 0;
    Constraints::VertexNormalConstraint *normC = 0;

    for (scene::ConstraintData &data : scene.constraints)
    {
        switch (data.type)
        {
        case scene::ConstraintType::Barycenter:
            kernelRemoved = true;
            flow->addSimpleConstraint<Constraints::BarycenterConstraint3X>(m.mesh, m.geom);
            break;
        case scene::ConstraintType::TotalArea:
            flow->addSchurConstraint<Constraints::TotalAreaConstraint>(m.mesh, m.geom, data.targetMultiplier, data.numIterations);
            break;
        case scene::ConstraintType::TotalVolume:
            flow->addSchurConstraint<Constraints::TotalVolumeConstraint>(m.mesh, m.geom, data.targetMultiplier, data.numIterations);
            break;

        case scene::ConstraintType::BoundaryPins:
        {
            if (!pinC)
            {
                pinC = flow->addSimpleConstraint<Constraints::VertexPinConstraint>(m.mesh, m.geom);
            }
            // Manually add all of the boundary vertex indices as pins
            std::vector<size_t> boundaryInds;
            VertexIndices inds = m.mesh->getVertexIndices();
            for (GCVertex v : m.mesh->vertices())
            {
                if (v.isBoundary())
                {
                    boundaryInds.push_back(inds[v]);
                }
            }
            pinC->pinVertices(m.mesh, m.geom, boundaryInds);
            kernelRemoved = true;
        }
        break;

        case scene::ConstraintType::VertexPins:
        {
            if (!pinC)
            {
                pinC = flow->addSimpleConstraint<Constraints::VertexPinConstraint>(m.mesh, m.geom);
            }
            // Add the specified vertices as pins
            pinC->pinVertices(m.mesh, m.geom, scene.vertexPins);
            // Clear the data vector so that we don't add anything twice
            scene.vertexPins.clear();
            kernelRemoved = true;
        }
        break;

        case scene::ConstraintType::BoundaryNormals:
        {
            if (!normC)
            {
                normC = flow->addSimpleConstraint<Constraints::VertexNormalConstraint>(m.mesh, m.geom);
            }
            // Manually add all of the boundary vertex indices as pins
            std::vector<size_t> boundaryInds;
            VertexIndices inds = m.mesh->getVertexIndices();
            for (GCVertex v : m.mesh->vertices())
            {
                if (v.isBoundary())
                {
                    boundaryInds.push_back(inds[v]);
                }
            }
            normC->pinVertices(m.mesh, m.geom, boundaryInds);
        }

        case scene::ConstraintType::VertexNormals:
        {
            if (!normC)
            {
                normC = flow->addSimpleConstraint<Constraints::VertexNormalConstraint>(m.mesh, m.geom);
            }
            // Add the specified vertices as pins
            normC->pinVertices(m.mesh, m.geom, scene.vertexNormals);
            // Clear the data vector so that we don't add anything twice
            scene.vertexNormals.clear();
        }
        break;

        default:
            std::cout << "  * Skipping unrecognized constraint type" << std::endl;
            break;
        }
    }

    if (!kernelRemoved)
    {
        // std::cout << "Auto-adding barycenter constraint to eliminate constant kernel of Laplacian" << std::endl;
        // flow->addSimpleConstraint<Constraints::BarycenterConstraint3X>(m.mesh, m.geom);
    }

    return flow;
}

rsurfaces::scene::SceneData defaultScene(std::string meshName)
{
    using namespace rsurfaces;
    using namespace rsurfaces::scene;
    SceneData data;
    data.meshName = meshName;
    data.alpha = 6;
    data.beta = 12;
    data.constraints = std::vector<ConstraintData>({ConstraintData{scene::ConstraintType::Barycenter, 1, 0},
                                                    ConstraintData{scene::ConstraintType::TotalArea, 1, 0},
                                                    ConstraintData{scene::ConstraintType::TotalVolume, 1, 0}});
    return data;
}

int main(int argc, char **argv)
{
    using namespace rsurfaces;

    // Configure the argument parser
    args::ArgumentParser parser("geometry-central & Polyscope example project");
    args::Positional<std::string> inputFilename(parser, "mesh", "A mesh file.");
    args::ValueFlag<double> thetaFlag(parser, "Theta", "Theta value for Barnes-Hut approximation; 0 means exact.", args::Matcher{'t', "theta"});
    args::ValueFlagList<std::string> obstacleFiles(parser, "obstacles", "Obstacles to add", {'o'});
    args::Flag autologFlag(parser, "autolog", "Automatically start the flow, log performance, and exit when done.", {"autolog"});
    args::Flag coulombFlag(parser, "coulomb", "Use a coulomb energy instead of the tangent-point energy.", {"coulomb"});
    args::ValueFlag<int> threadFlag(parser, "threads", "How many threads to use in parallel.", {"threads"});

    polyscope::options::programName = "Repulsive Surfaces";
    polyscope::options::groundPlaneEnabled = false;

    std::cout << "Using Eigen version " << EIGEN_WORLD_VERSION << "." << EIGEN_MAJOR_VERSION << "." << EIGEN_MINOR_VERSION << std::endl;

    // Parse args
    try
    {
        parser.ParseCLI(argc, argv);
    }
    catch (args::Help)
    {
        std::cout << parser;
        return 0;
    }
    catch (args::ParseError e)
    {
        std::cerr << e.what() << std::endl;
        std::cerr << parser;
        return 1;
    }
    // Make sure a mesh name was given
    if (!inputFilename)
    {
        std::cerr << "Please specify a mesh file as argument" << std::endl;
        return EXIT_FAILURE;
    }

    if (threadFlag)
    {
        int nThreads = args::get(threadFlag);
        std::cout << "Using " << nThreads << " threads as specified." << std::endl;
        omp_set_num_threads(nThreads);
    }
    else
    {
        int default_threads = omp_get_max_threads() / 2 + 2;
        omp_set_num_threads(default_threads);
        std::cout << "Defaulting to " << default_threads << " threads." << std::endl;
    }

    double theta = 0.5;
    if (!thetaFlag)
    {
        std::cout << "Barnes-Hut theta value not specified; defaulting to theta = " << theta << std::endl;
    }
    else
    {
        theta = args::get(thetaFlag);
    }

    // Initialize polyscope
    polyscope::init();
    // Set the callback function
    polyscope::state::userCallback = customCallback;

    // Parse the input file, either as a scene file or as a mesh
    std::string inFile = args::get(inputFilename);
    scene::SceneData data;

    if (endsWith(inFile, ".txt") || endsWith(inFile, ".scene"))
    {
        std::cout << "Parsing " << inFile << " as scene file." << std::endl;
        data = scene::parseScene(inFile);
    }

    else if (endsWith(inFile, ".obj"))
    {
        std::cout << "Parsing " << inFile << " as OBJ mesh file." << std::endl;
        data = defaultScene(inFile);
    }

    else
    {
        throw std::runtime_error("Unknown file extension for " + inFile + ".");
    }

    bool useCoulomb = false;
    if (coulombFlag)
    {
        useCoulomb = true;
        std::cout << "Using Coulomb energy. (Note: Not expected to work well.)" << std::endl;
    }

    MeshAndEnergy m = initTPEOnMesh(data.meshName, data.alpha, data.beta);

    EnergyOverride eo = EnergyOverride::TangentPoint;
    if (useCoulomb)
    {
        eo = EnergyOverride::Coulomb;
    }
    else if (data.defaultMethod == GradientMethod::Willmore)
    {
        eo = EnergyOverride::Willmore;
    }

    SurfaceFlow *flow = setUpFlow(m, theta, data, eo);
    flow->disableNearField = data.disableNearField;

    MainApp::instance = new MainApp(m.mesh, m.geom, flow, m.psMesh, m.meshName);
    MainApp::instance->bh_theta = theta;
    MainApp::instance->kernel = m.kernel;
    MainApp::instance->stepLimit = data.iterationLimit;
    MainApp::instance->realTimeLimit = data.realTimeLimit;
    MainApp::instance->methodChoice = data.defaultMethod;
    MainApp::instance->sceneData = data;
    MainApp::instance->uvs = m.uvs;

    if (autologFlag)
    {
        std::cout << "Autolog flag was used; starting flow automatically." << std::endl;
        MainApp::instance->exitWhenDone = true;
        MainApp::instance->logPerformance = true;
        run = true;
        std::ofstream outfile;
        outfile.open(data.performanceLogFile, std::ios_base::out);
        outfile.close();
    }

    for (scene::PotentialData &p : data.potentials)
    {
        MainApp::instance->AddPotential(p.type, p.weight);
    }
    for (scene::ObstacleData &obs : data.obstacles)
    {
        MainApp::instance->AddObstacle(obs.obstacleName, obs.weight, obs.recenter);
    }
    for (scene::ImplicitBarrierData &barrierData : data.implicitBarriers)
    {
        MainApp::instance->AddImplicitBarrier(barrierData);
    }

    if (data.autoComputeVolumeTarget)
    {
        double targetVol = MainApp::instance->totalObstacleVolume * data.autoVolumeTargetRatio;
        std::cout << "Retargeting volume constraint to value " << targetVol << " (" << data.autoVolumeTargetRatio << "x obstacle volume)" << std::endl;
        MainApp::instance->flow->retargetSchurConstraintOfType<Constraints::TotalVolumeConstraint>(targetVol);
    }

    MainApp::instance->updateMeshPositions();

    // Give control to the polyscope gui
    polyscope::show();

    return EXIT_SUCCESS;
}
