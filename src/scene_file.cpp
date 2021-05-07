#include "scene_file.h"

#include <fstream>

namespace rsurfaces
{
    bool endsWith(std::string const &fullString, std::string const &ending)
    {
        if (fullString.length() >= ending.length())
        {
            return (0 == fullString.compare(fullString.length() - ending.length(), ending.length(), ending));
        }
        else
        {
            return false;
        }
    }

    namespace scene
    {
        std::string getDirectoryFromPath(std::string str)
        {
            using namespace std;
            vector<string> parts;
            splitString(str, parts, '/');

            int nParts = parts.size();
            if (nParts == 1)
                return "./";

            string path = "";

            for (int i = 0; i < nParts - 1; i++)
            {
                path = path + parts[i] + "/";
            }

            return path;
        }

        std::string nameOfConstraint(ConstraintType type)
        {
            switch (type)
            {
            case ConstraintType::TotalArea:
                return "area";
            case ConstraintType::TotalVolume:
                return "volume";
            case ConstraintType::Barycenter:
                return "barycenter";
            case ConstraintType::BoundaryPins:
                return "boundary pins";
            case ConstraintType::VertexPins:
                return "vertex pins";
            case ConstraintType::BoundaryNormals:
                return "boundary normals";
            case ConstraintType::VertexNormals:
                return "vertex normals";
            default:
                return "unknown";
            }
        }

        GradientMethod methodOfName(std::string name)
        {
            if (name == "hs")
            {
                return GradientMethod::HsProjectedIterative;
            }
            else if (name == "aqp")
            {
                return GradientMethod::AQP;
            }
            else if (name == "bqn")
            {
                return GradientMethod::BQN_LBFGS;
            }
            else if (name == "h1")
            {
                return GradientMethod::H1Projected;
            }
            else if (name == "h1-lbfgs")
            {
                return GradientMethod::H1_LBFGS;
            }
            else if (name == "h2")
            {
                return GradientMethod::H2Projected;
            }
            else if (name == "willmore")
            {
                return GradientMethod::Willmore;
            }
            else
            {
                throw std::runtime_error("Unknown method name " + name);
            }
        }

        ConstraintType getConstraintType(std::string consType)
        {

            if (consType == "area")
            {
                return ConstraintType::TotalArea;
            }
            else if (consType == "volume")
            {
                return ConstraintType::TotalVolume;
            }
            else if (consType == "barycenter")
            {
                return ConstraintType::Barycenter;
            }
            else if (consType == "boundary_vertices")
            {
                return ConstraintType::BoundaryPins;
            }
            else if (consType == "vertices" || consType == "vertex")
            {
                return ConstraintType::VertexPins;
            }
            else if (consType == "boundary_normals")
            {
                return ConstraintType::BoundaryNormals;
            }
            else if (consType == "normal" || consType == "normals")
            {
                return ConstraintType::VertexNormals;
            }
            else
            {
                std::cerr << "  * Unknown constraint type " << consType << std::endl;
                std::exit(1);
            }
        }

        void processLine(SceneData &data, std::string dir_root, std::vector<std::string> &parts)
        {
            using namespace std;
            if (parts.size() == 0)
            {
                return;
            }
            if (parts[0] == "repel_mesh")
            {
                data.meshName = dir_root + parts[1];
                cout << "  * Using mesh at " << data.meshName << endl;
                if (parts.size() == 4)
                {
                    data.alpha = stod(parts[2]);
                    data.beta = stod(parts[3]);
                    cout << "  * Using exponents (" << data.alpha << ", " << data.beta << ")" << endl;
                }
            }
            else if (parts[0] == "minimize")
            {
                double weight = 1;
                double targetValue = 0;
                if (parts.size() >= 3)
                {
                    weight = stod(parts[2]);
                    if (parts.size() >= 4)
                    {
                        targetValue = stod(parts[3]);
                    }
                }
                if (parts[1] == "squared_error")
                {
                    data.potentials.push_back(PotentialData{PotentialType::SquaredError, weight, targetValue});
                    cout << "  * Adding squared error potential (weight " << weight << ")" << endl;
                }
                if (parts[1] == "boundary_length")
                {
                    data.potentials.push_back(PotentialData{PotentialType::BoundaryLength, weight, targetValue});
                    cout << "  * Adding boundary length potential (weight " << weight << ", target value = " << targetValue << ")" << endl;
                }
                else if (parts[1] == "area")
                {
                    data.potentials.push_back(PotentialData{PotentialType::Area, weight, targetValue});
                    cout << "  * Adding area potential (weight " << weight << ")" << endl;
                }
                else if (parts[1] == "volume")
                {
                    data.potentials.push_back(PotentialData{PotentialType::Volume, weight, targetValue});
                    cout << "  * Adding volume potential (weight " << weight << ")" << endl;
                }
                else if (parts[1] == "area_deviation")
                {
                    data.potentials.push_back(PotentialData{PotentialType::SoftAreaConstraint, weight, targetValue});
                    cout << "  * Adding soft area potential (weight " << weight << ")" << endl;
                }
                else if (parts[1] == "volume_deviation")
                {
                    data.potentials.push_back(PotentialData{PotentialType::SoftVolumeConstraint, weight, targetValue});
                    cout << "  * Adding soft volume potential (weight " << weight << ")" << endl;
                }
                else if (parts[1] == "willmore")
                {
                    data.potentials.push_back(PotentialData{PotentialType::Willmore, weight, targetValue});
                    cout << "  * Adding Willmore energy term (weight " << weight << ")" << endl;
                }
            }
            else if (parts[0] == "iteration_limit")
            {
                data.iterationLimit = stoi(parts[1]);
                std::cout << "Adding iteration limit of " << data.iterationLimit << std::endl;
            }
            else if (parts[0] == "time_limit")
            {
                data.realTimeLimit = stol(parts[1]);
                std::cout << "Adding time limit of " << data.realTimeLimit << " milliseconds" << std::endl;
            }
            else if (parts[0] == "allow_barycenter_shift")
            {
                std::cout << "Allowing barycenter shift. Note that this does not work with vertex pins." << std::endl;
                data.allowBarycenterShift = true;
            }
            else if (parts[0] == "disable_near_field")
            {
                data.disableNearField = true;
            }
            else if (parts[0] == "constrain")
            {
                ConstraintData consData{getConstraintType(parts[1]), 1, 0};

                // Get the list of vertices specified to be pinned
                if (consData.type == ConstraintType::VertexPins)
                {
                    std::cout << "WARNING: Vertex pins currently do not work with splits or collapses." << std::endl;
                    std::cout << "Make sure to change the remeshing mode to \"smooth + flip\"." << std::endl;
                    size_t i = 2;
                    bool hasMove = false;
                    size_t pinsStart = data.vertexPins.size();

                    for (i = 2; i < parts.size(); i++)
                    {
                        if (parts[i] == "move")
                        {
                            hasMove = true;
                            break;
                        }
                        size_t pin = stoul(parts[i]);
                        data.vertexPins.push_back(VertexPinData{pin, Vector3{0, 0, 0}, 0});
                    }

                    if (hasMove)
                    {
                        double offX = stod(parts[i+1]);
                        double offY = stod(parts[i+2]); 
                        double offZ = stod(parts[i+3]); 
                        size_t iters = stoul(parts[i+4]);
                        Vector3 pinOff{offX, offY, offZ};

                        std::cout << "Setting offset for pins " << pinsStart << " - " << data.vertexPins.size()
                            << " to " << pinOff << " over " << iters << " iterations" << std::endl;

                        for (size_t j = pinsStart; j < data.vertexPins.size(); j++)
                        {
                            data.vertexPins[j].offset = pinOff;
                            data.vertexPins[j].iterations = iters;
                        }
                    }
                }

                // Likewise get the list of normals
                else if (consData.type == ConstraintType::VertexNormals)
                {
                    std::cout << "WARNING: Vertex normal constraints currently not supported." << std::endl;
                    // for (size_t i = 2; i < parts.size(); i++)
                    // {
                    //     size_t pin = stoul(parts[i]);
                    //     data.vertexNormals.push_back(pin);
                    // }
                }

                else if (parts.size() == 4)
                {
                    consData.targetMultiplier = stod(parts[2]);
                    consData.numIterations = stoi(parts[3]);
                    cout << "  * Adding " << nameOfConstraint(consData.type) << " constraint, growing "
                         << consData.targetMultiplier << "x over " << consData.numIterations
                         << " iterations" << endl;
                }
                else
                {
                    cout << "  * Adding constant " << nameOfConstraint(consData.type) << " constraint" << endl;
                }
                data.constraints.push_back(consData);
            }
            else if (parts[0] == "obstacle" || parts[0] == "point_cloud_obstacle")
            {
                ObstacleData obsData;
                obsData.obstacleName = dir_root + parts[1];
                if (parts.size() >= 3)
                {
                    obsData.weight = stod(parts[2]);
                    if (parts.size() == 4 && parts[3] == "centered")
                    {
                        obsData.recenter = true;
                    }
                }
                else
                {
                    obsData.weight = 1;
                }
                obsData.asPointCloud = (parts[0] == "point_cloud_obstacle");

                data.obstacles.push_back(obsData);
            }

            else if (parts[0] == "implicit")
            {
                ImplicitBarrierData implData;

                if (parts[1] == "sphere")
                {
                    implData.type = ImplicitType::Sphere;
                }
                else if (parts[1] == "torus")
                {
                    implData.type = ImplicitType::Torus;
                }
                else if (parts[1] == "plane")
                {
                    implData.type = ImplicitType::Plane;
                }
                else
                {
                    throw std::runtime_error("Unrecognized implicit type " + parts[1]);
                }

                if (parts[2] == "repel")
                {
                    implData.repel = true;
                }
                else if (parts[2] == "attract")
                {
                    implData.repel = false;
                }
                else
                {
                    throw std::runtime_error("Implicit action must be either 'attract' or 'repel'.");
                }

                implData.power = stod(parts[3]);
                implData.weight = stod(parts[4]);

                for (size_t i = 5; i < parts.size(); i++)
                {
                    implData.parameters.push_back(stod(parts[i]));
                }

                data.implicitBarriers.push_back(implData);
            }

            else if (parts[0] == "autotarget_volume")
            {
                std::cout << "Will ignore volume constraint parameters and auto-compute targets based on obstacle volume." << std::endl;
                data.autoComputeVolumeTarget = true;
                data.autoVolumeTargetRatio = stod(parts[1]);
            }

            else if (parts[0] == "method")
            {
                data.defaultMethod = methodOfName(parts[1]);
                std::cout << "Set default method to " << parts[1] << " (" << (int)data.defaultMethod << ")" << std::endl;
            }
            else if (parts[0] == "log")
            {
                std::string sep = "";
                if (dir_root[dir_root.size() - 1] != '/')
                {
                    sep = "/";
                }
                data.performanceLogFile = dir_root + sep + parts[1];
                std::cout << "Logging to " << data.performanceLogFile << std::endl;
            }
            else
            {
                cout << "  * Unrecognized statement: " << parts[0] << endl;
            }
        }

        SceneData parseScene(std::string filename)
        {
            using namespace std;

            string directory = getDirectoryFromPath(filename);
            ifstream file;
            file.open(filename);

            SceneData data;

            std::vector<std::string> parts;
            for (string line; getline(file, line);)
            {
                if (line == "" || line == "\n" || line[0] == '#')
                    continue;
                parts.clear();
                splitString(line, parts, ' ');
                processLine(data, directory, parts);
            }

            return data;
        }

    } // namespace scene

} // namespace rsurfaces