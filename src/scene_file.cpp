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
                if (parts.size() >= 3)
                {
                    weight = stod(parts[2]);
                }
                if (parts[1] == "squared_error")
                {
                    data.potentials.push_back(PotentialData{PotentialType::SquaredError, weight});
                    cout << "  * Adding squared error potential (weight " << weight << ")" << endl;
                }
                else if (parts[1] == "area")
                {
                    data.potentials.push_back(PotentialData{PotentialType::Area, weight});
                    cout << "  * Adding area potential (weight " << weight << ")" << endl;
                }
                else if (parts[1] == "volume")
                {
                    data.potentials.push_back(PotentialData{PotentialType::Volume, weight});
                    cout << "  * Adding volume potential (weight " << weight << ")" << endl;
                }
                else if (parts[1] == "area_deviation")
                {
                    data.potentials.push_back(PotentialData{PotentialType::SoftAreaConstraint, weight});
                    cout << "  * Adding soft area potential (weight " << weight << ")" << endl;
                }
                else if (parts[1] == "volume_deviation")
                {
                    data.potentials.push_back(PotentialData{PotentialType::SoftVolumeConstraint, weight});
                    cout << "  * Adding soft volume potential (weight " << weight << ")" << endl;
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
                data.allowBarycenterShift = true;
            }
            else if (parts[0] == "constrain")
            {
                ConstraintData consData{getConstraintType(parts[1]), 1, 0};

                // Get the list of vertices specified to be pinned
                if (consData.type == ConstraintType::VertexPins)
                {
                    for (size_t i = 2; i < parts.size(); i++)
                    {
                        size_t pin = stoul(parts[i]);
                        data.vertexPins.push_back(pin);
                    }
                }

                // Likewise get the list of normals
                else if (consData.type == ConstraintType::VertexNormals)
                {
                    for (size_t i = 2; i < parts.size(); i++)
                    {
                        size_t pin = stoul(parts[i]);
                        data.vertexNormals.push_back(pin);
                    }
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
            else if (parts[0] == "obstacle")
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
                data.obstacles.push_back(obsData);
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