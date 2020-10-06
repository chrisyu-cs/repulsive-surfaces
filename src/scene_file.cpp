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
            default:
                return "unknown";
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
            else
            {
                std::cerr << "  * Unknown constraint type " << consType << std::endl;
                std::exit(1);
            }
        }

        void processLine(SceneData &data, std::string dir_root, std::vector<std::string> &parts)
        {
            using namespace std;
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
            else if (parts[0] == "constrain")
            {
                ConstraintData consData{getConstraintType(parts[1]), 1, 0};

                if (consData.type == ConstraintType::VertexPins)
                {
                    for (size_t i = 2; i < parts.size(); i++)
                    {
                        size_t pin = stoul(parts[i]);
                        data.vertexPins.push_back(pin);
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
                if (parts.size() == 3)
                {
                    obsData.weight = stod(parts[2]);
                }
                else
                {
                    obsData.weight = 1;
                }
                data.obstacles.push_back(obsData);
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