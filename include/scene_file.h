#pragma once

#include <iostream>
#include <vector>
#include <sstream>

namespace rsurfaces
{
    bool endsWith(std::string const &fullString, std::string const &ending);

    namespace scene
    {
        enum class ConstraintType
        {
            Barycenter,
            TotalArea,
            TotalVolume,
            BoundaryPins
        };

        struct ConstraintData
        {
            ConstraintType type;
            double targetMultiplier;
            long numIterations;
        };

        struct ObstacleData
        {
            std::string obstacleName;
            double weight;
        };

        struct SceneData
        {
            std::string meshName;
            double alpha;
            double beta;
            std::vector<ObstacleData> obstacles;
            std::vector<ConstraintData> constraints;
        };

        template <class Container>
        void splitString(const std::string &str, Container &cont, char delim = ' ')
        {
            std::stringstream ss(str);
            std::string token;
            while (std::getline(ss, token, delim))
            {
                cont.push_back(token);
            }
        }
        SceneData parseScene(std::string filename);

    } // namespace scene

} // namespace rsurfaces