#include "spatial/bvh_data.h"

namespace rsurfaces
{

    GCFace BVHData::getSingleFace(MeshPtr mesh) const
    {
        if (nodeType != BVHNodeType::Leaf)
        {
            std::cerr << "Tried to getSingleFace() from a non-leaf node" << std::endl;
            exit(1);
        }
        return mesh->face(elementID);
    }

    bool BVHData::isAdmissibleFrom(Vector3 atPos, double thresholdTheta) const
    {
        if (nodeType == BVHNodeType::Leaf)
        {
            if (centerOfMass == atPos)
                return false;
            else
                return true;
        }
        if (nodeType == BVHNodeType::Interior)
        {
            double d = norm(centerOfMass - atPos);
            return nodeRatioBox(minCoords, maxCoords, d) < thresholdTheta;
        }
        else
            return true;
    }

    MassNormalPoint BVHData::GetMassNormalPoint() const
    {
        return MassNormalPoint{totalMass, averageNormal, centerOfMass, minCoords, maxCoords, elementID};
    }
} // namespace rsurfaces
