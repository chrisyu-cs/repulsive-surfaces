#include "spatial/bvh_6d.h"

namespace rsurfaces
{

    inline int NextAxis(int axis)
    {
        switch (axis)
        {
        case 0:
            return 1;
        case 1:
            return 2;
        case 2:
            return 0;
        default:
            return 0;
        }
    }

    inline double GetCoordFromBody(MassNormalPoint &mp, int axis)
    {
        switch (axis)
        {
        case 0:
            return mp.point.x;
        case 1:
            return mp.point.y;
        case 2:
            return mp.point.z;
        case 3:
            return mp.normal.x;
        case 4:
            return mp.normal.y;
        case 5:
            return mp.normal.z;
        default:
            std::cout << "Invalid axis passed to GetCoordFromBody: " << axis << std::endl;
            exit(1);
            return 0;
        }
    }

    BVHNode6D::BVHNode6D(std::vector<MassNormalPoint> &points, int axis)
    {
        // If we have no points, then the node is empty
        if (points.size() == 0)
        {
            nodeType = BVHNodeType::Empty;
            totalMass = 0;
            centerOfMass = Vector3{0, 0, 0};
            averageNormal = Vector3{0, 0, 0};
            elementID = -1;
            numNodesInBranch = 1;
            nElements = 0;
            children[0] = 0;
            children[1] = 0;
        }
        // If we have only one point, then the node is a leaf
        else if (points.size() == 1)
        {
            MassNormalPoint mp = points[0];
            nodeType = BVHNodeType::Leaf;
            totalMass = mp.mass;
            centerOfMass = mp.point;
            averageNormal = mp.normal;
            minCoords = mp.minCoords;
            maxCoords = mp.maxCoords;
            elementID = mp.elementID;
            numNodesInBranch = 1;
            nElements = 1;
            clusterIndices.resize(1);
            clusterIndices[0] = elementID;
            children[0] = 0;
            children[1] = 0;
        }
        // Otherwise, we need to recursively split and compute averages
        else
        {
            nodeType = BVHNodeType::Interior;
            elementID = -1;
            // Reserve space for splitting the points into lesser and greater
            int nPoints = points.size();
            std::vector<MassNormalPoint> lesserPoints;
            lesserPoints.reserve(nPoints / 2 + 1);
            std::vector<MassNormalPoint> greaterPoints;
            greaterPoints.reserve(nPoints / 2 + 1);

            // Compute the plane over which to split the points
            double splitPoint = AxisSplittingPlane(points, axis);
            int firstAxis = axis;
            // If we failed to find a split point because all of the points
            // are aligned on this axis, then try again with the next one
            while (isinff(splitPoint))
            {
                axis = NextAxis(axis);
                if (axis == firstAxis)
                {
                    std::cerr << "ERROR: Mesh includes multiple vertices with identical positions." << std::endl;
                    std::exit(1);
                }
                splitPoint = AxisSplittingPlane(points, axis);
            }

            // Split the points over the median
            for (int i = 0; i < nPoints; i++)
            {
                double coord = GetCoordFromBody(points[i], axis);
                if (coord <= splitPoint)
                    lesserPoints.push_back(points[i]);
                else
                    greaterPoints.push_back(points[i]);
            }
            // Recursively construct children
            int nextAxis = NextAxis(axis);

            // Add the children
            BVHNode6D *lesserNode = new BVHNode6D(lesserPoints, nextAxis);
            BVHNode6D *greaterNode = new BVHNode6D(greaterPoints, nextAxis);
            children[0] = lesserNode;
            children[1] = greaterNode;
            numNodesInBranch = lesserNode->numNodesInBranch + greaterNode->numNodesInBranch + 1;

            // Get the averages from children
            averageDataFromChildren();
            mergeIndicesFromChildren();
        }
    }

    BVHNode6D::~BVHNode6D()
    {
        for (size_t i = 0; i < 2; i++)
        {
            if (children[i])
            {
                delete children[i];
            }
        }
    }

    size_t BVHNode6D::assignIDsRecursively(size_t startID)
    {
        nodeID = startID;
        size_t nextID = nodeID + 1;
        if (nodeType == BVHNodeType::Interior)
        {
            for (size_t i = 0; i < BVH_N_CHILDREN; i++)
            {
                nextID = children[i]->assignIDsRecursively(nextID);
            }
        }
        return nextID;
    }

    void BVHNode6D::printSummary()
    {
        if (nodeType == BVHNodeType::Empty)
        {
            std::cout << "Empty node" << std::endl;
        }
        else if (nodeType == BVHNodeType::Leaf)
        {
            std::cout << "Leaf node (mass " << totalMass << ", center " << centerOfMass << ")" << std::endl;
        }
        else
        {
            std::cout << "Interior node (mass " << totalMass << ",\n  center " << centerOfMass
                      << ",\n  " << BVH_N_CHILDREN << " children)" << std::endl;

            for (BVHNode6D *child : children)
            {
                child->printSummary();
            }
        }
    }

    MassNormalPoint BVHNode6D::GetMassNormalPoint()
    {
        return MassNormalPoint{totalMass, averageNormal, centerOfMass, minCoords, maxCoords, elementID};
    }

    void BVHNode6D::averageDataFromChildren()
    {
        totalMass = 0;
        centerOfMass = Vector3{0, 0, 0};
        averageNormal = Vector3{0, 0, 0};
        minCoords = children[0]->minCoords;
        maxCoords = children[0]->maxCoords;
        nElements = 0;

        for (BVHNode6D *child : children)
        {
            totalMass += child->totalMass;
            centerOfMass += child->totalMass * child->centerOfMass;
            averageNormal += child->totalMass * child->averageNormal;
            minCoords = vectorMin(minCoords, child->minCoords);
            maxCoords = vectorMax(maxCoords, child->maxCoords);
            nElements += child->nElements;
        }

        centerOfMass /= totalMass;
        averageNormal = averageNormal.normalize();
    }

    void BVHNode6D::mergeIndicesFromChildren()
    {
        clusterIndices.resize(nElements);
        int currI = 0;

        for (BVHNode6D *child : children)
        {
            for (size_t i = 0; i < child->clusterIndices.size(); i++)
            {
                clusterIndices[currI] = child->clusterIndices[i];
                currI++;
            }
        }
    }

    double BVHNode6D::AxisSplittingPlane(std::vector<MassNormalPoint> &points, int axis)
    {
        size_t nPoints = points.size();
        std::vector<double> coords(nPoints);

        for (size_t i = 0; i < nPoints; i++)
        {
            coords[i] = GetCoordFromBody(points[i], axis);
        }

        std::sort(coords.begin(), coords.end());

        // Check to see if there is actually spread between the points on this axis.
        // If there isn't, then this axis can't be split along
        double spread = coords[coords.size() - 1] - coords[0];
        if (spread <= 1e-10)
        {
            return INFINITY;
        }

        size_t splitIndex = -1;
        double minWidths = INFINITY;

        // Find the partition that will result in the smallest sum of squared widths along this axis
        for (size_t i = 0; i < nPoints; i++)
        {
            double width1 = coords[i] - coords[0];
            double width2 = (i == nPoints - 1) ? 0 : coords[nPoints - 1] - coords[i + 1];

            double sumSquares = width1 * width1 + width2 * width2;
            if (sumSquares < minWidths)
            {
                minWidths = sumSquares;
                splitIndex = i;
            }
        }

        double splitPoint = (coords[splitIndex] + coords[splitIndex + 1]) / 2;
        return splitPoint;
    }

    bool BVHNode6D::isAdmissibleFrom(Vector3 atPos, double thresholdTheta)
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

    void BVHNode6D::recomputeCentersOfMass(MeshPtr const &mesh, GeomPtr const &geom)
    {
        if (nodeType == BVHNodeType::Empty)
        {
            totalMass = 0;
            centerOfMass = Vector3{0, 0, 0};
        }
        // For a leaf, just set centers and bounds from the one body
        else if (nodeType == BVHNodeType::Leaf)
        {
            GCFace f = mesh->face(elementID);
            totalMass = geom->faceArea(f);
            centerOfMass = faceBarycenter(geom, f);
            minCoords = minCoordOfFace(f, geom);
            maxCoords = maxCoordOfFace(f, geom);
        }
        else
        {
            // Recursively compute bounds for all children
            for (size_t i = 0; i < BVH_N_CHILDREN; i++)
            {
                children[i]->recomputeCentersOfMass(mesh, geom);
            }
            averageDataFromChildren();
        }
    }

} // namespace rsurfaces