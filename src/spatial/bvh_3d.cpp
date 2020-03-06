#include "spatial/bvh_3d.h"
#include "helpers.h"

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

inline MassPoint meshFaceToBody(const GCFace &f, GeomPtr &geom, FaceIndices &indices)
{
    Vector3 pos = faceBarycenter(geom, f);
    double mass = geom->faceArea(f);

    return MassPoint{mass, pos, indices[f]};
}

inline double GetCoordFromBody(MassPoint &mp, int axis)
{
    switch (axis)
    {
    case 0:
        return mp.point.x;
    case 1:
        return mp.point.y;
    case 2:
        return mp.point.z;
    default:
        std::cout << "Invalid axis passed to GetCoordFromBody: " << axis << std::endl;
        exit(1);
        return 0;
    }
}

BVHNode3D *CreateBVHFromMesh(MeshPtr &mesh, GeomPtr &geom)
{
    std::vector<MassPoint> verts(mesh->nFaces());
    FaceIndices indices = mesh->getFaceIndices();

    // Loop over all the vertices
    for (const GCFace &f : mesh->faces())
    {
        MassPoint curBody = meshFaceToBody(f, geom, indices);
        // Put vertex body into full list
        verts[curBody.elementID] = curBody;
    }

    BVHNode3D *tree = new BVHNode3D(verts, 0, 0);
    return tree;
}

BVHNode3D::BVHNode3D(std::vector<MassPoint> &points, int axis, BVHNode3D *root)
{
    // Split the points into sets somehow
    thresholdTheta = 0.25;
    splitAxis = axis;

    if (!root)
        bvhRoot = this;
    else
        bvhRoot = root;

    // If we have no points, then the node is empty
    if (points.size() == 0)
    {
        nodeType = BVHNodeType::Empty;
        totalMass = 0;
        centerOfMass = Vector3{0, 0, 0};
        elementID = -1;
    }
    // If we have only one point, then the node is a leaf
    else if (points.size() == 1)
    {
        MassPoint mp = points[0];
        nodeType = BVHNodeType::Leaf;
        totalMass = mp.mass;
        centerOfMass = mp.point;
        minCoords = mp.point;
        maxCoords = mp.point;
        elementID = mp.elementID;
    }
    // Otherwise, we need to recursively split and compute averages
    else
    {
        nodeType = BVHNodeType::Interior;
        elementID = -1;
        // Reserve space for splitting the points into lesser and greater
        int nPoints = points.size();
        std::vector<MassPoint> lesserPoints;
        lesserPoints.reserve(nPoints / 2 + 1);
        std::vector<MassPoint> greaterPoints;
        greaterPoints.reserve(nPoints / 2 + 1);

        // Compute the plane over which to split the points
        splitPoint = AxisSplittingPlane(points, axis);

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

        // If a root node was provided when constructing this, keep it;
        // otherwise, this must be the root, so use it.
        BVHNode3D *nextRoot = (root) ? root : this;
        // Add the children
        BVHNode3D *lesserNode = new BVHNode3D(lesserPoints, nextAxis, nextRoot);
        BVHNode3D *greaterNode = new BVHNode3D(greaterPoints, nextAxis, nextRoot);
        children.push_back(lesserNode);
        children.push_back(greaterNode);
        // Get the averages from children
        averageDataFromChildren();
    }
}

BVHNode3D::~BVHNode3D()
{
    for (size_t i = 0; i < children.size(); i++)
    {
        delete children[i];
    }
}

void BVHNode3D::printSummary()
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
                  << ",\n  " << children.size() << " children)" << std::endl;

        for (BVHNode3D *child : children)
        {
            child->printSummary();
        }
    }
}

MassPoint BVHNode3D::GetMassPoint()
{
    return MassPoint{totalMass, centerOfMass, elementID};
}

GCFace BVHNode3D::getSingleFace(MeshPtr &mesh)
{
    if (nodeType != BVHNodeType::Leaf) {
        std::cerr << "Tried to getSingleFace() from a non-leaf node" << std::endl;
        exit(1);
    }
    return mesh->face(elementID);
}

void BVHNode3D::averageDataFromChildren()
{
    totalMass = 0;
    centerOfMass = Vector3{0, 0, 0};
    minCoords = children[0]->minCoords;
    maxCoords = children[0]->maxCoords;

    for (BVHNode3D *child : children)
    {
        totalMass += child->totalMass;
        centerOfMass += child->totalMass * child->centerOfMass;
        minCoords = vectorMin(minCoords, child->minCoords);
        maxCoords = vectorMax(maxCoords, child->maxCoords);
    }

    centerOfMass /= totalMass;
}

double BVHNode3D::AxisSplittingPlane(std::vector<MassPoint> &points, int axis)
{
    size_t nPoints = points.size();
    std::vector<double> coords(nPoints);

    for (size_t i = 0; i < nPoints; i++)
    {
        coords[i] = GetCoordFromBody(points[i], axis);
    }

    std::sort(coords.begin(), coords.end());

    size_t splitIndex = -1;
    double minWidths = INFINITY;

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

bool BVHNode3D::shouldUseCell(Vector3 atPos)
{
    double d = norm(centerOfMass - atPos);
    // Vector2 ratios = viewspaceBounds(vertPos) / d;
    // TODO: take into account some tangent-related criteria?
    // return fmax(ratios.x, ratios.y) < thresholdTheta;
    return nodeRatio(d) < thresholdTheta;
}

void BVHNode3D::recomputeCentersOfMass(MeshPtr &mesh, GeomPtr &geom)
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
        minCoords = centerOfMass;
        maxCoords = centerOfMass;
    }
    else
    {
        // Recursively compute bounds for all children
        for (size_t i = 0; i < children.size(); i++)
        {
            children[i]->recomputeCentersOfMass(mesh, geom);
        }
        averageDataFromChildren();
    }
}

} // namespace rsurfaces