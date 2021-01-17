#include "remeshing/remeshing.h"

namespace rsurfaces
{
    namespace remeshing
    {
        using std::cout;
        using std::queue;
        
        // immediate functions not in geometry-central
        
        Vector3 vertexNormal(GeomPtr const &geometry, Vertex v){
            Vector3 norm = Vector3::zero();
            for(Corner c : v.adjacentCorners()){
                norm += geometry->cornerAngle(c) * geometry->faceNormal(c.face());
            }
            //std::cerr<<norm<<std::endl;
            return normalize(norm);
        }
        
        double vertexAngleSum(GeomPtr const &geometry, Vertex v){
            double sum = 0;
            for(Corner c : v.adjacentCorners()){
                sum += geometry->cornerAngle(c);
            }
            return sum;
        }
        
        double vertexGaussianCurvature(GeomPtr const &geometry, Vertex v){
            return 2*PI - vertexAngleSum(geometry, v);
        }
        
        double faceGaussianCurvature(GeomPtr const &geometry, Face f){
            Halfedge he = f.halfedge();
            double a1 = geometry->cornerAngle(he.corner()) * 2*PI/vertexAngleSum(geometry, he.vertex());
            he = he.next();
            double a2 = geometry->cornerAngle(he.corner()) * 2*PI/vertexAngleSum(geometry, he.vertex());
            he = he.next();
            double a3 = geometry->cornerAngle(he.corner()) * 2*PI/vertexAngleSum(geometry, he.vertex());
            return PI - a1 - a2 - a3;
        }
        
        double faceMeanCurvature(GeomPtr const &geometry, Face f){
            Halfedge he = f.halfedge();
            double h1 = geometry->vertexMeanCurvature(he.vertex());
            he = he.next();
            double h2 = geometry->vertexMeanCurvature(he.vertex());
            he = he.next();
            double h3 = geometry->vertexMeanCurvature(he.vertex());
            return (h1 + h2 + h3)/3;
        }
        
        // helper functions
        
        inline Vector3 projectToPlane(Vector3 v, Vector3 norm)
        {
            return v - norm * dot(norm, v);
        }
        
        inline int degreeDifference(int d1, int d2, int d3, int d4)
        {
            //return abs(d1 - 6) + abs(d2- 6) + abs(d3 - 6) + abs(d4 - 6);
            return abs(d1 - 6)*abs(d1 - 6) + abs(d2- 6)*abs(d2 - 6) + abs(d3 - 6)*abs(d3 - 6) + abs(d4 - 6)*abs(d4 - 6);
        }
        
        inline Vector3 edgeMidpoint(MeshPtr const &mesh, GeomPtr const &geometry, Edge e)
        {
            Vector3 endPos1 = geometry->inputVertexPositions[e.halfedge().tailVertex()];
            Vector3 endPos2 = geometry->inputVertexPositions[e.halfedge().tipVertex()];
            return (endPos1+endPos2)/2;
        }
        
        Vector3 findCircumcenter(Vector3 p1, Vector3 p2, Vector3 p3)
        {
            // barycentric coordinates of circumcenter
            double a = (p3 - p2).norm();
            double b = (p3 - p1).norm();
            double c = (p2 - p1).norm();
            double a2 = a * a;
            double b2 = b * b;
            double c2 = c * c;
            Vector3 O{a2 * (b2 + c2 - a2), b2 * (c2 + a2 - b2), c2 * (a2 + b2 - c2)};
            // normalize to sum of 1
            O /= O[0] + O[1] + O[2];
            // change back to space
            return O[0] * p1 + O[1] * p2 + O[2] * p3;
        }

        Vector3 findCircumcenter(GeomPtr const &geometry, Face f)
        {
            // retrieve the face's vertices
            int index = 0;
            Vector3 p[3];
            for (Vertex v0 : f.adjacentVertices())
            {
                p[index] = geometry->inputVertexPositions[v0];
                index++;
            }
            return findCircumcenter(p[0], p[1], p[2]);
        }
        bool isDelaunay(GeomPtr const &geometry, Edge e)
        {
            float angle1 = geometry->cornerAngle(e.halfedge().next().next().corner());
            float angle2 = geometry->cornerAngle(e.halfedge().twin().next().next().corner());
            return angle1 + angle2 <= PI;
        }
        
        inline double diamondAngle(Vector3 a, Vector3 b, Vector3 c, Vector3 d) // dihedral angle at edge a-b
        {
            Vector3 n1 = cross(b-a, c-a);
            Vector3 n2 = cross(b-d, a-d);
            return PI-angle(n1, n2);
        }
        
        inline bool checkFoldover(Vector3 a, Vector3 b, Vector3 c, Vector3 x, double angle)
        {
            return diamondAngle(a, b, c, x) < angle;
        }
        
        bool shouldFlip(MeshPtr const &mesh, GeomPtr const &geometry, Edge e)
        {
            Halfedge he = e.halfedge();
            Vertex v1 = he.vertex();
            Vertex v2 = he.next().vertex();
            Vertex v3 = he.next().next().vertex();
            Vertex v4 = he.twin().next().next().vertex();
            
            Vector3 a = geometry->inputVertexPositions[v1];
            Vector3 b = geometry->inputVertexPositions[v2];
            Vector3 c = geometry->inputVertexPositions[v3];
            Vector3 d = geometry->inputVertexPositions[v4];
            //if(diamondAngle(d, c, a, b) < PI/2) return false;
            
            // check how close the diamond vertices are to degree 6
            int d1 = v1.degree();
            int d2 = v2.degree();
            int d3 = v3.degree();
            int d4 = v4.degree();
            
//            if(d1 == 3 || d2 == 3 || d3 == 3 || d4 == 3 || d1 - 1 <= 3 || d3 - 1 <= 3) return false;
            return degreeDifference(d1, d2, d3, d4) > degreeDifference(d1 - 1, d2 - 1, d3 + 1, d4 + 1);
        }
        
        bool shouldCollapse(MeshPtr const &mesh, GeomPtr const &geometry, Edge e)
        {
            std::vector<Halfedge> toCheck;
            Vertex v1 = e.halfedge().vertex();
            Vertex v2 = e.halfedge().twin().vertex();
            Vector3 midpoint = edgeMidpoint(mesh, geometry, e);
            // find (halfedge) link around the edge, starting with those surrounding v1
            Halfedge he = v1.halfedge();
            Halfedge st = he;
            do{
                he = he.next();
                if(he.vertex() != v2 && he.next().vertex() != v2){
                    toCheck.push_back(he);
                }
                he = he.next().twin();
            }
            while(he != st);
            // v2
            he = v2.halfedge();
            st = he;
            do{
                he = he.next();
                if(he.vertex() != v1 && he.next().vertex() != v1){
                    toCheck.push_back(he);
                }
                he = he.next().twin();
            }
            while(he != st);
            
            for(Halfedge he0 : toCheck){
                Halfedge heT = he0.twin();
                Vertex v1 = heT.vertex();
                Vertex v2 = heT.next().vertex();
                Vertex v3 = heT.next().next().vertex();
                Vector3 a = geometry->inputVertexPositions[v1];
                Vector3 b = geometry->inputVertexPositions[v2];
                Vector3 c = geometry->inputVertexPositions[v3];
                if(checkFoldover(a, b, c, midpoint, 2)){
                    // std::cout<<"prevented foldover"<<std::endl;
                    return false;
                }
            }
            return true;
        }
        
        bool shouldMove(MeshPtr const &mesh, GeomPtr const &geometry, Vertex v, Vector3 x)
        {
            std::vector<Halfedge> toCheck;
            // find (halfedge) link around v
            Halfedge he = v.halfedge();
            Halfedge st = he;
            do{
                he = he.next();
                toCheck.push_back(he);
                he = he.next().twin();
            }
            while(he != st);
            
            for(Halfedge he0 : toCheck){
                Halfedge heT = he0.twin();
                Vertex v1 = heT.vertex();
                Vertex v2 = heT.next().vertex();
                Vertex v3 = heT.next().next().vertex();
                Vector3 a = geometry->inputVertexPositions[v1];
                Vector3 b = geometry->inputVertexPositions[v2];
                Vector3 c = geometry->inputVertexPositions[v3];
                if(checkFoldover(a, b, c, x, 2)){
                    // std::cout<<"prevented foldover"<<std::endl;
                    return false;
                }
            }
            return true;
        }
        
        // debug functions
        
        void collapseEdge(MeshPtr const &mesh, GeomPtr const &geometry, Edge e)
        {
            Vertex v;
//            if(v.degree() <= 3) return;
//            for(int i = v.getIndex(); i < (int)mesh->nVertices()-1; i++)
//            {
//                geometry->inputVertexPositions[i] = geometry->inputVertexPositions[i+1];
//            }
           mesh->collapseEdgeTriangular(e);
        }
        
        void testCollapseEdge(MeshPtr const &mesh, GeomPtr const &geometry, int index)
        {
            int i = 0;
            for(Edge e : mesh->edges())
            {
        //      if(i <= 10000) std::cerr<<e.getIndex()<<std::endl;
                if((int)e.getIndex() == index){
        //          std::cerr<<e.halfedge().next()<<std::endl;
                    std::cerr<<"collapsed "<<e<<std::endl;        
                    collapseEdge(mesh, geometry, e);
                    break;
                }
                i++;
            }
//            std::cerr<<"Here1!"<<std::endl;
            mesh->validateConnectivity();
//            std::cerr<<"Here2!"<<std::endl;
            mesh->compress();
//            std::cerr<<"Here3!"<<std::endl;
            mesh->validateConnectivity();
 //           std::cerr<<"Here4!"<<std::endl;
        }
        
        bool checkDegree(MeshPtr const &mesh){
            for(Vertex v : mesh->vertices()){
                if(v.degree() == 2 && !v.isBoundary()){
                    std::cerr<<"broken: "<<v<<std::endl;
                    return false;
                }
            }
            return true;
        }
        
        void testStuff(MeshPtr const &mesh, GeomPtr const &geometry, int index)
        {
            
            /*for(Vertex v : mesh->vertices())
            {
                if((int)v.getIndex() == index){
                    mesh->removeVertex(v);
                    break;
                }
            }*/
            for(Edge e : mesh->edges()){
                if((int)e.getIndex() == index){
                    mesh->flip(e);
                    break;
                }
            }
            /*std::vector<int> toRemove{888,886,881,879,877,876,873,868,866,864,863,862,861,860,859,857,856,854,853,851,850,849,848,847,846,842,841,840,838,836};
            reverse(std::begin(toRemove), std::end(toRemove));
            
            while(!toRemove.empty()){
                int ind = toRemove.back();
                toRemove.pop_back();
                for(Edge e : mesh->edges()){
                    if((int)e.getIndex() == ind){
                        std::cerr<<e<<std::endl;
                        mesh->myCollapseEdgeTriangular(e);
                    }
                }
            }*/
            
            std::cerr<<"Here1!"<<std::endl;
            mesh->validateConnectivity();
            std::cerr<<"Here2!"<<std::endl;
            mesh->compress();
            std::cerr<<"Here3!"<<std::endl;
            
        }
        
        void testStuff2(MeshPtr const &mesh, GeomPtr const &geometry, GeomPtr const & geometryOriginal)
        {
            int i = 0;
            /*geometry->requireVertexAngleSums();
            geometry->requireVertexGaussianCurvatures();
            for(Vertex v : mesh->vertices())
            {
                //std::cerr<<geometry->vertexNormals[v]<<"  "<<vertexNormal(geometry, v)<<std::endl;
                //std::cerr<<geometry->vertexDualAreas[v]<<"  "<<vertexDualArea(geometry, v)<<std::endl;
                //std::cerr<<geometry->vertexAngleSums[v]<<"  "<<vertexAngleSum(geometry, v)<<std::endl;
                //std::cerr<<geometry->vertexGaussianCurvatures[v]<<"  "<<vertexGaussianCurvature(geometry, v)<<std::endl;
                i++;
            }*/
            
            for(Face f : mesh->faces())
            {
                std::cerr<<f.getIndex()<<": "<<geometry->faceArea(f)<<"/"<<geometryOriginal->faceArea(f)<<std::endl;
                i++;
            }
        }
        
        void showEdge(MeshPtr const &mesh, GeomPtr const &geometry, int index)
        {
            int i = 0;
            /*for(Edge e : mesh->edges())
            {
                if((int)e.getIndex() == index){
                    geometry->inputVertexPositions[e.halfedge().vertex()] *= 1.1;
                    geometry->inputVertexPositions[e.halfedge().twin().vertex()] *= 1.1;
                    break;
                }
                i++;
            }*/
            for(Vertex v : mesh->vertices())
            {
                if((int)v.getIndex() == index){
                    geometry->inputVertexPositions[v] *= 1.1;
                    break;
                }
                i++;
            }
        }
        
        
        void testVertex(MeshPtr const &mesh, GeomPtr const &geometry, int index)
        {
            for(Vertex v : mesh->vertices()){
                if((int)v.getIndex() == index){
                    std::cerr<<geometry->vertexNormals[v]<<std::endl;
                    break;
                }
            }
            std::cerr<<"Here1!"<<std::endl;
            mesh->validateConnectivity();
            std::cerr<<"Here2!"<<std::endl;
            mesh->compress();
            std::cerr<<"Here3!"<<std::endl;
            
        }
        void testEdge(MeshPtr const &mesh, GeomPtr const &geometry, int index)
        {
            for(Edge e : mesh->edges()){
                if((int)e.getIndex() == index){
                    Vector3 newPos = edgeMidpoint(mesh, geometry, e);
                    Halfedge he = mesh->splitEdgeTriangular(e);
                    Vertex newV = he.vertex();
                    geometry->inputVertexPositions[newV] = newPos;
                    break;
                }
            }
            std::cerr<<"Here1!"<<std::endl;
            mesh->validateConnectivity();
            std::cerr<<"Here2!"<<std::endl;
            mesh->compress();
            std::cerr<<"Here3!"<<std::endl;
            
        }
        void testFace(MeshPtr const &mesh, GeomPtr const &geometry, int index)
        {
            for(Face f : mesh->faces()){
                if((int)f.getIndex() == index){
                    //std::cerr<<geometry->faceAreas[f];
                    Halfedge he = f.halfedge();
                    geometry->inputVertexPositions[he.vertex()] *= 2;
                    geometry->inputVertexPositions[he.next().vertex()] *= 2;
                    geometry->inputVertexPositions[he.next().next().vertex()] *= 2;
                    break;
                }
            }

            std::cerr<<"Here1!"<<std::endl;
            mesh->validateConnectivity();
            std::cerr<<"Here2!"<<std::endl;
            mesh->compress();
            std::cerr<<"Here3!"<<std::endl;
            
        }
        
        // non-debug functions from here
        
        void adjustVertexDegrees(MeshPtr const &mesh, GeomPtr const &geometry)
        {
            for(Edge e : mesh->edges())
            {
                if(!e.isBoundary() && shouldFlip(mesh, geometry, e))
                {
                    //checkDegree(mesh);
                    mesh->flip(e);
                }
            }
        }

        void fixDelaunay(MeshPtr const &mesh, GeomPtr const &geometry)
        {
            // queue of edges to check if Delaunay
            queue<Edge> toCheck;
            // true if edge is currently in toCheck
            EdgeData<bool> inQueue(*mesh);
            // start with all edges
            for (Edge e : mesh->edges())
            {
                toCheck.push(e);
                inQueue[e] = true;
            }
            // counter and limit for number of flips
            int flipMax = 100 * mesh->nVertices();
            int flipCnt = 0;
            while (!toCheck.empty() && flipCnt < flipMax)
            {
                Edge e = toCheck.front();
                toCheck.pop();
                inQueue[e] = false;
                // if not Delaunay, flip edge and enqueue the surrounding "diamond" edges (if not already)
                if (!e.isBoundary() && !isDelaunay(geometry, e))
                {
                    flipCnt++;
                    Halfedge he = e.halfedge();
                    Halfedge he1 = he.next();
                    Halfedge he2 = he1.next();
                    Halfedge he3 = he.twin().next();
                    Halfedge he4 = he3.next();

                    if (!inQueue[he1.edge()])
                    {
                        toCheck.push(he1.edge());
                        inQueue[he1.edge()] = true;
                    }
                    if (!inQueue[he2.edge()])
                    {
                        toCheck.push(he2.edge());
                        inQueue[he2.edge()] = true;
                    }
                    if (!inQueue[he3.edge()])
                    {
                        toCheck.push(he3.edge());
                        inQueue[he3.edge()] = true;
                    }
                    if (!inQueue[he4.edge()])
                    {
                        toCheck.push(he4.edge());
                        inQueue[he4.edge()] = true;
                    }
                    mesh->flip(e);
                }
            }
        }

        void smoothByLaplacian(MeshPtr const &mesh, GeomPtr const &geometry)
        {
            // smoothed vertex positions
            VertexData<Vector3> newVertexPosition(*mesh);
            for (Vertex v : mesh->vertices())
            {
                if(v.isBoundary())
                {
                    newVertexPosition[v] = geometry->inputVertexPositions[v];
                }
                else
                {
                    // calculate average of surrounding vertices
                    newVertexPosition[v] = Vector3::zero();
                    for (Vertex j : v.adjacentVertices())
                    {
                        newVertexPosition[v] += geometry->inputVertexPositions[j];
                    }
                    newVertexPosition[v] /= v.degree();
                    // and project the average to the tangent plane
                    Vector3 updateDirection = newVertexPosition[v] - geometry->inputVertexPositions[v];
                    updateDirection = projectToPlane(updateDirection, vertexNormal(geometry, v));
                    
                    newVertexPosition[v] = geometry->inputVertexPositions[v] + 1*updateDirection;
                }
            }
            // update final vertices
            for (Vertex v : mesh->vertices())
            {
                geometry->inputVertexPositions[v] = newVertexPosition[v];
            }
        }
        
        double getFaceSmoothGaussianCurvature(GeomPtr const &geometry, Face f)
        {
            double A = geometry->faceArea(f);
            double S = faceGaussianCurvature(geometry, f); 
            double K = S / A;
            return K;
        }
        
        double getFaceSmoothMeanCurvature(GeomPtr const &geometry, Face f)
        {
            double A = geometry->faceArea(f);
            double S = faceMeanCurvature(geometry, f); 
            double K = S / A;
            return K;
        }
        
        double findFaceTargetL(MeshPtr const &mesh, GeomPtr const &geometry, Face f, double flatLength, double epsilon)
        {
            double K = getFaceSmoothGaussianCurvature(geometry, f);
            double L = flatLength * epsilon / (sqrt(fabs(K)) + epsilon);
            return L;
            // return flatLength;
        }

        void smoothByCircumcenter(MeshPtr const &mesh, GeomPtr const &geometry)
        {
            geometry->requireFaceAreas();
            // smoothed vertex positions
            VertexData<Vector3> newVertexPosition(*mesh);
            for (Vertex v : mesh->vertices())
            {
                newVertexPosition[v] = geometry->inputVertexPositions[v]; // default
                if(!v.isBoundary())
                {
                    Vector3 updateDirection = Vector3::zero();
                    //double totalD = 0;
                    for (Face f : v.adjacentFaces())
                    {
                        // add the circumcenter weighted by face area to the update direction
                        Vector3 circum = findCircumcenter(geometry, f);
                        //double D = 1/findFaceTargetL(mesh, geometry, f, 1, 0.1);
                        //D = D*D;
                        updateDirection += geometry->faceArea(f) * (circum - geometry->inputVertexPositions[v]);
                        //totalD += geometry->faceArea(f) * D;
                    }
                    //std::cerr<<updateDirection<<std::endl;
                    updateDirection /= (3 * geometry->vertexDualArea(v));
                    //updateDirection /= totalD;
                    //std::cerr<<"  "<<updateDirection<<std::endl;
                    // project update direction to tangent plane
                    updateDirection = projectToPlane(updateDirection, vertexNormal(geometry, v));
                    Vector3 newPos = geometry->inputVertexPositions[v] + .5 * updateDirection;
                    newVertexPosition[v] = newPos;
                }
            }
            // update final vertices
            for (Vertex v : mesh->vertices())
            {
                geometry->inputVertexPositions[v] = newVertexPosition[v];
            }
        }

        
        double getSmoothGaussianCurvature(GeomPtr const &geometry, Vertex v)
        {
            double A = geometry->vertexDualArea(v);
            double S = vertexGaussianCurvature(geometry, v);
            double K = S / A;
            return K;
        }
        
        double getSmoothMeanCurvature(GeomPtr const &geometry, Vertex v)
        {
            double A = geometry->vertexDualArea(v);
            double S = geometry->vertexMeanCurvature(v);
            double K = S / A;
            return K;
        }
        
        // flatLength: specifies how long the target edge length should be in flat regions
        // epsilon: controls how much variation in target length occurs due to curvature
        double findGaussianTargetL(MeshPtr const &mesh, GeomPtr const &geometry, Edge e, double flatLength, double epsilon)
        {
            // Areas and curvatures are already required in main.cpp
            double averageK = 0;
            for (Vertex v : e.adjacentVertices()) {
                averageK += getSmoothGaussianCurvature(geometry, v);
            }
            averageK /= 2;
            double L = flatLength * epsilon / (fabs(sqrt(averageK)) + epsilon);
            return L;
            // return flatLength;
        }
        
        double findMeanTargetL(MeshPtr const &mesh, GeomPtr const &geometry, Edge e, double flatLength, double epsilon)
        {
            // Areas and curvatures are already required in main.cpp
            double averageH = 0;
            for (Vertex v : e.adjacentVertices()) {
                averageH += getSmoothMeanCurvature(geometry, v);
            }
            averageH /= 2;
            double L = flatLength * epsilon / (fabs(averageH) + epsilon);
            return L;
            // return flatLength;
        }
        
        void adjustEdgeLengths(MeshPtr const &mesh, GeomPtr const &geometry, GeomPtr const &geometryOriginal, double flatLength, double epsilon, double minLength, bool curvatureAdaptive)
        {
        	// sanity checks for geometryOriginal
        	
        	/*for(Vertex v : mesh->vertices())
            {
                std::cerr<<v.getIndex()<<": "<<geometry->inputVertexPositions[v]<<"/"<<geometryOriginal->inputVertexPositions[v]<<std::endl;
            }*/
        	
            // queues of edges to CHECK to change
            std::vector<Edge> toSplit;
            std::vector<Edge> toCollapse;
            
            for(Edge e : mesh->edges())
            {
                toSplit.push_back(e);
            }
            
            // actually do it
            std::cerr<<"spliting..."<<std::endl;
            while(!toSplit.empty())
            {
                Edge e = toSplit.back();
                toSplit.pop_back();
                double length_e = geometry->edgeLength(e);
                double threshold = (curvatureAdaptive) ? findMeanTargetL(mesh, geometry, e, flatLength, epsilon) : flatLength;
                if(length_e > minLength && length_e > threshold * 1.5)
                {
                    Vector3 newPos = edgeMidpoint(mesh, geometry, e);
                    Vector3 newPosOrig = edgeMidpoint(mesh, geometryOriginal, e);
                    Halfedge he = mesh->splitEdgeTriangular(e);
                    Vertex newV = he.vertex();
                    geometry->inputVertexPositions[newV] = newPos;
                    geometryOriginal->inputVertexPositions[newV] = newPosOrig;
                }
                else
                {
                    toCollapse.push_back(e);
                }                
                
            }
            std::cerr<<"collapsing..."<<std::endl;
            while(!toCollapse.empty())
            {
                Edge e = toCollapse.back();
                toCollapse.pop_back();
                if(e.halfedge().next().getIndex() != INVALID_IND) // make sure it exists
                {
                    double threshold = (curvatureAdaptive) ? findMeanTargetL(mesh, geometry, e, flatLength, epsilon) : flatLength;
                    if(geometry->edgeLength(e) < threshold * 0.5)
                    {
                        Vector3 newPos = edgeMidpoint(mesh, geometry, e);
                        Vector3 newPosOrig = edgeMidpoint(mesh, geometryOriginal, e);
                        if(shouldCollapse(mesh, geometry, e)){
                        	Vertex v = mesh->collapseEdgeTriangular(e);
                            if (v != Vertex()) {
                                if(!v.isBoundary()) {
                                    geometry->inputVertexPositions[v] = newPos;
                                    geometryOriginal->inputVertexPositions[v] = newPosOrig;
                                }
                            }
                        }
                    }
                }
            }
            
            mesh->validateConnectivity();
            //std::cerr<<"hereeee"<<std::endl;
            mesh->compress();
            geometry->refreshQuantities();
        }

        void remesh(MeshPtr const &mesh, GeomPtr const &geometry, GeomPtr const &geometryOriginal){
            for(int i = 0; i < 1; i++){
                std::cout<<"fixing edges"<<std::endl;
                adjustEdgeLengths(mesh, geometry, geometryOriginal, 1, 0.1, 0.05); 
                std::cout<<"flipping"<<std::endl;
                fixDelaunay(mesh, geometry);
                for(int j = 0; j < 10; j++){
                    std::cout<<"smoothing"<<std::endl;
                    smoothByCircumcenter(mesh, geometry);
                    std::cout<<"flipping"<<std::endl;
                    fixDelaunay(mesh, geometry);
                }
                std::cout<<"done"<<std::endl;
            }
            mesh->validateConnectivity();
        }
        
        // not used right now
        
        Vector3 findBarycenter(Vector3 p1, Vector3 p2, Vector3 p3)
        {
            return (p1 + p2 + p3)/3;
        }

        Vector3 findBarycenter(GeomPtr const &geometry, Face f)
        {
            // retrieve the face's vertices
            int index = 0;
            Vector3 p[3];
            for (Vertex v0 : f.adjacentVertices())
            {
                p[index] = geometry->inputVertexPositions[v0];
                index++;
            }
            return findBarycenter(p[0], p[1], p[2]);
        }
        
        void smoothByFaceWeight(MeshPtr const &mesh, GeomPtr const &geometry, FaceData<double> faceWeight)
        {
            // smoothed vertex positions
            VertexData<Vector3> newVertexPosition(*mesh);
            for (Vertex v : mesh->vertices())
            {
                if(v.isBoundary())
                {
                    newVertexPosition[v] = geometry->inputVertexPositions[v];
                }
                else
                {
                    Vector3 newV = Vector3::zero();
                    Vector3 updateDirection = Vector3::zero();
                    double s = 0;
                    for (Face f : v.adjacentFaces())
                    {
                        // add the barycenter weighted by face area
                        Vector3 bary = findBarycenter(geometry, f);
                        double w = geometry->faceArea(f) / faceWeight[f];
                        newV += w * bary;
                        s += w;
                    }
                    newV /= s;
                    
                    // project update direction to tangent plane
                    updateDirection = newV - geometry->inputVertexPositions[v];
                    updateDirection = projectToPlane(updateDirection, vertexNormal(geometry, v));
                    newVertexPosition[v] = geometry->inputVertexPositions[v] + 0.1*updateDirection;
                }
            }
            // update final vertices
            for (Vertex v : mesh->vertices())
            {
                geometry->inputVertexPositions[v] = newVertexPosition[v];
            }
        }
    } // namespace remeshing
} // namespace rsurfaces
