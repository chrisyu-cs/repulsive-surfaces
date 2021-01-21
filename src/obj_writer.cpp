#include "obj_writer.h"
#include <iostream>
#include <fstream>

namespace rsurfaces
{

    void writeMeshToOBJ(MeshPtr mesh, GeomPtr geom, GeomPtr geomOrig, bool writeAreaRatios, std::string output)
    {
        using namespace std;
        ofstream outfile;
        outfile.open(output);

        VertexIndices inds = mesh->getVertexIndices();

        // Write all vertices in order
        for (size_t i = 0; i < mesh->nVertices(); i++)
        {
            GCVertex vert = mesh->vertex(i);
            Vector3 pos = geom->inputVertexPositions[vert];
            outfile << "v " << pos.x << " " << pos.y << " " << pos.z << endl;
        }
        
        // Write area ratios to uvs
        if (writeAreaRatios)
        {
		    for (size_t i = 0; i < mesh->nVertices(); i++)
		    {
		        GCVertex vert = mesh->vertex(i);
		        double r = geomOrig->vertexDualArea(vert) / geom->vertexDualArea(vert);
		        outfile << "vt " << r << " " << 0 << endl;
		    }
        }

        // Write all face indices
        for (GCFace face : mesh->faces())
        {
            outfile << "f ";
            for (GCVertex adjVert : face.adjacentVertices())
            {
                // OBJ is 1-indexed
                int vertInd = inds[adjVert] + 1;
                outfile << vertInd;
                if(writeAreaRatios)
                {
                	outfile << "/" << vertInd;
                }
                outfile << " ";
            }
            outfile << endl;
        }
        
        outfile << endl;
        outfile.close();
    }

} // namespace rsurfaces
