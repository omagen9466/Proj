#include "fvCFD.H"
#include "volFields.H"
#include <cmath>
#include "dimensionSet.H"
#include "dimensionedScalar.H"
#include "wallFvPatch.H"
#include "fvMesh.H"
#include "fvBoundaryMesh.H"
#include "fvPatchList.H"

// volScalarField calculateI(const volScalarField& T)
{
    
    label faceId(-1);
    label patchId(-1);
    label nWallFaces(0);
    vector wallNormal(Zero);
    std::cout << "\nstarting I\n" << endl;
    const fvPatchList& patches = mesh.boundary();
    // volScalarField r_min = T*Dim;
    // volScalarField N_s = T*Dim2;
    // volScalarField I =T*Dim3;
    forAll(patches, patchi)
    {
        const fvPatch& currPatch = patches[patchi];
        // std::cout << "\nTest1\n" << endl;
        if (isA<wallFvPatch>(currPatch))
        {
            std::cout << "\nIS WALL\n" << endl;
            const vectorField nf(currPatch.nf());
            // std::cout << "\nTest2\n" << endl;
            forAll(nf, facei)
            {
                
                
                    // std::cout << "\nTest4\n" << endl;
                    wallNormal = -nf[facei];
                    faceId = facei;
                    patchId = patchi;
                    label cellId = patches[patchId].faceCells()[faceId];

                    I[cellId] = 0;
                    // std::cout << "\nTest5\n" << endl;
                
                
            }
            std::cout << "\nFINISHED PATCH\n" << endl;
            
        }
        
    }

    
    
    
    std::cout << "\nFINISHED I\n" << endl;
}
