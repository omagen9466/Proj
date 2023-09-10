#include "fvCFD.H"
#include "volFields.H"
#include <cmath>
#include "dimensionSet.H"
#include "dimensionedScalar.H"
#include "wallFvPatch.H"
#include "fvMesh.H"
#include "fvBoundaryMesh.H"
#include "fvPatchList.H"
#include "drdt.H"
// #include "IEqn.H"

// volScalarField calculateI(const volScalarField& T)
{

    const Foam::volScalarField& T = mesh.lookupObject<Foam::volScalarField>("T");

    
    label faceId(-1);
    label patchId(-1);
    label nWallFaces(0);
    vector wallNormal(Zero);
    std::cout << "\nstarting I\n" << endl;
    const fvPatchList& patches = mesh.boundary();

    forAll(patches, patchi)
    {
        const fvPatch& currPatch = patches[patchi];
        // std::cout << "\nTest1\n" << endl;
        if (isA<wallFvPatch>(currPatch))
        {
            // std::cout << "\nIS WALL\n" << endl;
            const vectorField nf(currPatch.nf());
            // std::cout << "\nTest2\n" << endl;
            forAll(nf, facei)
            {
                
                // std::cout << "\nTest3\n" << endl;
                
                
                    
                    wallNormal = -nf[facei];
                    faceId = facei;
                    patchId = patchi;
                    label cellId = patches[patchId].faceCells()[faceId];
                    
                    hp[cellId]=3646+(T[cellId]-274)*(419170-3646)/(373-274);
                    Sd[cellId]=N[cellId]*rhol_kg.value()*M_PI*std::pow(r_het[cellId],2.0)*(2-3*std::cos(theta.value()*M_PI/180)+pow(std::cos(theta.value()*M_PI/180),3))*drdt[cellId];
                    // Sd[cellId]=(6*M[cellId]*(1-std::cos(theta.value()*M_PI/180))*drdt[cellId]*rho[cellId]*rhol_kg.value())/(r_het[cellId]*(2-3*std::cos(theta.value()*M_PI/180)+pow(std::cos(theta.value()*M_PI/180),3)));
            }
            // std::cout << "\nFINISHED PATCH\n" << endl;
            
        }
        
    }
        
}
