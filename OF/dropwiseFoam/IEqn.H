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
    const Foam::volScalarField& T = mesh.lookupObject<Foam::volScalarField>("T");


    dimensionedScalar Dim
    (
        "Dim",
        dimensionSet(0, 0, -1, 0, 0,0,0),
        scalar(1)
    );


    label faceId(-1);
    label patchId(-1);
    label nWallFaces(0);
    vector wallNormal(Zero);
    // std::cout << "\nstarting I\n" << endl;
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
                    dimensionedScalar T_wall
                    (
                        "T_wall",
                        dimensionSet(0, 0, 0, 1, 0,0,0),
                        scalar(T.boundaryField()[patchId][faceId])
                    );
                    // Info<< "T_wall = " << T_wall << nl << endl;
                    r_min[cellId] =(( 2.0*sigma *T_sat)/(dhkg*(T_sat-T_wall)*rhol_kg)).value();
                    // std::cout << "\nTest4.5\n" << endl;
                    // N_s[cellId]=0.037/pow(r_min[cellId],2);
                    N_s[cellId]=1e15;
                    I[cellId] = 1e-4*N_s[cellId]*mag(mesh.Sf().boundaryField()[patchId][faceId])/mesh.V()[cellId];
                    r_eq[cellId]=1/(std::pow(4*(0.037/std::pow(r_min[cellId],2)),scalar(1.0/2.0)));
                    // r_max[cellId]=0.4*std::pow((sigma.value()/(rhol_kg.value()*9.81)),scalar(1.0/2.0));
                    r_max[cellId]=std::pow(((6*std::pow(std::sin(theta.value()*M_PI/180),2))/((2-3*std::cos(theta.value()*M_PI/180)+std::pow(std::cos(theta.value()*M_PI/180),3))))*((sigma.value())/(958*9.81)),scalar(1.0/2.0));
                    // std::cout << "\nTest5\n" << endl;
                
                
            }
            // std::cout << "\nFINISHED PATCH\n" << endl;
            
        }
        
    }

    
    
    
    // std::cout << "\nFINISHED I\n" << endl;
}
