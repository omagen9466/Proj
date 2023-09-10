#include "fvCFD.H"
#include "volFields.H"
#include <cmath>
#include "dimensionSet.H"
#include "dimensionedScalar.H"
#include "wallFvPatch.H"
#include "fvMesh.H"
#include "fvBoundaryMesh.H"
#include "fvPatchList.H"
// #include "IEqn.H"

// volScalarField calculateI(const volScalarField& T)
{


    const Foam::volScalarField& T = mesh.lookupObject<Foam::volScalarField>("T");
    
    label faceId(-1);
    label patchId(-1);
    label nWallFaces(0);
    vector wallNormal(Zero);
    // std::cout << "\nHERE?\n" << endl;
    // volScalarField r_t=(3*M)/(4*M_PI*N);
    
    // r_het=Foam::pow(r_t,scalar(1.0/3.0));
    
    std::cout << "\nstarting r_het\n" << endl;
    const fvPatchList& patches = mesh.boundary();
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
                
                // std::cout << "\nTest3\n" << endl;
                
                    // std::cout << "\nTest4\n" << endl;
                    wallNormal = -nf[facei];
                    faceId = facei;
                    patchId = patchi;
                    label cellId = patches[patchId].faceCells()[faceId];
                    r_het[cellId]=std::pow((3*M[cellId]*rho[cellId])/(rhol_kg.value()*M_PI*N[cellId]*(2-3*std::cos(theta.value()*M_PI/180)+std::pow(std::cos(theta.value()*M_PI/180),3.0))),scalar(1.0/3.0));
                    
                    std::cout << "\ncalculating drdt2\n" << endl;
                    h_int[cellId]=((2*0.02)/(2-0.02))*(std::pow(dhkg.value(),2)/(T_sat.value()*(spec_vol.value())))*pow((mol_mass.value()/(2*M_PI*R_.value()*T_sat.value())),0.5f);
                    // h_int[cellId]=15.7e6;
                     dimensionedScalar T_wall
                    (
                        "T_wall",
                        dimensionSet(0, 0, 0, 1, 0,0,0),
                        scalar(T.boundaryField()[patchId][faceId])
                    );
                    std::cout << "\ncalculating drdt\n" << endl;
                    // Info<< mesh.V()[cellId] << endl;
                    // std::cout << "\ncalculating drdt\n" << endl;
                    drdt[cellId]=(4*(T_sat.value()-T_wall.value())/(rhol_kg.value()*dhkg.value()))*((1-std::cos(theta.value()*M_PI/180))/(2-3*std::cos(theta.value()*M_PI/180)+pow(std::cos(theta.value()*M_PI/180),3)))*(1-(r_min[cellId]/r_het[cellId]))/((2/h_int[cellId])+((r_het[cellId]*(1-std::cos(theta.value()*M_PI/180)))/k_copp.value()));
                    std::cout << "\ncalculating drdt4\n" << endl;
                    // drdt[cellId]=((T_sat.value()-T_wall.value())/(rhol_kg.value()*dhkg.value()*(2-3*std::cos(theta.value()*M_PI/180)+pow(std::cos(theta.value()*M_PI/180),3))))*(1-(r_min[cellId]/r_het[cellId]))/(((theta.value()*M_PI/180)/(4*k_copp.value()*std::sin(theta.value()*M_PI/180)))*r_het[cellId]+(1/(2*h_int[cellId]*(1-std::cos(theta.value()*M_PI/180)))));
                    qd[cellId]=M_PI*std::pow(r_het[cellId],2.0)*rhol_kg.value()*dhkg.value()*drdt[cellId]*(2-3*std::cos(theta.value()*M_PI/180)+std::pow(std::cos(theta.value()*M_PI/180),3))*1e-3;
                    std::cout << "\ncalculating drdt5\n" << endl;
                    heat_flux[cellId]=(dhkg.value()*rhol_kg.value()*(2-3*std::cos(theta.value()*M_PI/180)+std::pow(std::cos(theta.value()*M_PI/180),3.0))/(std::pow(r_max[cellId],scalar(1.0/3.0))))*drdt[cellId]*(std::pow(r_max[cellId],scalar(1.0/3.0))-std::pow(r_min[cellId],scalar(1.0/3.0)));
                    // N_r[cellId]=(1/(3*M_PI*r_max[cellId]*std::pow(r_het[cellId],2)))*std::pow((r_het[cellId]/r_max[cellId]),scalar(-2.0/3));
                    // delta_r[cellId]=std::pow(std::pow(r_het[cellId]-r_dummy[cellId],2),0.5f);
                    // r_dummy[cellId]=r_het[cellId];
                    
                    // if (r_het[cellId] > r_eq[cellId])
                    // {
                    // heat_vol[cellId]=qd[cellId]*N_r[cellId];
                    // heat_flux[cellId]=heat_vol[cellId]*delta_r[cellId]+heat_flux[cellId];
                    // }
                    
                    // if (r_het[cellId] > r_max[cellId])
                    // {
                    //     r_max.write();
                    //     r_het.write();
                    // FatalError
                    // << "r_max reached" << nl
                    // << exit(FatalError);
                    // }
            }
            // std::cout << "\nFINISHED PATCH\n" << endl;
            
        }
        
    }

    

    
}
