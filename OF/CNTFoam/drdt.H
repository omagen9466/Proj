#include "fvCFD.H"
#include "volFields.H"
#include <cmath>
#include "dimensionSet.H"
#include "dimensionedScalar.H"
// #include "wallFvPatch.H"
#include "fvMesh.H"
#include "fvBoundaryMesh.H"
#include "fvPatchList.H"
// #include "IEqn.H"

// volScalarField calculateI(const volScalarField& T)
{

    const Foam::volScalarField& T = mesh.lookupObject<Foam::volScalarField>("T");
    std::cout << "\ntest1\n" << endl;
    volScalarField r_t=((3*M)/(4*M_PI*N));
    // r_t.write();
    
    r_het=pow(r_t,scalar(1.0/3.0));
    std::cout << "\ntest2\n" << endl;
    drdt=(k_copp/(r_het*(1+3.18*0.01)))*((atm_T-T)/(rhol_kg*dhkg));       
    std::cout << "\ntest3\n" << endl;        
         
           
            
    

    
}
