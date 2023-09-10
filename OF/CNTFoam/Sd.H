#include "fvCFD.H"
#include "volFields.H"
#include <cmath>
#include "dimensionSet.H"
#include "dimensionedScalar.H"
// #include "wallFvPatch.H"
#include "fvMesh.H"
#include "fvBoundaryMesh.H"
#include "fvPatchList.H"
#include "drdt.H"
// #include "IEqn.H"

// volScalarField calculateI(const volScalarField& T)
{

    const Foam::volScalarField& rho = mesh.lookupObject<Foam::volScalarField>("rho");

    

                    
    
    Sd=(3*M*rho*drdt)/(r_het);
       
        
}
