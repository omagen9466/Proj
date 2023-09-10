#include "IEqn.H"
{
    
    fvScalarMatrix NEqn
    (
        fvm::ddt(rhol_kg, N) 
      + fvm::div(rhol_kg*(phi/fvc::interpolate(rho)), N)
    //  - fvm::laplacian(turbulence->muEff(), N)
     ==
        fvc::Sp(rhol_kg*M,I)
      //+ fvOptions(rho, N)
    );
    
    NEqn.relax();

    fvOptions.constrain(NEqn);
    // std::cout << "\nProb\n" << endl;
    NEqn.solve();
    // std::cout << "\nlem\n" << endl;
    fvOptions.correct(N);
}
