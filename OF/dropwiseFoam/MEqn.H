
// #include "IEqn.H"
{
    // volScalarField m_crit=(rhol_kg*M_PI*pow(r_min,3)/3)*(2-3*std::cos(theta.value()*M_PI/180)+std::pow(std::cos(theta.value()*M_PI/180),3));

    
    fvScalarMatrix MEqn
    (
        fvm::ddt(rho, M) 
      + fvm::div(phi, M)
//      - fvm::laplacian(turbulence->muEff(), M)
     ==
        // -fvc::Sp(m_crit* M,I)
        -fvc::Sp(dummy,Sd)
      //+ fvOptions(rho, M)
    );
    
    MEqn.relax();

    fvOptions.constrain(MEqn);

    MEqn.solve();

    fvOptions.correct(M);
}
