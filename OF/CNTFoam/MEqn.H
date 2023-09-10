{
    dimensionedScalar dummy
    (
        "dummy",
        dimensionSet(0, 0, 0, 0, 0, 0, 0),
        scalar(1)
    );
    fvScalarMatrix MEqn
    (
        fvm::ddt(rho, M) 
      + fvm::div(phi, M)
//      - fvm::laplacian(turbulence->muEff(), M)
     ==
        -fvc::Sp(crit_mass* M,I)
        -fvc::Sp(dummy,Sd)
    );
    
    MEqn.relax();

    fvOptions.constrain(MEqn);

    MEqn.solve();

    fvOptions.correct(M);
}
