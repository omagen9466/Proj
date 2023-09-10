{
    // const Foam::volScalarField& T = mesh.lookupObject<Foam::volScalarField>("T");
    // const Foam::volScalarField& p = mesh.lookupObject<Foam::volScalarField>("p");
    // std::cout << "M solver" << endl;
    // I = calculateI(T,p);
    dimensionedScalar crit_mass
    (
        "crit_mass",
        dimensionSet(1, 0, 0, 0, 0, 0, 0),
        scalar(4e-15)
    );
    
    fvScalarMatrix MEqn
    (
        fvm::ddt(rho, M) 
      + fvm::div(phi, M)
//      - fvm::laplacian(turbulence->muEff(), M)
     ==
        -fvc::Sp(crit_mass* M,I)
      //+ fvOptions(rho, M)
    );
    
    MEqn.relax();

    fvOptions.constrain(MEqn);

    MEqn.solve();

    fvOptions.correct(M);
}
