{
    const Foam::volScalarField& T = mesh.lookupObject<Foam::volScalarField>("T");
    const Foam::volScalarField& p = mesh.lookupObject<Foam::volScalarField>("p");
    std::cout << "N solver" << endl;
    I = calculateI(T,p);
    dimensionedScalar rho_l
    (
        "rho_l",
        dimensionSet(1, -3, 0, 0, 0, 0, 0),
        scalar(958.77)
    );
    fvScalarMatrix NEqn
    (
        fvm::ddt(rho_l, N) 
      + fvm::div(rho_l*(phi/fvc::interpolate(rho)), N)
    //  - fvm::laplacian(turbulence->muEff(), N)
     ==
        fvc::Sp(rho_l*M,I)
      //+ fvOptions(rho, N)
    );
    
    NEqn.relax();

    fvOptions.constrain(NEqn);
    std::cout << "\nProb\n" << endl;
    NEqn.solve();
    std::cout << "\nlem\n" << endl;
    fvOptions.correct(N);
}
