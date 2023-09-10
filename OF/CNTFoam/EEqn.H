{
    volScalarField& he = thermo.he();

    fvScalarMatrix EEqn
    (
        fvm::ddt(rho*M, he) + fvm::div(phi*fvc::interpolate(M),he)
      + fvc::ddt(rho*M, K) + fvc::div(phi*fvc::interpolate(M), K)
      + (
            he.name() == "e"
          ? fvc::div
            (
                fvc::absolute(phi/fvc::interpolate(rho), U),
                p,
                "div(phiv,p)"
            )
          : -M*dpdt
        )
      + thermophysicalTransport->divq(he)
     ==
        fvOptions(rho, he)
        -fvc::Sp(hp,Sd)
    );

    EEqn.relax();

    fvOptions.constrain(EEqn);

    EEqn.solve();

    fvOptions.correct(he);

    thermo.correct();
}
