// #include "Sd.H"
{
    volScalarField& he = thermo.he();
    // volScalarField m_crit=(rhol_kg*M_PI*pow(r_het,3)/3)*(2-3*std::cos(90)+pow(std::cos(90),3));
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
        // fvOptions(rho, he)
        -fvc::Sp(hp,Sd)
    );

    EEqn.relax();

    fvOptions.constrain(EEqn);
    // std::cout << "\nsolving e\n" << endl;
    EEqn.solve();
    // std::cout << "\nfinished e\n" << endl;
    fvOptions.correct(he);
    // std::cout << "\nfinished correcting\n" << endl;  
    // he.write();
    thermo.correct();
    // std::cout << "\nfinished correcting2\n" << endl;  
}
