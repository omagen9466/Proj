#include "fvCFD.H"
#include "volFields.H"
#include <cmath>
#include "dimensionSet.H"
#include "dimensionedScalar.H"


volScalarField calculateI(const volScalarField& T,const volScalarField& p)
{
    //interface thickness assumed constant and value taken from granasy 95
    dimensionedScalar d
    (
        "d",
        dimensionSet(0, 1, 0, 0, 0,0,0),
        scalar(1e-10)
    );
    //boltzmann constant J/KN
    dimensionedScalar k
    (
        "k",
        dimensionSet(1, 2, -2, -1, -1,0,0),
        scalar(8.314462618)
    );
    //Boltzmann constant J/K
    dimensionedScalar kb
    (
        "kb",
        dimensionSet(1, 2, -2, -1, 0,0,0),
        scalar(1.380649e-23)
    );
    
    //molar density calculated as rho/M, M=18g/mol
    dimensionedScalar mol_mass
    (
        "mol_mass",
        dimensionSet(1, 0, 0, 0, -1, 0, 0),
        scalar(0.018)
    );

    //Normanl density
    dimensionedScalar rhol_kg
    (
        "rhol_kg",
        dimensionSet(1, -3, 0, 0, 0, 0, 0),
        scalar(958.77)
    );
    dimensionedScalar rhov_kg
    (
        "rhov_kg",
        dimensionSet(1, -3, 0, 0, 0, 0, 0),
        scalar(0.5977)
    );
    //Mass of one single droplet calculated with the assumption of r=1 micron and density taken from above
    dimensionedScalar drop_radius
    (
        "drop_radius",
        dimensionSet(0, 1, 0, 0, 0, 0, 0),
        scalar(1e-6)
    );

    dimensionedScalar atm_T
    (
        "atm_T",
        dimensionSet(0, 0, 0, 1, 0, 0, 0),
        scalar(373)
    );
    dimensionedScalar atm_p
    (
        "atm_p",
        dimensionSet(1, -1, -2, 0, 0, 0, 0),
        scalar(101325)
    );
    //dh=2256500 J/kg * 18 g/mol
    dimensionedScalar dhkg
    (
        "dhkg",
        dimensionSet(0, 2, -2, 0, 0,0,0),
        scalar(-2256500)
    );
    dimensionedScalar dhmol
    (
        "dhmol",
        dimensionSet(1, 2, -2, 0, -1,0,0),
        scalar(-40660)
    );    
    dimensionedScalar dsmol
    (
        "dsmol",
        dimensionSet(1, 2, -2, -1, -1,0,0),
        scalar(-85)
    ); 
   
    
    // dimensionedScalar dskg
    // (
    //     "dskg",
    //     dimensionSet(0, 2, -2, -1, 0,0,0),
    //     scalar(-6047)
    // );
    //surface tension from: https://www.engineeringtoolbox.com/surface-tension-d_962.html at 100C
    dimensionedScalar sigma
    (
        "sigma",
        dimensionSet(1, 0, -2, 0, 0,0,0),
        scalar(0.0588)
    );

    const dimensionedScalar drop_vol = (4.0 * M_PI / 3.0)*pow(drop_radius,3);
    const dimensionedScalar drop_mass = rhol_kg*drop_vol;
    const dimensionedScalar Nmol=drop_mass/(mol_mass*drop_vol); //mol/m3

    //Calculation of FIXED W
    // const dimensionedScalar dg =  dhmol*Nmol-atm_T*dsmol*Nmol;
    // Info << "\ndg = " << dg<< "\n";
    // const dimensionedScalar etha = dg / (dhmol*Nmol);
    //     std::cout << "\nFinished etha" << "\n";
    // Info << "\netha = " << etha<< "\n";
    // const dimensionedScalar q =  pow(1.0-etha, 0.5f);
    // std::cout << "\nFinished q\n" << endl;
    // const dimensionedScalar psii = 2.0 * (1.0 + q) * pow(etha, -3) - (3.0 + 2.0 * q) * pow(etha, -2) + pow(etha, -1);
    // std::cout << "\nFinished psii\n" << endl;
    // Info << "\npsii = " << psii<< "\n";
    // const dimensionedScalar W = -(4.0 * M_PI / 3.0) * pow(d, 3) * dg * psii;
    // std::cout << "\nFinished W\n" << endl;
    // Info << "\nW = " << W << "\n";
    // const dimensionedScalar J = pow(rhov_kg, 2) * pow(rhol_kg, -1) * pow(2.0 * sigma / (M_PI * pow(drop_mass,3.0)), 0.5f);
    // const volScalarField I = J * exp(-W / (T * kb));
    // std::cout << "\nFinished I\n" << endl;

    //Calculation of VARIABLE W
     
    // const volScalarField ds= dskg*drop_mass*Nmol;
    // const volScalarField T_pos=pow(pow(T,2),0.5f);

    const volScalarField dg =  dhmol*Nmol-T*dsmol*Nmol;
    // Info << "\ndg = " << dg<< "\n";
    volScalarField etha=pow(pow(dg / (dhmol*Nmol),2),0.5f);
    // std::cout << "\nFinished etha" << "\n";
    
    // forAll(T, cellI) {
    // if (etha[cellI] > 1){
    //     std::cout << "\nCORRECTING ETHA\n" << endl;
    //     etha[cellI] = etha[cellI-1];
    // }
    // }
    // Info << "\nT = " << T_pos<< "\n";
    // Info << "\netha = " << etha<< "\n";
    const volScalarField q =  pow(1.0-etha, 0.5f);
    // std::cout << "\nFinished q\n" << endl;
    // Info << "\nq = " << q<< "\n";
    const volScalarField psii = 2.0 * (1.0 + q) * pow(etha, -3) - (3.0 + 2.0 * q) * pow(etha, -2) + pow(etha, -1);
    // std::cout << "\nFinished psii\n" << endl;
    // Info << "\npsii = " << psii<< "\n";
    const volScalarField W = -(4.0 * M_PI / 3.0) * pow(d, 3) * dg * psii;
    // const volScalarField W=-((4.0 * M_PI / 3.0)*pow(d, 3)*dh*T*ds)/(dh+T*ds-2*pow(dh*T*ds,0.5f));
    // std::cout << "\nFinished W\n" << endl;
    //pre-exponent constant calculated according to granasy 95
    // const dimensionedScalar J = pow(rhov_kg, 2) * pow(rhol_kg, -1) * pow(2.0 * sigma / (M_PI * pow(drop_mass,3.0)), 0.5f);
    // Info << "\nW = " << W << "\n";
 
    // Info << "\nJ0 = " << J << "\n";


    //Calculation of pre-exponent J:
    const volScalarField Rs=d*(1+q)*pow(etha,-1);
    // std::cout << "\nFinished Rs\n" << endl;
    const volScalarField A_star=4*M_PI*pow(Rs,2);
    // std::cout << "\nFinished A_star\n" << endl;
    const volScalarField Z_star = pow(pow(pow(dg,2),0.5f)*(Rs-d/etha)/(kb*T),0.5f)/(Nmol*(2*M_PI*pow(Rs,2)));
    // std::cout << "\nFinished Z_star\n" << endl;
    const volScalarField J = Nmol*A_star*p*pow(2*M_PI*drop_mass*kb*T,-0.5f)*Z_star;
    // std::cout << "\nFinished J\n" << endl;

    // Info << "\nW/kb = " << -W/(T_pos*kb) << "\n";
    const volScalarField I = J * exp(-pow(pow(W,2),0.5f) / (T * kb));
    // Info << "\nI = " << I << "\n";
    // std::cout << "\nFinished I\n" << endl;

    return I;
}
