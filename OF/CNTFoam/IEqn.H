#include "fvCFD.H"
#include "volFields.H"
#include <cmath>
#include "dimensionSet.H"
#include "dimensionedScalar.H"

{
    std::cout << "\nstarting I\n" << endl;
    const Foam::volScalarField& T = mesh.lookupObject<Foam::volScalarField>("T");
    supersat=p/(atm_p*exp(-(dhkg/R)*((1.0/T)-(1/atm_T))))+1;

    r_crit=(2*sigma)/(rhol_kg*R*T*log(supersat));
    // std::cout << "\ncrit\n" << endl;
    crit_mass=pow(r_crit,3.0)*4*M_PI*rhol_kg/3;
    // std::cout << "\nmass\n" << endl;
    eta=2*(0.333/2.333)*(dhkg/(R*T))*(dhkg/(R*T)-0.5);
    // std::cout << "\neta\n" << endl;
    I=(1/(1+eta))*pow((2*sigma)/(M_PI*pow(crit_mass,3.0)),0.5f)*(pow(rho,2)/rhol_kg)*exp(-(4*M_PI*sigma*pow(r_crit,2))/(3*kb*T));
    std::cout << "\nEnding I\n" << endl;
    
     
    
    // supersat.write();
    

}
