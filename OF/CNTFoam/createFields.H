#include "createRDeltaT.H"

Info<< "Reading thermophysical properties\n" << endl;

autoPtr<fluidThermo> pThermo
(
    fluidThermo::New(mesh)
);
fluidThermo& thermo = pThermo();
thermo.validate(args.executable(), "h", "e");

volScalarField& p = thermo.p();

volScalarField rho
(
    IOobject
    (
        "rho",
        runTime.timeName(),
        mesh,
        IOobject::READ_IF_PRESENT,
        IOobject::AUTO_WRITE
    ),
    thermo.rho()
);

Info<< "Reading field U\n" << endl;
volVectorField U
(
    IOobject
    (
        "U",
        runTime.timeName(),
        mesh,
        IOobject::MUST_READ,
        IOobject::AUTO_WRITE
    ),
    mesh
);

Info<< "Reading field N\n" << endl;
volScalarField N
(
    IOobject
    (
        "N",
        runTime.timeName(),
        mesh,
        IOobject::MUST_READ,
        IOobject::AUTO_WRITE
    ),
    mesh
);

Info<< "Reading field M\n" << endl;
volScalarField M
(
    IOobject
    (
        "M",
        runTime.timeName(),
        mesh,
        IOobject::MUST_READ,
        IOobject::AUTO_WRITE
    ),
    mesh
);
Info<< "Reading field I\n" << endl;
volScalarField I
(
    IOobject
    (
        "I",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::AUTO_WRITE
    ),
    mesh,
    dimensionedScalar("I",dimensionSet(0,-3,-1,0,0,0,0),0)
);
volScalarField eta
(
    IOobject
    (
        "eta",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::NO_WRITE
    ),
    mesh,
    dimensionedScalar("eta",dimensionSet(0,0,0,0,0,0,0),0)
);
volScalarField r_crit
(
    IOobject
    (
        "r_crit",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::AUTO_WRITE
    ),
    mesh,
    dimensionedScalar("r_crit",dimensionSet(0,1,0,0,0,0,0),0)
);
volScalarField supersat
(
    IOobject
    (
        "supersat",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::AUTO_WRITE
    ),
    mesh,
    dimensionedScalar("supersat",dimensionSet(0,0,0,0,0,0,0),0)
);
volScalarField Kn
(
    IOobject
    (
        "Kn",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::AUTO_WRITE
    ),
    mesh,
    dimensionedScalar("Kn",dimensionSet(0,0,0,0,0,0,0),0)
);
volScalarField crit_mass
(
    IOobject
    (
        "crit_mass",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::NO_WRITE
    ),
    mesh,
    dimensionedScalar("crit_mass",dimensionSet(1,0,0,0,0,0,0),0)
);
dimensionedScalar R
(
    "R",
    dimensionSet(0, 2, -2, -1, 0, 0, 0),
    scalar(461.4)
);
    //interface thickness assumed constant and value taken from granasy 95
    dimensionedScalar d
    (
        "d",
        dimensionSet(0, 1, 0, 0, 0,0,0),
        scalar(0.8e-10)
    );
        dimensionedScalar dummy
    (
        "dummy",
        dimensionSet(0, 1, 0, 0, 0,0,0),
        scalar(1)
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
        dimensionedScalar visc
    (
        "visc",
        dimensionSet(1, -1, -1, 0, 0, 0, 0),
        scalar(0.0002816)
    );
    dimensionedScalar k_copp
    (
        "k_copp",
        dimensionSet(1, 1, -3, -1, 0, 0, 0),
        scalar(0.0248)
    );
    volScalarField hp
    (
        IOobject
        (
            "hp",
            runTime.timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::NO_WRITE
        ),
        mesh,
        dimensionedScalar("hp",dimensionSet(0, 2, -2, 0, 0,0,0),419170)
    );
    volScalarField drdt
(
    IOobject
    (
        "drdt",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::AUTO_WRITE
    ),
    mesh,
    dimensionedScalar("drdt",dimensionSet(0,1,-1,0,0,0,0),0)
);

volScalarField Sd
(
    IOobject
    (
        "Sd",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::NO_WRITE
    ),
    mesh,
    dimensionedScalar("Sd",dimensionSet(1,-3,-1,0,0,0,0),0)
);
    volScalarField r_het
    (
        IOobject
        (
            "r_het",
            runTime.timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        mesh,
        dimensionedScalar("r_het",dimensionSet(0,1,0,0,0,0,0),0)
    );


    dimensionedScalar atm_p
    (
        "atm_p",
        dimensionSet(1, -1, -2, 0, 0, 0, 0),
        scalar(101325)
    );
    volScalarField atm_T
    (
        IOobject
        (
            "atm_T",
            runTime.timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::NO_WRITE
        ),
        mesh,
        dimensionedScalar("atm_T",dimensionSet(0,0,0,1,0,0,0),373)
    );
    volScalarField sd_corrector
    (
        IOobject
        (
            "sd_corrector",
            runTime.timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::NO_WRITE
        ),
        mesh,
        dimensionedScalar("sd_corrector",dimensionSet(0,1,0,0,0,0,0),1)
    );
    //dh=2256500 J/kg * 18 g/mol
    dimensionedScalar dhkg
    (
        "dhkg",
        dimensionSet(0, 2, -2, 0, 0,0,0),
        scalar(2256500)
    );
   
  
    
    dimensionedScalar sigma
    (
        "sigma",
        dimensionSet(1, 0, -2, 0, 0,0,0),
        scalar(0.0588)
    );
#include "compressibleCreatePhi.H"

pressureControl pressureControl
(
    p,
    rho,
    pimple.dict(),
    thermo.incompressible()
);

mesh.setFluxRequired(p.name());

Info<< "Creating turbulence model\n" << endl;
autoPtr<compressible::momentumTransportModel> turbulence
(
    compressible::momentumTransportModel::New
    (
        rho,
        U,
        phi,
        thermo
    )
);

Info<< "Creating thermophysical transport model\n" << endl;
autoPtr<fluidThermophysicalTransportModel> thermophysicalTransport
(
    fluidThermophysicalTransportModel::New(turbulence(), thermo)
);

Info<< "Creating field dpdt\n" << endl;
volScalarField dpdt
(
    IOobject
    (
        "dpdt",
        runTime.timeName(),
        mesh
    ),
    mesh,
    dimensionedScalar(p.dimensions()/dimTime, 0)
);

Info<< "Creating field kinetic energy K\n" << endl;
volScalarField K("K", 0.5*magSqr(U));

dimensionedScalar initialMass = fvc::domainIntegrate(rho);

#include "createMRF.H"
#include "createFvOptions.H"
