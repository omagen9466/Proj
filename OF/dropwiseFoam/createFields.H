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

volScalarField r_min
(
    IOobject
    (
        "r_min",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::AUTO_WRITE
    ),
    mesh,
    dimensionedScalar("r_min",dimensionSet(0,1,0,0,0,0,0),0)
);
volScalarField N_s
(
    IOobject
    (
        "N_s",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::NO_WRITE
    ),
    mesh,
    dimensionedScalar("N_s",dimensionSet(0,-2,0,0,0,0,0),0)
);
volScalarField I
(
    IOobject
    (
        "I",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::NO_WRITE
    ),
    mesh,
    dimensionedScalar("I",dimensionSet(0,-3,-1,0,0,0,0),0)
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
volScalarField rdim
(
    IOobject
    (
        "rdim",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::AUTO_WRITE
    ),
    mesh,
    dimensionedScalar("rdim",dimensionSet(0,1,0,0,0,0,0),1)
);
dimensionedScalar dummy
    (
        "dummy",
        dimensionSet(0, 0, 0, 0, 0, 0, 0),
        scalar(1)
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
dimensionedScalar rhol_kg
(
    "rhol_kg",
    dimensionSet(1, -3, 0, 0, 0, 0, 0),
    scalar(960.77)
);
dimensionedScalar sigma
(
    "sigma",
    dimensionSet(1, 0, -2, 0, 0,0,0),
    scalar(0.0588)
);
dimensionedScalar dhkg
(
    "dhkg",
    dimensionSet(0, 2, -2, 0, 0,0,0),
    scalar(2256400)
);
dimensionedScalar T_sat
(
    "T_sat",
    dimensionSet(0, 0, 0, 1, 0,0,0),
    scalar(373)
);

dimensionedScalar mol_mass
(
    "mol_mass",
    dimensionSet(1, 0, 0, 0, -1, 0, 0),
    scalar(0.018)
);
dimensionedScalar R_
(
    "R_",
    dimensionSet(1, 2, -2, -1, -1, 0, 0),
    scalar(8.314)
);
dimensionedScalar spec_vol
(
    "spec_vol",
    dimensionSet(-1, 3, 0, 0, 0, 0, 0),
    scalar(1.671958)
);
dimensionedScalar k_copp
(
    "k_copp",
    dimensionSet(1, 0, -3, -1, 0, 0, 0),
    scalar(0.64)
);
volScalarField dVol
(
    IOobject
    (
        "dVol",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::NO_WRITE
    ),
    mesh,
    dimensionedScalar("dVol",dimensionSet(0, 3, 0, 0, 0,0,0),0)
);

dimensionedScalar theta
(
    "theta",
    dimensionSet(0, 0, 0, 0, 0, 0, 0),
    scalar(100)
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
    dimensionedScalar("hp",dimensionSet(0, 2, -2, 0, 0,0,0),0)
);
    volScalarField h_int
(
    IOobject
    (
        "h_int",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::AUTO_WRITE
    ),
    mesh,
    dimensionedScalar("h_int",dimensionSet(1,0,-3,-1,0,0,0),0)
);
    volScalarField qd
(
    IOobject
    (
        "qd",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::AUTO_WRITE
    ),
    mesh,
    dimensionedScalar("qd",dimensionSet(1,2,-3,0,0,0,0),0)
);
volScalarField r_eq
(
    IOobject
    (
        "r_eq",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::AUTO_WRITE
    ),
    mesh,
    dimensionedScalar("r_eq",dimensionSet(0,1,0,0,0,0,0),0)
);
volScalarField r_max
(
    IOobject
    (
        "r_max",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::AUTO_WRITE
    ),
    mesh,
    dimensionedScalar("r_max",dimensionSet(0,1,0,0,0,0,0),0)
);
volScalarField N_r
(
    IOobject
    (
        "N_r",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::AUTO_WRITE
    ),
    mesh,
    dimensionedScalar("N_r",dimensionSet(0,-3,0,0,0,0,0),0)
);
volScalarField delta_r
(
    IOobject
    (
        "delta_r",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::NO_WRITE
    ),
    mesh,
    dimensionedScalar("delta_r",dimensionSet(0,1,0,0,0,0,0),0)
);
volScalarField r_dummy
(
    IOobject
    (
        "r_dummy",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::NO_WRITE
    ),
    mesh,
    dimensionedScalar("r_dummy",dimensionSet(0,1,0,0,0,0,0),0)
);
volScalarField heat_vol
(
    IOobject
    (
        "heat_vol",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::NO_WRITE
    ),
    mesh,
    dimensionedScalar("heat_vol",dimensionSet(1,-1,-3,0,0,0,0),0)
);
volScalarField heat_flux
(
    IOobject
    (
        "heat_flux",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::AUTO_WRITE
    ),
    mesh,
    dimensionedScalar("heat_flux",dimensionSet(1,0,-3,0,0,0,0),0)
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
