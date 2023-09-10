// Solve the Momentum equation
#include "Sd.H"
MRF.correctBoundaryVelocity(U);

tmp<fvVectorMatrix> tUEqn
(
    fvm::ddt(rho*M, U) + fvm::div(phi*fvc::interpolate(M), U)
//   + MRF.DDt(rho, U)
  + turbulence->divDevTau(U)
 ==
    fvOptions(rho, U)
    -fvc::Sp(Sd,U)
);
fvVectorMatrix& UEqn = tUEqn.ref();

UEqn.relax();

fvOptions.constrain(UEqn);

if (pimple.momentumPredictor())
{
    std::cout << "\nBefore\n" << endl;
    solve(UEqn == -M*fvc::grad(p));
    std::cout << "\nAfter\n" << endl;
    fvOptions.correct(U);
    K = 0.5*magSqr(U);
}
