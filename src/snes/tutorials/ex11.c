static char help[] = "Linear poroelasticity with finite elements.\n\
We solve the poroelasticity problem in a rectangular\n\
domain, using a parallel unstructured mesh (DMPLEX) to discretize it.\n\
This example supports automatic convergence estimation\n\
and eventually adaptivity.\n\n\n";

/* This development comes from "Poroelasticity" by Cheng. We have

     \frac{d\zeta}{dt} + \nabla\cdot\vb{q} = g (mass conservation)
                         \nabla\cdot\sigma = f (momentum conservation)

   for which we need constitutive laws

     q      = -\kappa \nabla p (Darcy's Law)
     \zeta  = \frac{p}{M} + \alpha \varepsilon
     \sigma = 2 G \epsilon + \frac{2 G \nu}{1 - 2\nu} \varepsilon I - \alpha p I

   where the strain tensor is defined by

     \epsilon    = \frac{1}{2} \left( \nabla u + \nabla u^T)
     \varepsilon = \Tr{\epsilon} = \nabla \cdot u

   Plugging in these definitions gives

     \frac{1}{M} \frac{dp}{dt} + \alpha \frac{d\varepsilon}{dt} - \nabla \cdot \kappa \nabla p = g
        2 G \nabla\cdot\epsilon + \frac{2 G \nu}{1 - 2\nu} \nabla\varepsilon - \alpha \nabla p = f
                                                                  \nabla \cdot u - \varepsilon = 0

   for fields $(u p, \varepsilon)$. The weak form would then be, using test function $(v, q, \tau)$,

               (q, \frac{1}{M} \frac{dp}{dt}) + (q, \alpha \frac{d\varepsilon}{dt}) + (\nabla q, \kappa \nabla p) = (q, g)
    -(\nabla v, 2 G \epsilon) - (\nabla\cdot v, \frac{2 G \nu}{1 - 2\nu} \varepsilon) + \alpha (\nabla\cdot v, p) = (v, f)
                                                                             (\tau, \nabla \cdot u - \varepsilon) = 0

   In matrix form, we have for $(u, p, \varepsilon)$

     \begin{pmatrix}
           G E     &          -\alpha \nabla          & \beta \nabla \\
            0      & \frac{1}{M dt} I - \kappa \Delta & \alpha I \\
       \nabla\cdot &            0                     &     -I
     \end{pmatrix}

   or $(u, \varepsilon, p)$

     \begin{pmatrix}
           G E     & \beta \nabla &       -\alpha \nabla             \\
       \nabla\cdot &     -I       &              0                   \\
            0      &   \alpha I   & \frac{1}{M dt} I - \kappa \Delta
     \end{pmatrix}

*/

#include <petscdmplex.h>
#include <petscsnes.h>
#include <petscds.h>
#include <petscconvest.h>
#include <petscbag.h>

typedef enum {SOL_QUADRATIC, NUM_SOLUTION_TYPES} SolutionType;
const char *solutionTypes[NUM_SOLUTION_TYPES+1] = {"quadratic", "unknown"};

typedef struct {
  PetscScalar mu;    /* shear modulus */
  PetscScalar K_u;   /* undrained bulk modulus */
  PetscScalar alpha; /* Biot effective stress coefficient */
  PetscScalar M;     /* Biot modulus */
  PetscScalar k;     /* (isotropic) permeability */
  PetscScalar mu_f;  /* fluid dynamic viscosity */
} Parameter;

typedef struct {
  /* Domain and mesh definition */
  char         dmType[256]; /* DM type for the solve */
  PetscInt     dim;         /* The topological mesh dimension */
  PetscBool    simplex;     /* Simplicial mesh */
  /* Problem definition */
  SolutionType solType;     /* Type of exact solution */
  PetscBag     bag;         /* Problem parameters */
} AppCtx;

static PetscErrorCode quadratic_2d_u(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  u[0] = x[0]*x[0];
  u[1] = x[1]*x[1] - 2.0*x[0]*x[1];
  return 0;
}

static PetscErrorCode linear_2d_eps(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  u[0] = 2.0*x[1];
  return 0;
}

static PetscErrorCode linear_2d_p(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  u[0] = x[0] + x[1];
  return 0;
}

static PetscErrorCode quadratic_3d_u(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  u[0] = x[0]*x[0];
  u[1] = x[1]*x[1] - 2.0*x[0]*x[1];
  u[2] = x[2]*x[2] - 2.0*x[1]*x[2];
  return 0;
}

/*
  u = x^2
  v = y^2 - 2xy
  p = x + y
  e = 2y
  f = <2 G, 4 G + 2 \lambda >
  g = 0
  \epsilon = / 2x     -y    \
             \ -y   2y - 2x /
  Tr(\epsilon) = e = div u = 2y
  div \sigma = \partial_i 2 G \epsilon_{ij} + \partial_i \lambda \varepsilon \delta_{ij} - \partial_i \alpha p \delta_{ij}
    = 2 G < 2-1, 2 > + \lambda <0, 2> - alpha <1, 1>
    = <2 G, 4 G + 2 \lambda> - <alpha, alpha>
  \frac{1}{M} \frac{dp}{dt} + \alpha \frac{d\varepsilon}{dt} - \nabla \cdot \kappa \nabla p
    = \kappa \Delta p
    = 0

  u = x^2
  v = y^2 - 2xy
  w = z^2 - 2yz
  \varepsilon = / 2x     -y       0   \
                | -y   2y - 2x   -z   |
                \  0     -z    2z - 2y/
  Tr(\varepsilon) = div u = 2z
  div \sigma = \partial_i \lambda \delta_{ij} \varepsilon_{kk} + \partial_i 2\mu\varepsilon_{ij}
    = \lambda \partial_j (2z) + 2\mu < 2-1, 2-1, 2 >
    = \lambda < 0, 0, 2 > + \mu < 2, 2, 4 >
*/
static void f0_quadratic_u(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                           const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                           const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                           PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  const PetscReal G      = constants[0];
  const PetscReal K_u    = constants[1];
  const PetscReal alpha  = constants[2];
  const PetscReal M      = constants[3];
  const PetscReal K_d    = K_u - alpha*alpha*M;
  const PetscReal lambda = K_d - (2.0 * G) / 3.0;
  PetscInt        d;

  for (d = 0; d < dim-1; ++d) f0[d] -= 2.0*G - alpha;
  f0[dim-1] -= 2.0*lambda + 4.0*G - alpha;
}

static void f1_u(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                 const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                 const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                 PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f1[])
{
  const PetscInt  Nc     = dim;
  const PetscReal G      = constants[0];
  const PetscReal K_u    = constants[1];
  const PetscReal alpha  = constants[2];
  const PetscReal M      = constants[3];
  const PetscReal K_d    = K_u - alpha*alpha*M;
  const PetscReal lambda = K_d - (2.0 * G) / 3.0;
  PetscInt        c, d;

  for (c = 0; c < Nc; ++c) {
    for (d = 0; d < dim; ++d) {
      f1[c*dim+d] -= G*(u_x[c*dim+d] + u_x[d*dim+c]);
    }
    f1[c*dim+c] -= lambda*u[uOff[1]];
    f1[c*dim+c] += alpha*u[uOff[2]];
  }
}

static void f0_epsilon(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                       const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                       const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                       PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  PetscInt d;

  for (d = 0; d < dim; ++d) {
    f0[0] += u_x[d*dim+d];
  }
  f0[0] -= u[uOff[1]];
}

static void f0_p(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                 const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                 const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                 PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  const PetscReal alpha  = constants[2];
  const PetscReal M      = constants[3];

  f0[0] += u_t ? alpha*u_t[uOff[1]] : 0.0;
  f0[0] += u_t ? u_t[uOff[2]]/M     : 0.0;
}

static void f1_p(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                 const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                 const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                 PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f1[])
{
  const PetscReal kappa = constants[4];
  PetscInt        d;

  for (d = 0; d < dim; ++d) {
    f1[d] += kappa*u_x[uOff_x[2]+d];
  }
}

/*
  \partial_df \phi_fc g_{fc,gc,df,dg} \partial_dg \phi_gc

  \partial_df \phi_fc \lambda \delta_{fc,df} \sum_gc \partial_dg \phi_gc \delta_{gc,dg}
  = \partial_fc \phi_fc \sum_gc \partial_gc \phi_gc
*/
static void g3_uu(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                  const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                  const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                  PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g3[])
{
  const PetscInt  Nc = dim;
  const PetscReal G  = constants[0];
  PetscInt        c, d;

  for (c = 0; c < Nc; ++c) {
    for (d = 0; d < dim; ++d) {
      g3[((c*Nc + c)*dim + d)*dim + d] -= G;
      g3[((c*Nc + d)*dim + d)*dim + c] -= G;
    }
  }
}

static void g2_ue(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                  const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                  const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                  PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g2[])
{
  const PetscReal G      = constants[0];
  const PetscReal K_u    = constants[1];
  const PetscReal alpha  = constants[2];
  const PetscReal M      = constants[3];
  const PetscReal K_d    = K_u - alpha*alpha*M;
  const PetscReal lambda = K_d - (2.0 * G) / 3.0;
  PetscInt        d;

  for (d = 0; d < dim; ++d) {
    g2[d*dim + d] -= lambda;
  }
}

static void g2_up(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                  const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                  const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                  PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g2[])
{
  const PetscReal alpha = constants[2];
  PetscInt        d;

  for (d = 0; d < dim; ++d) {
    g2[d*dim + d] += alpha;
  }
}

/* < q, \nabla\cdot u >
   NcompI = 1, NcompJ = dim */
static void g1_eu(PetscInt dim, PetscInt Nf, PetscInt NfAux,
           const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
           const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
           PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g1[])
{
  PetscInt d;
  for (d = 0; d < dim; ++d) g1[d*dim+d] = 1.0; /* \frac{\partial\phi^{u_d}}{\partial x_d} */
}

static void g0_ee(PetscInt dim, PetscInt Nf, PetscInt NfAux,
           const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
           const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
           PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g0[])
{
  g0[0] = -1.0;
}

static void g3_pp(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                  const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                  const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                  PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g3[])
{
  const PetscReal kappa = constants[4];
  PetscInt        d;

  for (d = 0; d < dim; ++d) g3[d*dim+d] = kappa;
}

static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscInt       sol;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  options->dim     = 2;
  options->simplex = PETSC_TRUE;
  options->solType = SOL_QUADRATIC;
  ierr = PetscStrncpy(options->dmType, DMPLEX, 256);CHKERRQ(ierr);

  ierr = PetscOptionsBegin(comm, "", "Linear Elasticity Problem Options", "DMPLEX");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-dim", "The topological mesh dimension", "ex17.c", options->dim, &options->dim, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-simplex", "Simplicial (true) or tensor (false) mesh", "ex17.c", options->simplex, &options->simplex, NULL);CHKERRQ(ierr);
  sol  = options->solType;
  ierr = PetscOptionsEList("-sol_type", "Type of exact solution", "ex17.c", solutionTypes, NUM_SOLUTION_TYPES, solutionTypes[options->solType], &sol, NULL);CHKERRQ(ierr);
  options->solType = (SolutionType) sol;
  ierr = PetscOptionsFList("-dm_type", "Convert DMPlex to another format", "ex17.c", DMList, options->dmType, options->dmType, 256, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();
  PetscFunctionReturn(0);
}

static PetscErrorCode SetupParameters(AppCtx *ctx)
{
  PetscBag       bag;
  Parameter     *p;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = PetscBagGetData(ctx->bag, (void **) &p);CHKERRQ(ierr);
  ierr = PetscBagSetName(ctx->bag, "par", "Poroelastic Parameters");CHKERRQ(ierr);
  bag  = ctx->bag;
  ierr = PetscBagRegisterScalar(bag, &p->mu,    1.0, "mu",    "Shear Modulus, Pa");CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &p->K_u,   1.0, "K_u",   "Undrained Bulk Modulus, Pa");CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &p->alpha, 1.0, "alpha", "Biot Effective Stress Coefficient");CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &p->M,     1.0, "M",     "Biot Modulus");CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &p->k,     1.0, "k",     "Isotropic Permeability, m**2");CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &p->mu_f,  1.0, "mu_f",  "Fluid Dynamic Viscosity, Pa*s");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *user, DM *dm)
{
  PetscBool      flg;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = DMPlexCreateBoxMesh(comm, user->dim, user->simplex, NULL, NULL, NULL, NULL, PETSC_TRUE, dm);CHKERRQ(ierr);
  {
    DM               pdm = NULL;
    PetscPartitioner part;

    ierr = DMPlexGetPartitioner(*dm, &part);CHKERRQ(ierr);
    ierr = PetscPartitionerSetFromOptions(part);CHKERRQ(ierr);
    ierr = DMPlexDistribute(*dm, 0, NULL, &pdm);CHKERRQ(ierr);
    if (pdm) {
      ierr = DMDestroy(dm);CHKERRQ(ierr);
      *dm  = pdm;
    }
  }
  ierr = PetscStrcmp(user->dmType, DMPLEX, &flg);CHKERRQ(ierr);
  if (flg) {
    DM ndm;

    ierr = DMConvert(*dm, user->dmType, &ndm);CHKERRQ(ierr);
    if (ndm) {
      ierr = DMDestroy(dm);CHKERRQ(ierr);
      *dm  = ndm;
    }
  }
  ierr = DMLocalizeCoordinates(*dm);CHKERRQ(ierr);

  ierr = PetscObjectSetName((PetscObject) *dm, "Mesh");CHKERRQ(ierr);
  ierr = DMSetApplicationContext(*dm, user);CHKERRQ(ierr);
  ierr = DMSetFromOptions(*dm);CHKERRQ(ierr);
  ierr = DMViewFromOptions(*dm, NULL, "-dm_view");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode SetupPrimalProblem(DM dm, AppCtx *user)
{
  PetscErrorCode (*exact[3])(PetscInt, PetscReal, const PetscReal[], PetscInt, PetscScalar *, void *);
  PetscDS        prob;
  PetscInt       id;
  PetscInt       dim;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = DMGetDS(dm, &prob);CHKERRQ(ierr);
  ierr = PetscDSGetSpatialDimension(prob, &dim);CHKERRQ(ierr);
  switch (user->solType) {
  case SOL_QUADRATIC:
    ierr = PetscDSSetResidual(prob, 0, f0_quadratic_u, f1_u);CHKERRQ(ierr);
    ierr = PetscDSSetResidual(prob, 1, f0_epsilon,     NULL);CHKERRQ(ierr);
    ierr = PetscDSSetResidual(prob, 2, f0_p,           f1_p);CHKERRQ(ierr);
    ierr = PetscDSSetJacobian(prob, 0, 0, NULL,  NULL,  NULL,  g3_uu);CHKERRQ(ierr);
    ierr = PetscDSSetJacobian(prob, 0, 1, NULL,  NULL,  g2_ue, NULL);CHKERRQ(ierr);
    ierr = PetscDSSetJacobian(prob, 0, 2, NULL,  NULL,  g2_up, NULL);CHKERRQ(ierr);
    ierr = PetscDSSetJacobian(prob, 1, 0, NULL,  g1_eu, NULL,  NULL);CHKERRQ(ierr);
    ierr = PetscDSSetJacobian(prob, 1, 1, g0_ee, NULL,  NULL,  NULL);CHKERRQ(ierr);
    ierr = PetscDSSetJacobian(prob, 2, 2, NULL,  NULL,  NULL,  g3_pp);CHKERRQ(ierr);
    switch (dim) {
    case 2:
      exact[0] = quadratic_2d_u;
      exact[1] = linear_2d_eps;
      exact[2] = linear_2d_p;
      break;
    case 3:
      exact[0] = quadratic_3d_u;
      exact[1] = linear_2d_eps;
      exact[2] = linear_2d_p;
      break;
    default: SETERRQ1(PetscObjectComm((PetscObject) prob), PETSC_ERR_ARG_WRONG, "Invalid dimension: %D", dim);
    }
    break;
  default: SETERRQ2(PetscObjectComm((PetscObject) prob), PETSC_ERR_ARG_WRONG, "Invalid solution type: %s (%D)", solutionTypes[PetscMin(user->solType, NUM_SOLUTION_TYPES)], user->solType);
  }
  ierr = PetscDSSetExactSolution(prob, 0, exact[0], user);CHKERRQ(ierr);
  ierr = PetscDSSetExactSolution(prob, 1, exact[1], user);CHKERRQ(ierr);
  ierr = PetscDSSetExactSolution(prob, 2, exact[2], user);CHKERRQ(ierr);
  {
    id = 1;
    ierr = DMAddBoundary(dm, DM_BC_ESSENTIAL, "wall displacement", "marker", 0, 0, NULL, (void (*)(void)) exact[0], 1, &id, user);CHKERRQ(ierr);
    ierr = DMAddBoundary(dm, DM_BC_ESSENTIAL, "wall pressure",     "marker", 2, 0, NULL, (void (*)(void)) exact[2], 1, &id, user);CHKERRQ(ierr);
  }
  {
    Parameter  *param;
    PetscScalar constants[5];

    ierr = PetscBagGetData(user->bag, (void **) &param);CHKERRQ(ierr);

    constants[0] = param->mu;            /* shear modulus */
    constants[1] = param->K_u;           /* undrained bulk modulus */
    constants[2] = param->alpha;         /* Biot effective stress coefficient */
    constants[3] = param->M;             /* Biot modulus */
    constants[4] = param->k/param->mu_f; /* Darcy coefficient */
    ierr = PetscDSSetConstants(prob, 5, constants);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateElasticityNullSpace(DM dm, PetscInt dummy, MatNullSpace *nullspace)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMPlexCreateRigidBody(dm, nullspace);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode SetupFE(DM dm, PetscBool simplex, PetscInt Nf, PetscInt Nc[], const char *name[], PetscErrorCode (*setup)(DM, AppCtx *), void *ctx)
{
  AppCtx         *user = (AppCtx *) ctx;
  DM              cdm  = dm;
  PetscFE         fe;
  PetscQuadrature q = NULL;
  char            prefix[PETSC_MAX_PATH_LEN];
  PetscInt        dim, f;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  /* Create finite element */
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  for (f = 0; f < Nf; ++f) {
    ierr = PetscSNPrintf(prefix, PETSC_MAX_PATH_LEN, "%s_", name[f]);CHKERRQ(ierr);
    ierr = PetscFECreateDefault(PetscObjectComm((PetscObject) dm), dim, Nc[f], simplex, prefix, -1, &fe);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) fe, name[f]);CHKERRQ(ierr);
    if (!q) {ierr = PetscFEGetQuadrature(fe, &q);CHKERRQ(ierr);}
    ierr = PetscFESetQuadrature(fe, q);CHKERRQ(ierr);
    ierr = DMSetField(dm, f, NULL, (PetscObject) fe);CHKERRQ(ierr);
    ierr = PetscFEDestroy(&fe);CHKERRQ(ierr);
  }
  ierr = DMCreateDS(dm);CHKERRQ(ierr);
  ierr = (*setup)(dm, user);CHKERRQ(ierr);
  while (cdm) {
    ierr = DMCopyDisc(dm, cdm);CHKERRQ(ierr);
    if (0) {ierr = DMSetNearNullSpaceConstructor(cdm, 0, CreateElasticityNullSpace);CHKERRQ(ierr);}
    /* TODO: Check whether the boundary of coarse meshes is marked */
    ierr = DMGetCoarseDM(cdm, &cdm);CHKERRQ(ierr);
  }
  ierr = PetscFEDestroy(&fe);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  DM             dm;   /* Problem specification */
  SNES           snes; /* Nonlinear solver */
  Vec            u;    /* Solutions */
  AppCtx         user; /* User-defined work context */
  const char    *name[3] = {"displacement", "tracestrain", "pressure"};
  PetscInt       Nc[3];
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL,help);if (ierr) return ierr;
  ierr = ProcessOptions(PETSC_COMM_WORLD, &user);CHKERRQ(ierr);
  ierr = PetscBagCreate(PETSC_COMM_SELF, sizeof(Parameter), &user.bag);CHKERRQ(ierr);
  ierr = SetupParameters(&user);CHKERRQ(ierr);
  /* Primal system */
  ierr = SNESCreate(PETSC_COMM_WORLD, &snes);CHKERRQ(ierr);
  ierr = CreateMesh(PETSC_COMM_WORLD, &user, &dm);CHKERRQ(ierr);
  ierr = SNESSetDM(snes, dm);CHKERRQ(ierr);
  Nc[0] = user.dim;
  Nc[1] = 1;
  Nc[2] = 1;
  ierr = SetupFE(dm, user.simplex, 3, Nc, name, SetupPrimalProblem, &user);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(dm, &u);CHKERRQ(ierr);
  ierr = VecSet(u, 0.0);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) u, "solution");CHKERRQ(ierr);
  ierr = DMPlexSetSNESLocalFEM(dm, &user, &user, &user);CHKERRQ(ierr);
  ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);
  ierr = DMSNESCheckFromOptions(snes, u, NULL, NULL);CHKERRQ(ierr);
  ierr = SNESSolve(snes, NULL, u);CHKERRQ(ierr);
  ierr = SNESGetSolution(snes, &u);CHKERRQ(ierr);
  ierr = VecViewFromOptions(u, NULL, "-sol_view");CHKERRQ(ierr);
  /* Cleanup */
  ierr = VecDestroy(&u);CHKERRQ(ierr);
  ierr = SNESDestroy(&snes);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = PetscBagDestroy(&user.bag);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

  test:
    suffix: 2d_p1_quad
    requires: triangle
    args: -displacement_petscspace_degree 2 -tracestrain_petscspace_degree 1 -pressure_petscspace_degree 1 -dm_refine 2 -dmsnes_check .0001
  test:
    suffix: 2d_p1_trig
    requires: triangle
    args: -displacement_petscspace_degree 2 -tracestrain_petscspace_degree 1 -pressure_petscspace_degree 1 -dm_refine 2 -convest_num_refine 3 -snes_convergence_estimate
  test:
    suffix: 2d_p2_quad_vlap
    requires: triangle
    args: -displacement_petscspace_degree 2 -dm_refine 2 -dmsnes_check .0001
  test:
    suffix: 2d_p3_quad_vlap
    requires: triangle
    args: -displacement_petscspace_degree 3 -dm_refine 2 -dmsnes_check .0001
  test:
    suffix: 2d_q1_quad_vlap
    args: -simplex 0 -displacement_petscspace_degree 1 -dm_refine 2 -convest_num_refine 3 -snes_convergence_estimate
  test:
    suffix: 2d_q2_quad_vlap
    args: -simplex 0 -displacement_petscspace_degree 2 -dm_refine 2 -dmsnes_check .0001
  test:
    suffix: 2d_q3_quad_vlap
    requires: !single
    args: -simplex 0 -displacement_petscspace_degree 3 -dm_refine 2 -dmsnes_check .0001
  test:
    suffix: 2d_p1_quad_elas
    requires: triangle
    args: -sol_type elas_quad -displacement_petscspace_degree 1 -dm_refine 2 -convest_num_refine 3 -snes_convergence_estimate
  test:
    suffix: 2d_p2_quad_elas
    requires: triangle
    args: -sol_type elas_quad -displacement_petscspace_degree 2 -dmsnes_check .0001
  test:
    suffix: 2d_p3_quad_elas
    requires: triangle
    args: -sol_type elas_quad -displacement_petscspace_degree 3 -dmsnes_check .0001
  test:
    suffix: 2d_q1_quad_elas
    args: -sol_type elas_quad -simplex 0 -displacement_petscspace_degree 1 -dm_refine 1 -convest_num_refine 3 -snes_convergence_estimate
  test:
    suffix: 2d_q1_quad_elas_shear
    args: -sol_type elas_quad -simplex 0 -shear -displacement_petscspace_degree 1 -dm_refine 1 -convest_num_refine 3 -snes_convergence_estimate
  test:
    suffix: 2d_q2_quad_elas
    args: -sol_type elas_quad -simplex 0 -displacement_petscspace_degree 2 -dmsnes_check .0001
  test:
    suffix: 2d_q2_quad_elas_shear
    args: -sol_type elas_quad -simplex 0 -shear -displacement_petscspace_degree 2 -dmsnes_check
  test:
    suffix: 2d_q3_quad_elas
    args: -sol_type elas_quad -simplex 0 -displacement_petscspace_degree 3 -dmsnes_check .0001
  test:
    suffix: 2d_q3_quad_elas_shear
    requires: !single
    args: -sol_type elas_quad -simplex 0 -shear -displacement_petscspace_degree 3 -dmsnes_check

  test:
    suffix: 3d_p1_quad_vlap
    requires: ctetgen
    args: -dim 3 -cells 2,2,2 -displacement_petscspace_degree 1 -convest_num_refine 2 -snes_convergence_estimate
  test:
    suffix: 3d_p2_quad_vlap
    requires: ctetgen
    args: -dim 3 -displacement_petscspace_degree 2 -dm_refine 1 -dmsnes_check .0001
  test:
    suffix: 3d_p3_quad_vlap
    requires: ctetgen
    args: -dim 3 -displacement_petscspace_degree 3 -dm_refine 0 -dmsnes_check .0001
  test:
    suffix: 3d_q1_quad_vlap
    args: -dim 3 -cells 2,2,2 -simplex 0 -displacement_petscspace_degree 1 -convest_num_refine 2 -snes_convergence_estimate
  test:
    suffix: 3d_q2_quad_vlap
    args: -dim 3 -simplex 0 -displacement_petscspace_degree 2 -dm_refine 1 -dmsnes_check .0001
  test:
    suffix: 3d_q3_quad_vlap
    args: -dim 3 -simplex 0 -displacement_petscspace_degree 3 -dm_refine 0 -dmsnes_check .0001
  test:
    suffix: 3d_p1_quad_elas
    requires: ctetgen
    args: -sol_type elas_quad -dim 3 -cells 2,2,2 -displacement_petscspace_degree 1 -convest_num_refine 2 -snes_convergence_estimate
  test:
    suffix: 3d_p2_quad_elas
    requires: ctetgen
    args: -sol_type elas_quad -dim 3 -displacement_petscspace_degree 2 -dm_refine 1 -dmsnes_check .0001
  test:
    suffix: 3d_p3_quad_elas
    requires: ctetgen
    args: -sol_type elas_quad -dim 3 -displacement_petscspace_degree 3 -dm_refine 0 -dmsnes_check .0001
  test:
    suffix: 3d_q1_quad_elas
    args: -sol_type elas_quad -dim 3 -cells 2,2,2 -simplex 0 -displacement_petscspace_degree 1 -convest_num_refine 2 -snes_convergence_estimate
  test:
    suffix: 3d_q2_quad_elas
    args: -sol_type elas_quad -dim 3 -simplex 0 -displacement_petscspace_degree 2 -dm_refine 1 -dmsnes_check .0001
  test:
    suffix: 3d_q3_quad_elas
    requires: !single
    args: -sol_type elas_quad -dim 3 -simplex 0 -displacement_petscspace_degree 3 -dm_refine 0 -dmsnes_check .0001

  test:
    suffix: 2d_p1_trig_vlap
    requires: triangle
    args: -sol_type vlap_trig -displacement_petscspace_degree 1 -dm_refine 1 -convest_num_refine 3 -snes_convergence_estimate
  test:
    suffix: 2d_p2_trig_vlap
    requires: triangle
    args: -sol_type vlap_trig -displacement_petscspace_degree 2 -dm_refine 1 -convest_num_refine 3 -snes_convergence_estimate
  test:
    suffix: 2d_p3_trig_vlap
    requires: triangle
    args: -sol_type vlap_trig -displacement_petscspace_degree 3 -dm_refine 1 -convest_num_refine 3 -snes_convergence_estimate
  test:
    suffix: 2d_q1_trig_vlap
    args: -sol_type vlap_trig -simplex 0 -displacement_petscspace_degree 1 -dm_refine 1 -convest_num_refine 3 -snes_convergence_estimate
  test:
    suffix: 2d_q2_trig_vlap
    args: -sol_type vlap_trig -simplex 0 -displacement_petscspace_degree 2 -dm_refine 1 -convest_num_refine 3 -snes_convergence_estimate
  test:
    suffix: 2d_q3_trig_vlap
    args: -sol_type vlap_trig -simplex 0 -displacement_petscspace_degree 3 -dm_refine 1 -convest_num_refine 3 -snes_convergence_estimate
  test:
    suffix: 2d_p1_trig_elas
    requires: triangle
    args: -sol_type elas_trig -displacement_petscspace_degree 1 -dm_refine 1 -convest_num_refine 3 -snes_convergence_estimate
  test:
    suffix: 2d_p2_trig_elas
    requires: triangle
    args: -sol_type elas_trig -displacement_petscspace_degree 2 -dm_refine 1 -convest_num_refine 3 -snes_convergence_estimate
  test:
    suffix: 2d_p3_trig_elas
    requires: triangle
    args: -sol_type elas_trig -displacement_petscspace_degree 3 -dm_refine 1 -convest_num_refine 3 -snes_convergence_estimate
  test:
    suffix: 2d_q1_trig_elas
    args: -sol_type elas_trig -simplex 0 -displacement_petscspace_degree 1 -dm_refine 1 -convest_num_refine 3 -snes_convergence_estimate
  test:
    suffix: 2d_q1_trig_elas_shear
    args: -sol_type elas_trig -simplex 0 -shear -displacement_petscspace_degree 1 -dm_refine 1 -convest_num_refine 3 -snes_convergence_estimate
  test:
    suffix: 2d_q2_trig_elas
    args: -sol_type elas_trig -simplex 0 -displacement_petscspace_degree 2 -dm_refine 1 -convest_num_refine 3 -snes_convergence_estimate
  test:
    suffix: 2d_q2_trig_elas_shear
    args: -sol_type elas_trig -simplex 0 -shear -displacement_petscspace_degree 2 -dm_refine 1 -convest_num_refine 3 -snes_convergence_estimate
  test:
    suffix: 2d_q3_trig_elas
    args: -sol_type elas_trig -simplex 0 -displacement_petscspace_degree 3 -dm_refine 1 -convest_num_refine 3 -snes_convergence_estimate
  test:
    suffix: 2d_q3_trig_elas_shear
    args: -sol_type elas_trig -simplex 0 -shear -displacement_petscspace_degree 3 -dm_refine 1 -convest_num_refine 3 -snes_convergence_estimate

  test:
    suffix: 3d_p1_trig_vlap
    requires: ctetgen
    args: -sol_type vlap_trig -dim 3 -cells 2,2,2 -displacement_petscspace_degree 1 -convest_num_refine 2 -snes_convergence_estimate
  test:
    suffix: 3d_p2_trig_vlap
    requires: ctetgen
    args: -sol_type vlap_trig -dim 3 -displacement_petscspace_degree 2 -dm_refine 0 -convest_num_refine 1 -snes_convergence_estimate
  test:
    suffix: 3d_p3_trig_vlap
    requires: ctetgen
    args: -sol_type vlap_trig -dim 3 -displacement_petscspace_degree 3 -dm_refine 0 -convest_num_refine 1 -snes_convergence_estimate
  test:
    suffix: 3d_q1_trig_vlap
    args: -sol_type vlap_trig -dim 3 -cells 2,2,2 -simplex 0 -displacement_petscspace_degree 1 -convest_num_refine 2 -snes_convergence_estimate
  test:
    suffix: 3d_q2_trig_vlap
    args: -sol_type vlap_trig -dim 3 -simplex 0 -displacement_petscspace_degree 2 -dm_refine 0 -convest_num_refine 1 -snes_convergence_estimate
  test:
    suffix: 3d_q3_trig_vlap
    requires: !__float128
    args: -sol_type vlap_trig -dim 3 -simplex 0 -displacement_petscspace_degree 3 -dm_refine 0 -convest_num_refine 1 -snes_convergence_estimate
  test:
    suffix: 3d_p1_trig_elas
    requires: ctetgen
    args: -sol_type elas_trig -dim 3 -cells 2,2,2 -displacement_petscspace_degree 1 -convest_num_refine 2 -snes_convergence_estimate
  test:
    suffix: 3d_p2_trig_elas
    requires: ctetgen
    args: -sol_type elas_trig -dim 3 -displacement_petscspace_degree 2 -dm_refine 0 -convest_num_refine 1 -snes_convergence_estimate
  test:
    suffix: 3d_p3_trig_elas
    requires: ctetgen
    args: -sol_type elas_trig -dim 3 -displacement_petscspace_degree 3 -dm_refine 0 -convest_num_refine 1 -snes_convergence_estimate
  test:
    suffix: 3d_q1_trig_elas
    args: -sol_type elas_trig -dim 3 -cells 2,2,2 -simplex 0 -displacement_petscspace_degree 1 -convest_num_refine 2 -snes_convergence_estimate
  test:
    suffix: 3d_q2_trig_elas
    args: -sol_type elas_trig -dim 3 -simplex 0 -displacement_petscspace_degree 2 -dm_refine 0 -convest_num_refine 1 -snes_convergence_estimate
  test:
    suffix: 3d_q3_trig_elas
    requires: !__float128
    args: -sol_type elas_trig -dim 3 -simplex 0 -displacement_petscspace_degree 3 -dm_refine 0 -convest_num_refine 1 -snes_convergence_estimate

  test:
    suffix: 2d_p1_axial_elas
    requires: triangle
    args: -sol_type elas_axial_disp -displacement_petscspace_degree 1 -dm_plex_separate_marker -dm_refine 2 -dmsnes_check .0001 -pc_type lu
  test:
    suffix: 2d_p2_axial_elas
    requires: triangle
    args: -sol_type elas_axial_disp -displacement_petscspace_degree 2 -dm_plex_separate_marker -dmsnes_check .0001 -pc_type lu
  test:
    suffix: 2d_p3_axial_elas
    requires: triangle
    args: -sol_type elas_axial_disp -displacement_petscspace_degree 3 -dm_plex_separate_marker -dmsnes_check .0001 -pc_type lu
  test:
    suffix: 2d_q1_axial_elas
    args: -sol_type elas_axial_disp -simplex 0 -displacement_petscspace_degree 1 -dm_plex_separate_marker -dm_refine 1 -dmsnes_check .0001 -pc_type lu
  test:
    suffix: 2d_q2_axial_elas
    args: -sol_type elas_axial_disp -simplex 0 -displacement_petscspace_degree 2 -dm_plex_separate_marker -dmsnes_check .0001 -pc_type lu
  test:
    suffix: 2d_q3_axial_elas
    args: -sol_type elas_axial_disp -simplex 0 -displacement_petscspace_degree 3 -dm_plex_separate_marker -dmsnes_check .0001 -pc_type lu

  test:
    suffix: 2d_p1_uniform_elas
    requires: triangle
    args: -sol_type elas_uniform_strain -displacement_petscspace_degree 1 -dm_refine 2 -dmsnes_check .0001 -pc_type lu
  test:
    suffix: 2d_p2_uniform_elas
    requires: triangle
    args: -sol_type elas_uniform_strain -displacement_petscspace_degree 2 -dm_refine 2 -dmsnes_check .0001 -pc_type lu
  test:
    suffix: 2d_p3_uniform_elas
    requires: triangle
    args: -sol_type elas_uniform_strain -displacement_petscspace_degree 3 -dm_refine 2 -dmsnes_check .0001 -pc_type lu
  test:
    suffix: 2d_q1_uniform_elas
    args: -sol_type elas_uniform_strain -simplex 0 -displacement_petscspace_degree 1 -dm_refine 2 -dmsnes_check .0001 -pc_type lu
  test:
    suffix: 2d_q2_uniform_elas
    requires: !single
    args: -sol_type elas_uniform_strain -simplex 0 -displacement_petscspace_degree 2 -dm_refine 2 -dmsnes_check .0001 -pc_type lu
  test:
    suffix: 2d_q3_uniform_elas
    requires: !single
    args: -sol_type elas_uniform_strain -simplex 0 -displacement_petscspace_degree 3 -dm_refine 2 -dmsnes_check .0001 -pc_type lu

TEST*/
