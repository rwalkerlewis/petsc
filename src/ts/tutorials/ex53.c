static char help[] = "Time dependent Biot Poroelasticity problem with finite elements.\n\
We solve three field, quasi-static poroelasticity in a rectangular\n\
domain, using a parallel unstructured mesh (DMPLEX) to discretize it.\n\
Contributed by: Robert Walker <rwalker6@buffalo.edu>\n\n\n";

#include <petscdmplex.h>
#include <petscsnes.h>
#include <petscts.h>
#include <petscds.h>
#include <petscbag.h>

/* See derivation in SNES ex11.

The weak form would then be, using test function $(v, q, \tau)$,

            (q, \frac{1}{M} \frac{dp}{dt}) + (q, \alpha \frac{d\varepsilon}{dt}) + (\nabla q, \kappa \nabla p) = (q, g)
 -(\nabla v, 2 G \epsilon) - (\nabla\cdot v, \frac{2 G \nu}{1 - 2\nu} \varepsilon) + \alpha (\nabla\cdot v, p) = (v, f)
                                                                          (\tau, \nabla \cdot u - \varepsilon) = 0
*/


typedef enum {SOL_QUADRATIC, SOL_TERZAGHI, NUM_SOLUTION_TYPES} SolutionType;
const char *solutionTypes[NUM_SOLUTION_TYPES+1] = {"quadratic", "unknown"};

typedef struct {
  PetscScalar mu;    /* shear modulus */
  PetscScalar K_u;   /* undrained bulk modulus */
  PetscScalar alpha; /* Biot effective stress coefficient */
  PetscScalar M;     /* Biot modulus */
  PetscScalar k;     /* (isotropic) permeability */
  PetscScalar mu_f;  /* fluid dynamic viscosity */
  PetscScalar ymax;  /* vertical maximum extent */
  PetscScalar ymin;  /* vertical minimum extent */
  PetscScalar P_0;   /* magnitude of vertical stress */
} Parameter;

typedef struct {
  /* Domain and mesh definition */
  char         dmType[256]; /* DM type for the solve */
  PetscInt     dim;         /* The topological mesh dimension */
  PetscBool    simplex;     /* Simplicial mesh */
  /* Problem definition */
  SolutionType solType;     /* Type of exact solution */
  PetscBag     bag;         /* Problem parameters */
  /* Exact solution terms */
  PetscScalar pi;    /* ratio of a circle's circumference to its diameter */
  PetscInt    niter; /* Number of series term iterations in exact solutions */
} AppCtx;

static PetscErrorCode zero(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  PetscInt d;
  for (d = 0; d < dim; ++d) u[d] = 0.0;
  return 0;
}

static PetscErrorCode vertical_stress_2d(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  Parameter  *param;
  PetscBag       bag;
  PetscErrorCode ierr;

  ierr = PetscBagGetData(ctx->bag, (void **) &param);CHKERRQ(ierr);
  const PetscScalar P = param->P_0;

  //PetscInt d;
  //for (d = 0; d < dim; ++d) u[d] = 0.0;
  u[1] = P;
  return 0;
}

/* Quadratic Solutions */

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
  u[0] = (x[0] + x[1])*PetscCosReal(time);
  return 0;
}

static PetscErrorCode quadratic_3d_u(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  u[0] = x[0]*x[0];
  u[1] = x[1]*x[1] - 2.0*x[0]*x[1];
  u[2] = x[2]*x[2] - 2.0*x[1]*x[2];
  return 0;
}

/* Terzaghi Solutions */

// Displacement
static PetscErrorCode terzaghi_2d_u(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{

  Parameter  *param;
  PetscBag       bag;
  PetscErrorCode ierr;

  ierr = PetscBagGetData(ctx->bag, (void **) &param);CHKERRQ(ierr);

  const PetscInt NITER = ctx->niter;
  const PetscScalar PI = ctx->pi;

  const PetscScalar YMAX = param->ymax;
  const PetscScalar YMIN = param->ymin;
  const PetscScalar alpha = param->alpha;
  const PetscScalar K_u = param->K_u;
  const PetscScalar M = param->M;
  const PetscScalar G = param->mu;
  const PetscScalar k = param->k;
  const PetscScalar mu_f = param->mu_f;
  const PetscScalar P_0 = param->P_0;

  const PetscScalar K_d    = K_u - alpha*alpha*M;
  const PetscScalar nu = (3.0*K_d - 2.0*G) / (2.0*(3.0*K_d + G ));
  const PetscScalar nu_u = (3.0*K_u - 2.0*G) / (2.0*(3.0*K_u + G ));
  const PetscScalar kappa = k / mu_f;

  const PetscScalar L = YMAX - YMIN;

  const PetscScalar eta = ( alpha * (1.0 - 2.0*nu) ) / ( 2.0*(1.0 - nu) );
  const PetscScalar c = ( (2.0*kappa*G) * (1.0 - nu) ) / (1.0 - 2.0*nu);
  const PetscScalar zstar = x[1] / L; /* m / m */
  const PetscScalar tstar = (c*time) / (4.0*L*L);

  // Series Term

  PetscScalar F2 = 0.0;
  for (PetscInt m = 1; m < NITER*2+1; m++) {
    if (m%2 == 1) {
      F2 += 8.0 / (m*m * PI*PI) * cos( (m*PI*zstar) / 2.0 ) * (1.0 - exp(-1*m*m*PI*PI*tstar) );
    }
  }

  u[0] = 0.0;
  u[1] = ( ( P_0*L*(1.0 - 2.0*nu_u) ) / ( 2.0*G*(1.0-nu_u) ) ) * (1.0 - zstar) + ( ( P_0*L*(nu_u - nu) ) / ( 2.0*G*(1.0-nu_u)*(1.0-nu) ) )*F2;
  return 0;
}

// Pressure
static PetscErrorCode terzaghi_2d_p(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{

  Parameter  *param;
  PetscBag       bag;
  PetscErrorCode ierr;

  ierr = PetscBagGetData(ctx->bag, (void **) &param);CHKERRQ(ierr);

  const PetscInt NITER = ctx->niter;
  const PetscScalar PI = ctx->pi;

  const PetscScalar YMAX = param->ymax;
  const PetscScalar YMIN = param->ymin;
  const PetscScalar alpha = param->alpha;
  const PetscScalar K_u = param->K_u;
  const PetscScalar M = param->M;
  const PetscScalar G = param->mu;
  const PetscScalar k = param->k;
  const PetscScalar mu_f = param->mu_f;
  const PetscScalar P_0 = param->P_0;

  const PetscScalar K_d = K_u - alpha*alpha*M;
  const PetscScalar nu = (3.0*K_d - 2.0*G) / (2.0*(3.0*K_d + G ));
  const PetscScalar nu_u = (3.0*K_u - 2.0*G) / (2.0*(3.0*K_u + G ));
  const PetscScalar kappa = k / mu_f;
  const PetscScalar S = (3.0*K_u + 4.0*G) / (M*(3.0*K_d + 4.0*G));

  const PetscScalar L = YMAX - YMIN;
  const PetscScalar eta = ( alpha * (1.0 - 2.0*nu) ) / ( 2.0*(1.0 - nu) );
  const PetscScalar c = ( (2.0*kappa*G) * (1.0 - nu) ) / (1.0 - 2.0*nu);
  const PetscScalar zstar = x[1] / L; /* m / m */
  const PetscScalar tstar = (c*time) / (4.0*L*L);

  // Series term

  PetscScalar F1 = 0.0;
  for (PetscInt m = 1; m < NITER*2+1; m++) {
    if (m%2 == 1) {
      F1 += 4.0 / (m*PI) * sin( (m*PI*zstar)/2.0 ) * exp(-1*m*m*PI*PI*tstar);
    }
  }

  u[0] = ( (P_0*eta)/(G*S) )*F1;
  return 0;
}

// Trace strain

// Pressure
static PetscErrorCode terzaghi_2d_eps(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{

  Parameter  *param;
  PetscBag       bag;
  PetscErrorCode ierr;

  ierr = PetscBagGetData(ctx->bag, (void **) &param);CHKERRQ(ierr);

  const PetscInt NITER = ctx->niter;
  const PetscScalar PI = ctx->pi;

  const PetscScalar YMAX = param->ymax;
  const PetscScalar YMIN = param->ymin;
  const PetscScalar PI = param->pi;
  const PetscScalar alpha = param->alpha;
  const PetscScalar K_u = param->K_u;
  const PetscScalar M = param->M;
  const PetscScalar G = param->mu;
  const PetscScalar k = param->k;
  const PetscScalar mu_f = param->mu_f;
  const PetscScalar P_0 = param->P_0;

  const PetscScalar K_d = K_u - alpha*alpha*M;
  const PetscScalar nu = (3.0*K_d - 2.0*G) / (2.0*(3.0*K_d + G ));
  const PetscScalar nu_u = (3.0*K_u - 2.0*G) / (2.0*(3.0*K_u + G ));
  const PetscScalar kappa = k / mu_f;
  const PetscScalar S = (3.0*K_u + 4.0*G) / (M*(3.0*K_d + 4.0*G));

  const PetscScalar L = YMAX - YMIN;
  const PetscScalar eta = ( alpha * (1.0 - 2.0*nu) ) / ( 2.0*(1.0 - nu) );
  const PetscScalar c = ( (2.0*kappa*G) * (1.0 - nu) ) / (1.0 - 2.0*nu);
  const PetscScalar zstar = x[1] / L; /* m / m */
  const PetscScalar tstar = (c*time) / (4.0*L*L);

  PetscScalar C = ( P_0*L*(1.0 - 2.0*nu_u) ) / ( 2.0*G*(1.0 - nu_u) );
  PetscScalar D = ( P_0*L*(nu_u - nu) ) / ( 2.0*G*(1.0 - nu_u)*(1.0 - nu) );

  // Series term

  PetscScalar dF2_dzstar = 0.0;
  for (PetscInt m = 1; m < NITER*2+1; m++) {
    if (m%2 == 1) {
      dF2_dzstar += -4.0/(m*PI) * sin( (m*PI*zstar) / 2.0 ) * (1.0 - exp(-1*m*m*PI*PI*tstar) );
    }
  }

  u[0] = -1.0*C/L + (D/L)*dF2_dzstar;
  return 0;
}

/*
  u = x^2
  v = y^2 - 2xy
  p = (x + y) cos(t)
  e = 2y
  f = <2 G, 4 G + 2 \lambda >
  g = 0
  \epsilon = / 2x     -y    \
             \ -y   2y - 2x /
  Tr(\epsilon) = e = div u = 2y
  div \sigma = \partial_i 2 G \epsilon_{ij} + \partial_i \lambda \varepsilon \delta_{ij} - \partial_i \alpha p \delta_{ij}
    = 2 G < 2-1, 2 > + \lambda <0, 2> - alpha <t^2, cos(t)>
    = <2 G, 4 G + 2 \lambda> - <alpha t^2, alpha cos(t()>
  \frac{1}{M} \frac{dp}{dt} + \alpha \frac{d\varepsilon}{dt} - \nabla \cdot \kappa \nabla p
    = \frac{1}{M} \frac{dp}{dt} + \kappa \Delta p
    = -(x + y)/M sin(t)

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

/* MMS Kernels */
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

  for (d = 0; d < dim-1; ++d) f0[d] -= 2.0*G - alpha*PetscCosReal(t);
  f0[dim-1] -= 2.0*lambda + 4.0*G - alpha*PetscCosReal(t);
}

static void f0_quadratic_p(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                 const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                 const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                 PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  const PetscReal alpha  = constants[2];
  const PetscReal M      = constants[3];

  f0[0] += u_t ? alpha*u_t[uOff[1]] : 0.0;
  f0[0] += u_t ? u_t[uOff[2]]/M     : 0.0;
  f0[0] += PetscSinReal(t)*(x[0] + x[1])/M;
}

static void f0_linear_p(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                 const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                 const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                 PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  const PetscReal alpha  = constants[2];
  const PetscReal M      = constants[3];

  f0[0] += u_t ? alpha*u_t[uOff[1]] : 0.0;
  f0[0] += u_t ? u_t[uOff[2]]/M     : 0.0;
  f0[0] += PetscSinReal(t)*(x[0] + x[1])/M;
}


/* Boundary Kernels */

static void f0_terzaghi_bd_u(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                                    const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                                    const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                                    PetscReal t, const PetscReal x[], const PetscReal n[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  const PetscReal P = constants[5];

  f0[1] = P;
}


/* Standard Kernels - Residual */

static void f0_u(PetscInt dim, PetscInt Nf, PetscInt NfAux,
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

/* Standard Kernels - Jacobian */

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

static void g0_pe(PetscInt dim, PetscInt Nf, PetscInt NfAux,
           const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
           const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
           PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g0[])
{
  const PetscReal alpha  = constants[2];

  g0[0] = u_tShift*alpha;
}

static void g0_pp(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                  const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                  const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                  PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g0[])
{
  const PetscReal M = constants[3];

  g0[0] = u_tShift/M;
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
  PetscInt sol;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  options->dim     = 2;
  options->simplex = PETSC_TRUE;
  options->solType = SOL_QUADRATIC;
  options->niter   = 200;
  options->pi      = 3.14159265359;
  ierr = PetscStrncpy(options->dmType, DMPLEX, 256);CHKERRQ(ierr);

  ierr = PetscOptionsBegin(comm, "", "Biot Poroelasticity Options", "DMPLEX");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-dim", "The topological mesh dimension", "ex53.c", options->dim, &options->dim, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-niter", "Number of series term iterations in exact solutions", "ex53.c", options->niter, &options->niter, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-simplex", "Simplicial (true) or tensor (false) mesh", "ex53.c", options->simplex, &options->simplex, NULL);CHKERRQ(ierr);
  sol  = options->solType;
  ierr = PetscOptionsEList("-sol_type", "Type of exact solution", "ex53.c", solutionTypes, NUM_SOLUTION_TYPES, solutionTypes[options->solType], &sol, NULL);CHKERRQ(ierr);
  options->solType = (SolutionType) sol;
  ierr = PetscOptionsFList("-dm_type", "Convert DMPlex to another format", "ex53.c", DMList, options->dmType, options->dmType, 256, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsScalar("-pi", "ratio of a circle's circumference to its diameter", "ex53.c", options->pi, &options->pi, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode SetupParameters(AppCtx *ctx)
{
  PetscBag       bag;
  Parameter     *p;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  /* setup PETSc parameter bag */
  ierr = PetscBagGetData(ctx->bag,(void**)&p);CHKERRQ(ierr);
  ierr = PetscBagSetName(ctx->bag,"par","Poroelastic Parameters");CHKERRQ(ierr);
  bag  = ctx->bag;
  ierr = PetscBagRegisterScalar(bag, &p->mu,     1.0, "mu",    "Shear Modulus, Pa");CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &p->K_u,    1.0, "K_u",   "Undrained Bulk Modulus, Pa");CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &p->alpha,  1.0, "alpha", "Biot Effective Stress Coefficient");CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &p->M,      1.0, "M",     "Biot Modulus");CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &p->k,      1.0, "k",     "Isotropic Permeability, m**2");CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &p->mu_f,   1.0, "mu_f",  "Fluid Dynamic Viscosity, Pa*s");CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &p->ymax,   1.0, "ymax",  "Vertical Maximum Extent, m");CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &p->ymax,  -1.0, "ymin",  "Vertical Minimum Extent, m");CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &p->P_0,    1.0, "P_0",   "Magnitude of Vertical Stress, Pa");CHKERRQ(ierr);
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
    ierr = PetscDSSetResidual(prob, 2, f0_linear_p,           f1_p);CHKERRQ(ierr);
    ierr = PetscDSSetJacobian(prob, 0, 0, NULL,  NULL,  NULL,  g3_uu);CHKERRQ(ierr);
    ierr = PetscDSSetJacobian(prob, 0, 1, NULL,  NULL,  g2_ue, NULL);CHKERRQ(ierr);
    ierr = PetscDSSetJacobian(prob, 0, 2, NULL,  NULL,  g2_up, NULL);CHKERRQ(ierr);
    ierr = PetscDSSetJacobian(prob, 1, 0, NULL,  g1_eu, NULL,  NULL);CHKERRQ(ierr);
    ierr = PetscDSSetJacobian(prob, 1, 1, g0_ee, NULL,  NULL,  NULL);CHKERRQ(ierr);
    ierr = PetscDSSetJacobian(prob, 2, 1, g0_pe,  NULL,  NULL,  NULL);CHKERRQ(ierr);
    ierr = PetscDSSetJacobian(prob, 2, 2, g0_pp,  NULL,  NULL,  g3_pp);CHKERRQ(ierr);
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
  case SOL_TERZAGHI:
    ierr = PetscDSSetResidual(prob, 0, f0_u, f1_u);CHKERRQ(ierr);
    ierr = PetscDSSetResidual(prob, 1, f0_epsilon,     NULL);CHKERRQ(ierr);
    ierr = PetscDSSetResidual(prob, 2, f0_p,           f1_p);CHKERRQ(ierr);
    ierr = PetscDSSetJacobian(prob, 0, 0, NULL,  NULL,  NULL,  g3_uu);CHKERRQ(ierr);
    ierr = PetscDSSetJacobian(prob, 0, 1, NULL,  NULL,  g2_ue, NULL);CHKERRQ(ierr);
    ierr = PetscDSSetJacobian(prob, 0, 2, NULL,  NULL,  g2_up, NULL);CHKERRQ(ierr);
    ierr = PetscDSSetJacobian(prob, 1, 0, NULL,  g1_eu, NULL,  NULL);CHKERRQ(ierr);
    ierr = PetscDSSetJacobian(prob, 1, 1, g0_ee, NULL,  NULL,  NULL);CHKERRQ(ierr);
    ierr = PetscDSSetJacobian(prob, 2, 1, g0_pe,  NULL,  NULL,  NULL);CHKERRQ(ierr);
    ierr = PetscDSSetJacobian(prob, 2, 2, g0_pp,  NULL,  NULL,  g3_pp);CHKERRQ(ierr);

    ierr = PetscDSSetBdResidual(prob, 1, f0_terzaghi_bd_u, NULL);CHKERRQ(ierr);

    exact[0] = terzaghi_2d_u;
    exact[1] = terzaghi_2d_eps;
    exact[2] = terzaghi_2d_p;
    break;

  default: SETERRQ2(PetscObjectComm((PetscObject) prob), PETSC_ERR_ARG_WRONG, "Invalid solution type: %s (%D)", solutionTypes[PetscMin(user->solType, NUM_SOLUTION_TYPES)], user->solType);
  }

  ierr = PetscDSSetExactSolution(prob, 0, exact[0], user);CHKERRQ(ierr);
  ierr = PetscDSSetExactSolution(prob, 1, exact[1], user);CHKERRQ(ierr);
  ierr = PetscDSSetExactSolution(prob, 2, exact[2], user);CHKERRQ(ierr);

  /* Setup Boundary Conditions */
  if (user->solType == SOL_TERZAGHI) {
    id = 3;
    ierr = DMAddBoundary(dm, DM_BC_NATURAL, "vertical stress", "marker", 0, 1, 1, (void (*)(void)) vertical_stress_2d, 1, &id, user);CHKERRQ(ierr);
    ierr = DMAddBoundary(dm, DM_BC_ESSENTIAL, "drained surface", "marker", 2, 0, NULL, (void (*)(void)) zero, 1, &id, user);CHKERRQ(ierr);
    id = 1;
    ierr = DMAddBoundary(dm, DM_BC_ESSENTIAL, "fixed base", "marker", 0, 1, 1, (void (*)(void)) zero, 1, &id, user);CHKERRQ(ierr);

  } else {
    id = 1;
    ierr = DMAddBoundary(dm, DM_BC_ESSENTIAL, "wall displacement", "marker", 0, 0, NULL, (void (*)(void)) exact[0], 1, &id, user);CHKERRQ(ierr);
    ierr = DMAddBoundary(dm, DM_BC_ESSENTIAL, "wall pressure",     "marker", 2, 0, NULL, (void (*)(void)) exact[2], 1, &id, user);CHKERRQ(ierr);
  }

  /* Setup constants */
  {
    Parameter  *param;
    PetscScalar constants[5];

    ierr = PetscBagGetData(user->bag, (void **) &param);CHKERRQ(ierr);

    constants[0] = param->mu;            /* shear modulus */
    constants[1] = param->K_u;           /* undrained bulk modulus */
    constants[2] = param->alpha;         /* Biot effective stress coefficient */
    constants[3] = param->M;             /* Biot modulus */
    constants[4] = param->k/param->mu_f; /* Darcy coefficient */
    constants[5] = param->P_0;           /* Magnitude of Vertical Stress */
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
    ierr = PetscFECreateDefault(PetscObjectComm((PetscObject) dm), dim, Nc[f], simplex, name[f] ? prefix : NULL, -1, &fe);CHKERRQ(ierr);
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

static PetscErrorCode SetInitialConditions(TS ts, Vec u)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecSet(u, 0.0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  AppCtx         ctx;       /* User-defined work context */
  DM             dm;        /* Problem specification */
  TS             ts;        /* Time Series / Nonlinear solver */
  Vec            u;         /* Solutions */
  const char    *name[3] = {"displacement", "tracestrain", "pressure"};
  PetscInt       Nc[3];
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL, help);if (ierr) return ierr;
  ierr = ProcessOptions(PETSC_COMM_WORLD, &ctx);CHKERRQ(ierr);
  ierr = PetscBagCreate(PETSC_COMM_SELF, sizeof(Parameter), &ctx.bag);CHKERRQ(ierr);
  ierr = SetupParameters(&ctx);CHKERRQ(ierr);
  /* Primal System */
  ierr = TSCreate(PETSC_COMM_WORLD, &ts);CHKERRQ(ierr);
  ierr = CreateMesh(PETSC_COMM_WORLD, &ctx, &dm);CHKERRQ(ierr);
  ierr = DMSetApplicationContext(dm, &ctx);CHKERRQ(ierr);
  ierr = TSSetDM(ts, dm);CHKERRQ(ierr);

  Nc[0] = ctx.dim;
  Nc[1] = 1;
  Nc[2] = 1;

  ierr = SetupFE(dm, ctx.simplex, 3, Nc, name, SetupPrimalProblem, &ctx);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(dm, &u);CHKERRQ(ierr);
  ierr = VecSet(u, 0.0);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) u, "solution");CHKERRQ(ierr);

  ierr = DMTSSetBoundaryLocal(dm, DMPlexTSComputeBoundary, &ctx);CHKERRQ(ierr);
  ierr = DMTSSetIFunctionLocal(dm, DMPlexTSComputeIFunctionFEM, &ctx);CHKERRQ(ierr);
  ierr = DMTSSetIJacobianLocal(dm, DMPlexTSComputeIJacobianFEM, &ctx);CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(ts, TS_EXACTFINALTIME_STEPOVER);CHKERRQ(ierr);
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);
  ierr = DMTSCheckFromOptions(ts, u, NULL, NULL);CHKERRQ(ierr);
  ierr = TSSetComputeInitialCondition(ts, SetInitialConditions);CHKERRQ(ierr);

  ierr = TSSolve(ts, u);CHKERRQ(ierr);
  ierr = TSGetSolution(ts, &u);CHKERRQ(ierr);
  ierr = VecViewFromOptions(u, NULL, "-sol_view");CHKERRQ(ierr);

  /* Cleanup */
  ierr = VecDestroy(&u);CHKERRQ(ierr);
  ierr = TSDestroy(&ts);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = PetscBagDestroy(&ctx.bag);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}


/*TEST

  test:
    suffix: 2d_p1_quad
    requires: triangle
    args: -displacement_petscspace_degree 2 -tracestrain_petscspace_degree 1 -pressure_petscspace_degree 1 -dm_refine 2 -dmts_check .0001 -ts_max_steps 5

  test:
    suffix: 2d_p1_quad_tconv
    requires: triangle
    args: -displacement_petscspace_degree 2 -tracestrain_petscspace_degree 1 -pressure_petscspace_degree 1 -dm_refine 2 -convest_num_refine 3 -ts_convergence_estimate -ts_max_steps 5

TEST*/
