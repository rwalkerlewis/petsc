/* $Id: dense.h,v 1.9 2000/10/24 20:25:29 bsmith Exp balay $ */

#include "src/mat/matimpl.h"

#if !defined(__DENSE_H)
#define __DENSE_H

/*  
  MATSEQDENSE format - conventional dense Fortran storage (by columns)
*/

typedef struct {
  PetscScalar  *v;                /* matrix elements */
  PetscTruth   roworiented;       /* if true, row oriented input (default) */
  int          pad;               /* padding */        
  int          *pivots;           /* pivots in LU factorization */
  PetscTruth   user_alloc;        /* true if the user provided the dense data */
} Mat_SeqDense;

EXTERN int MatMult_SeqDense(Mat A,Vec,Vec);
EXTERN int MatMultAdd_SeqDense(Mat A,Vec,Vec,Vec);
EXTERN int MatMultTranspose_SeqDense(Mat A,Vec,Vec);
EXTERN int MatMultTransposeAdd_SeqDense(Mat A,Vec,Vec,Vec);

#endif
