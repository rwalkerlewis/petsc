/*$Id: ex14.c,v 1.10 2001/03/23 23:20:59 balay Exp balay $*/

/* 
   Tests PetscOptionsGetScalar() for complex numbers
 */

#include "petsc.h"


#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  int    ierr;
  PetscScalar a;

  PetscInitialize(&argc,&argv,(char *)0,0);
  ierr = PetscOptionsGetScalar(PETSC_NULL,"-a",&a,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_SELF,"Scalar a = %g + %gi\n",PetscRealPart(a),PetscImaginaryPart(a));CHKERRQ(ierr);
  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}
 
