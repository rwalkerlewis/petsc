/*$Id: mmdense.c,v 1.38 2001/08/06 21:15:13 bsmith Exp balay $*/

/*
   Support for the parallel dense matrix vector multiply
*/
#include "src/mat/impls/dense/mpi/mpidense.h"
#include "src/vec/vecimpl.h"

#undef __FUNCT__  
#define __FUNCT__ "MatSetUpMultiply_MPIDense"
int MatSetUpMultiply_MPIDense(Mat mat)
{
  Mat_MPIDense *mdn = (Mat_MPIDense*)mat->data;
  int          ierr;
  IS           from,to;
  Vec          gvec;

  PetscFunctionBegin;
  /* Create local vector that is used to scatter into */
  ierr = VecCreateSeq(PETSC_COMM_SELF,mat->N,&mdn->lvec);CHKERRQ(ierr);

  /* Create temporary index set for building scatter gather */
  ierr = ISCreateStride(mat->comm,mat->N,0,1,&from);CHKERRQ(ierr);
  ierr = ISCreateStride(PETSC_COMM_SELF,mat->N,0,1,&to);CHKERRQ(ierr);

  /* Create temporary global vector to generate scatter context */
  /* n    = mdn->cowners[mdn->rank+1] - mdn->cowners[mdn->rank]; */

  ierr = VecCreateMPI(mat->comm,mdn->nvec,mat->N,&gvec);CHKERRQ(ierr);

  /* Generate the scatter context */
  ierr = VecScatterCreate(gvec,from,mdn->lvec,to,&mdn->Mvctx);CHKERRQ(ierr);
  PetscLogObjectParent(mat,mdn->Mvctx);
  PetscLogObjectParent(mat,mdn->lvec);
  PetscLogObjectParent(mat,from);
  PetscLogObjectParent(mat,to);
  PetscLogObjectParent(mat,gvec);

  ierr = ISDestroy(to);CHKERRQ(ierr);
  ierr = ISDestroy(from);CHKERRQ(ierr);
  ierr = VecDestroy(gvec);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN int MatGetSubMatrices_MPIDense_Local(Mat,int,IS*,IS*,MatReuse,Mat*);
#undef __FUNCT__  
#define __FUNCT__ "MatGetSubMatrices_MPIDense" 
int MatGetSubMatrices_MPIDense(Mat C,int ismax,IS *isrow,IS *iscol,MatReuse scall,Mat **submat)
{ 
  int           nmax,nstages_local,nstages,i,pos,max_no,ierr;

  PetscFunctionBegin;
  /* Allocate memory to hold all the submatrices */
  if (scall != MAT_REUSE_MATRIX) {
    ierr = PetscMalloc((ismax+1)*sizeof(Mat),submat);CHKERRQ(ierr);
  }
  /* Determine the number of stages through which submatrices are done */
  nmax          = 20*1000000 / (C->N * sizeof(int));
  if (!nmax) nmax = 1;
  nstages_local = ismax/nmax + ((ismax % nmax)?1:0);

  /* Make sure every processor loops through the nstages */
  ierr = MPI_Allreduce(&nstages_local,&nstages,1,MPI_INT,MPI_MAX,C->comm);CHKERRQ(ierr);


  for (i=0,pos=0; i<nstages; i++) {
    if (pos+nmax <= ismax) max_no = nmax;
    else if (pos == ismax) max_no = 0;
    else                   max_no = ismax-pos;
    ierr = MatGetSubMatrices_MPIDense_Local(C,max_no,isrow+pos,iscol+pos,scall,*submat+pos);CHKERRQ(ierr);
    pos += max_no;
  }
  PetscFunctionReturn(0);
}
/* -------------------------------------------------------------------------*/
#undef __FUNCT__  
#define __FUNCT__ "MatGetSubMatrices_MPIDense_Local" 
int MatGetSubMatrices_MPIDense_Local(Mat C,int ismax,IS *isrow,IS *iscol,MatReuse scall,Mat *submats)
{ 
  Mat_MPIDense  *c = (Mat_MPIDense*)C->data;
  Mat           A = c->A;
  Mat_SeqDense  *a = (Mat_SeqDense*)A->data,*mat;
  int           N = C->N,rstart = c->rstart,count;
  int           **irow,**icol,*nrow,*ncol,*w1,*w2,*w3,*w4,*rtable,start,end,size;
  int           **sbuf1,rank,m,i,j,k,l,ct1,ierr,**rbuf1,row,proc;
  int           nrqs,msz,**ptr,index,*ctr,*pa,*tmp,bsz,nrqr;
  int           is_no,jmax,*irow_i,**rmap,*rmap_i;
  int           len,ctr_j,*sbuf1_j,*rbuf1_i;
  int           tag0,tag1;
  MPI_Request   *s_waits1,*r_waits1,*s_waits2,*r_waits2;
  MPI_Status    *r_status1,*r_status2,*s_status1,*s_status2;
  MPI_Comm      comm;
  PetscScalar   **rbuf2,**sbuf2;

  PetscFunctionBegin;
  comm   = C->comm;
  tag0   = C->tag;
  size   = c->size;
  rank   = c->rank;
  m      = C->M;
  
  /* Get some new tags to keep the communication clean */
  ierr = PetscObjectGetNewTag((PetscObject)C,&tag1);CHKERRQ(ierr);

    /* Check if the col indices are sorted */
  for (i=0; i<ismax; i++) {
    ierr = ISSorted(isrow[i],(PetscTruth*)&j);CHKERRQ(ierr);
    if (!j) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"ISrow is not sorted");
    ierr = ISSorted(iscol[i],(PetscTruth*)&j);CHKERRQ(ierr);
    if (!j) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"IScol is not sorted");
  }

  len    =  2*ismax*(sizeof(int *)+sizeof(int)) + (m+1)*sizeof(int);
  ierr   = PetscMalloc(len,&irow);CHKERRQ(ierr);
  icol   = irow + ismax;
  nrow   = (int*)(icol + ismax);
  ncol   = nrow + ismax;
  rtable = ncol + ismax;

  for (i=0; i<ismax; i++) { 
    ierr = ISGetIndices(isrow[i],&irow[i]);CHKERRQ(ierr);
    ierr = ISGetIndices(iscol[i],&icol[i]);CHKERRQ(ierr);
    ierr = ISGetLocalSize(isrow[i],&nrow[i]);CHKERRQ(ierr);
    ierr = ISGetLocalSize(iscol[i],&ncol[i]);CHKERRQ(ierr);
  }

  /* Create hash table for the mapping :row -> proc*/
  for (i=0,j=0; i<size; i++) {
    jmax = c->rowners[i+1];
    for (; j<jmax; j++) {
      rtable[j] = i;
    }
  }

  /* evaluate communication - mesg to who,length of mesg, and buffer space
     required. Based on this, buffers are allocated, and data copied into them*/
  ierr   = PetscMalloc(size*4*sizeof(int),&w1);CHKERRQ(ierr); /* mesg size */
  w2     = w1 + size;      /* if w2[i] marked, then a message to proc i*/
  w3     = w2 + size;      /* no of IS that needs to be sent to proc i */
  w4     = w3 + size;      /* temp work space used in determining w1, w2, w3 */
  ierr = PetscMemzero(w1,size*3*sizeof(int));CHKERRQ(ierr); /* initialize work vector*/
  for (i=0; i<ismax; i++) { 
    ierr   = PetscMemzero(w4,size*sizeof(int));CHKERRQ(ierr); /* initialize work vector*/
    jmax   = nrow[i];
    irow_i = irow[i];
    for (j=0; j<jmax; j++) {
      row  = irow_i[j];
      proc = rtable[row];
      w4[proc]++;
    }
    for (j=0; j<size; j++) { 
      if (w4[j]) { w1[j] += w4[j];  w3[j]++;} 
    }
  }
  
  nrqs     = 0;              /* no of outgoing messages */
  msz      = 0;              /* total mesg length (for all procs) */
  w1[rank] = 0;              /* no mesg sent to self */
  w3[rank] = 0;
  for (i=0; i<size; i++) {
    if (w1[i])  { w2[i] = 1; nrqs++;} /* there exists a message to proc i */
  }
  ierr = PetscMalloc((nrqs+1)*sizeof(int),&pa);CHKERRQ(ierr); /*(proc -array)*/
  for (i=0,j=0; i<size; i++) {
    if (w1[i]) { pa[j] = i; j++; }
  } 

  /* Each message would have a header = 1 + 2*(no of IS) + data */
  for (i=0; i<nrqs; i++) {
    j     = pa[i];
    w1[j] += w2[j] + 2* w3[j];   
    msz   += w1[j];  
  }
  /* Do a global reduction to determine how many messages to expect*/
  {
    int *rw1;
    ierr  = PetscMalloc(2*size*sizeof(int),&rw1);CHKERRQ(ierr);
    ierr  = MPI_Allreduce(w1,rw1,2*size,MPI_INT,PetscMaxSum_Op,comm);CHKERRQ(ierr);
    bsz   = rw1[rank];
    nrqr  = rw1[size+rank];
    ierr  = PetscFree(rw1);CHKERRQ(ierr);
  }

  /* Allocate memory for recv buffers . Prob none if nrqr = 0 ???? */ 
  len      = (nrqr+1)*sizeof(int*) + nrqr*bsz*sizeof(int);
  ierr     = PetscMalloc(len,&rbuf1);CHKERRQ(ierr);
  rbuf1[0] = (int*)(rbuf1 + nrqr);
  for (i=1; i<nrqr; ++i) rbuf1[i] = rbuf1[i-1] + bsz;
  
  /* Post the receives */
  ierr = PetscMalloc((nrqr+1)*sizeof(MPI_Request),&r_waits1);CHKERRQ(ierr);
  for (i=0; i<nrqr; ++i) {
    ierr = MPI_Irecv(rbuf1[i],bsz,MPI_INT,MPI_ANY_SOURCE,tag0,comm,r_waits1+i);CHKERRQ(ierr);
  }

  /* Allocate Memory for outgoing messages */
  len   = 2*size*sizeof(int*) + 2*msz*sizeof(int)+ size*sizeof(int);
  ierr  = PetscMalloc(len,&sbuf1);CHKERRQ(ierr);
  ptr   = sbuf1 + size;   /* Pointers to the data in outgoing buffers */
  ierr  = PetscMemzero(sbuf1,2*size*sizeof(int*));CHKERRQ(ierr);
  /* allocate memory for outgoing data + buf to receive the first reply */
  tmp   = (int*)(ptr + size);
  ctr   = tmp + 2*msz;

  {
    int *iptr = tmp,ict = 0;
    for (i=0; i<nrqs; i++) {
      j         = pa[i];
      iptr     += ict;
      sbuf1[j]  = iptr;
      ict       = w1[j];
    }
  }

  /* Form the outgoing messages */
  /* Initialize the header space */
  for (i=0; i<nrqs; i++) {
    j           = pa[i];
    sbuf1[j][0] = 0;
    ierr        = PetscMemzero(sbuf1[j]+1,2*w3[j]*sizeof(int));CHKERRQ(ierr);
    ptr[j]      = sbuf1[j] + 2*w3[j] + 1;
  }
  
  /* Parse the isrow and copy data into outbuf */
  for (i=0; i<ismax; i++) {
    ierr = PetscMemzero(ctr,size*sizeof(int));CHKERRQ(ierr);
    irow_i = irow[i];
    jmax   = nrow[i];
    for (j=0; j<jmax; j++) {  /* parse the indices of each IS */
      row  = irow_i[j];
      proc = rtable[row];
      if (proc != rank) { /* copy to the outgoing buf*/
        ctr[proc]++;
        *ptr[proc] = row;
        ptr[proc]++;
      }
    }
    /* Update the headers for the current IS */
    for (j=0; j<size; j++) { /* Can Optimise this loop too */
      if ((ctr_j = ctr[j])) {
        sbuf1_j        = sbuf1[j];
        k              = ++sbuf1_j[0];
        sbuf1_j[2*k]   = ctr_j;
        sbuf1_j[2*k-1] = i;
      }
    }
  }

  /*  Now  post the sends */
  ierr = PetscMalloc((nrqs+1)*sizeof(MPI_Request),&s_waits1);CHKERRQ(ierr);
  for (i=0; i<nrqs; ++i) {
    j = pa[i];
    ierr = MPI_Isend(sbuf1[j],w1[j],MPI_INT,j,tag0,comm,s_waits1+i);CHKERRQ(ierr);
  }

  /* Post recieves to capture the row_data from other procs */
  ierr  = PetscMalloc((nrqs+1)*sizeof(MPI_Request),&r_waits2);CHKERRQ(ierr);
  ierr  = PetscMalloc((nrqs+1)*sizeof(PetscScalar*),&rbuf2);CHKERRQ(ierr);
  for (i=0; i<nrqs; i++) {
    j        = pa[i];
    count    = (w1[j] - (2*sbuf1[j][0] + 1))*N;
    ierr     = PetscMalloc((count+1)*sizeof(PetscScalar),&rbuf2[i]);CHKERRQ(ierr);
    ierr     = MPI_Irecv(rbuf2[i],count,MPIU_SCALAR,j,tag1,comm,r_waits2+i);CHKERRQ(ierr);
  }

  /* Receive messages(row_nos) and then, pack and send off the rowvalues
     to the correct processors */

  ierr = PetscMalloc((nrqr+1)*sizeof(MPI_Request),&s_waits2);CHKERRQ(ierr);
  ierr = PetscMalloc((nrqr+1)*sizeof(MPI_Status),&r_status1);CHKERRQ(ierr);
  ierr = PetscMalloc((nrqr+1)*sizeof(PetscScalar*),&sbuf2);CHKERRQ(ierr);
 
  {
    PetscScalar *sbuf2_i,*v_start;
    int    s_proc;
    for (i=0; i<nrqr; ++i) {
      ierr = MPI_Waitany(nrqr,r_waits1,&index,r_status1+i);CHKERRQ(ierr);
      s_proc          = r_status1[i].MPI_SOURCE; /* send processor */
      rbuf1_i         = rbuf1[index]; /* Actual message from s_proc */
      /* no of rows = end - start; since start is array index[], 0index, whel end
         is length of the buffer - which is 1index */
      start           = 2*rbuf1_i[0] + 1;
      ierr            = MPI_Get_count(r_status1+i,MPI_INT,&end);CHKERRQ(ierr);
      /* allocate memory sufficinet to hold all the row values */
      ierr = PetscMalloc((end-start)*N*sizeof(PetscScalar),&sbuf2[index]);CHKERRQ(ierr);
      sbuf2_i      = sbuf2[index];
      /* Now pack the data */
      for (j=start; j<end; j++) {
        row = rbuf1_i[j] - rstart;
        v_start = a->v + row;
        for (k=0; k<N; k++) {
          sbuf2_i[0] = v_start[0];
          sbuf2_i++; v_start += C->m;
        }
      }
      /* Now send off the data */
      ierr = MPI_Isend(sbuf2[index],(end-start)*N,MPIU_SCALAR,s_proc,tag1,comm,s_waits2+i);CHKERRQ(ierr);
    }
  }
  /* End Send-Recv of IS + row_numbers */
  ierr = PetscFree(r_status1);CHKERRQ(ierr);
  ierr = PetscFree(r_waits1);CHKERRQ(ierr);
  ierr = PetscMalloc((nrqs+1)*sizeof(MPI_Status),&s_status1);CHKERRQ(ierr);
  ierr      = MPI_Waitall(nrqs,s_waits1,s_status1);CHKERRQ(ierr);
  ierr = PetscFree(s_status1);CHKERRQ(ierr);
  ierr = PetscFree(s_waits1);CHKERRQ(ierr);

  /* Create the submatrices */
  if (scall == MAT_REUSE_MATRIX) {
    for (i=0; i<ismax; i++) {
      mat = (Mat_SeqDense *)(submats[i]->data);
      if ((submats[i]->m != nrow[i]) || (submats[i]->n != ncol[i])) {
        SETERRQ(PETSC_ERR_ARG_SIZ,"Cannot reuse matrix. wrong size");
      }
      ierr = PetscMemzero(mat->v,submats[i]->m*submats[i]->n*sizeof(PetscScalar));CHKERRQ(ierr);
      submats[i]->factor = C->factor;
    }
  } else {
    for (i=0; i<ismax; i++) {
      ierr = MatCreateSeqDense(PETSC_COMM_SELF,nrow[i],ncol[i],PETSC_NULL,submats+i);CHKERRQ(ierr);
    }
  }
  
  /* Assemble the matrices */
  {
    int    col;
    PetscScalar *imat_v,*mat_v,*imat_vi,*mat_vi;
  
    for (i=0; i<ismax; i++) {
      mat       = (Mat_SeqDense*)submats[i]->data;
      mat_v     = a->v;
      imat_v    = mat->v;
      irow_i    = irow[i];
      m         = nrow[i];
      for (j=0; j<m; j++) {
        row      = irow_i[j] ;
        proc     = rtable[row];
        if (proc == rank) {
          row      = row - rstart;
          mat_vi   = mat_v + row;
          imat_vi  = imat_v + j;
          for (k=0; k<ncol[i]; k++) {
            col = icol[i][k];
            imat_vi[k*m] = mat_vi[col*C->m];
          }
        } 
      }
    }
  }

  /* Create row map. This maps c->row to submat->row for each submat*/
  /* this is a very expensive operation wrt memory usage */
  len     = (1+ismax)*sizeof(int*)+ ismax*C->M*sizeof(int);
  ierr    = PetscMalloc(len,&rmap);CHKERRQ(ierr);
  rmap[0] = (int *)(rmap + ismax);
  ierr    = PetscMemzero(rmap[0],ismax*C->M*sizeof(int));CHKERRQ(ierr);
  for (i=1; i<ismax; i++) { rmap[i] = rmap[i-1] + C->M;}
  for (i=0; i<ismax; i++) {
    rmap_i = rmap[i];
    irow_i = irow[i];
    jmax   = nrow[i];
    for (j=0; j<jmax; j++) { 
      rmap_i[irow_i[j]] = j; 
    }
  }
 
  /* Now Receive the row_values and assemble the rest of the matrix */
  ierr = PetscMalloc((nrqs+1)*sizeof(MPI_Status),&r_status2);CHKERRQ(ierr);

  {
    int    is_max,tmp1,col,*sbuf1_i,is_sz;
    PetscScalar *rbuf2_i,*imat_v,*imat_vi;
  
    for (tmp1=0; tmp1<nrqs; tmp1++) { /* For each message */
      ierr = MPI_Waitany(nrqs,r_waits2,&i,r_status2+tmp1);CHKERRQ(ierr);
      /* Now dig out the corresponding sbuf1, which contains the IS data_structure */
      sbuf1_i = sbuf1[pa[i]];
      is_max  = sbuf1_i[0];
      ct1     = 2*is_max+1;
      rbuf2_i = rbuf2[i];
      for (j=1; j<=is_max; j++) { /* For each IS belonging to the message */
        is_no     = sbuf1_i[2*j-1];
        is_sz     = sbuf1_i[2*j];
        mat       = (Mat_SeqDense*)submats[is_no]->data;
        imat_v    = mat->v;
        rmap_i    = rmap[is_no];
        m         = nrow[is_no];
        for (k=0; k<is_sz; k++,rbuf2_i+=N) {  /* For each row */
          row      = sbuf1_i[ct1]; ct1++;
          row      = rmap_i[row];
          imat_vi  = imat_v + row;
          for (l=0; l<ncol[is_no]; l++) { /* For each col */
            col = icol[is_no][l];
            imat_vi[l*m] = rbuf2_i[col];
          }
        }
      }
    }
  }
  /* End Send-Recv of row_values */
  ierr = PetscFree(r_status2);CHKERRQ(ierr);
  ierr = PetscFree(r_waits2);CHKERRQ(ierr);
  ierr = PetscMalloc((nrqr+1)*sizeof(MPI_Status),&s_status2);CHKERRQ(ierr);
  ierr = MPI_Waitall(nrqr,s_waits2,s_status2);CHKERRQ(ierr);
  ierr = PetscFree(s_status2);CHKERRQ(ierr);
  ierr = PetscFree(s_waits2);CHKERRQ(ierr);

  /* Restore the indices */
  for (i=0; i<ismax; i++) {
    ierr = ISRestoreIndices(isrow[i],irow+i);CHKERRQ(ierr);
    ierr = ISRestoreIndices(iscol[i],icol+i);CHKERRQ(ierr);
  }

  /* Destroy allocated memory */
  ierr = PetscFree(irow);CHKERRQ(ierr);
  ierr = PetscFree(w1);CHKERRQ(ierr);
  ierr = PetscFree(pa);CHKERRQ(ierr);


  for (i=0; i<nrqs; ++i) {
    ierr = PetscFree(rbuf2[i]);CHKERRQ(ierr);
  }
  ierr = PetscFree(rbuf2);CHKERRQ(ierr);
  ierr = PetscFree(sbuf1);CHKERRQ(ierr);
  ierr = PetscFree(rbuf1);CHKERRQ(ierr);

  for (i=0; i<nrqr; ++i) {
    ierr = PetscFree(sbuf2[i]);CHKERRQ(ierr);
  }

  ierr = PetscFree(sbuf2);CHKERRQ(ierr);
  ierr = PetscFree(rmap);CHKERRQ(ierr);

  for (i=0; i<ismax; i++) {
    ierr = MatAssemblyBegin(submats[i],MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(submats[i],MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

