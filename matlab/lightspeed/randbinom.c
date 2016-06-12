/* compile with: 
      Windows: mex randbinom.c util.obj
      Others:  cmex randbinom.c util.o -lm
 */
#include "mexutil.h"
#include "util.h"

void mexFunction(int nlhs, mxArray *plhs[],
		 int nrhs, const mxArray *prhs[])
{
  mwSize ndims, *dims, pCount, nCount, i,j;
	mwSize ndims2, *dims2, noutdims, *outdims;
  double *pdata, *outdata, *ndata;
	mwSize n;

  if(nrhs != 2)
    mexErrMsgTxt("Usage: r = bino_sample(p, n)");

  /* prhs[0] is first argument.
   * mxGetPr returns double*  (data, col-major)
   */
  ndims = mxGetNumberOfDimensions(prhs[0]);
  dims = (mwSize*)mxGetDimensions(prhs[0]);
  pCount = mxGetNumberOfElements(prhs[0]);

  ndims2 = mxGetNumberOfDimensions(prhs[1]);
  dims2 = (mwSize*)mxGetDimensions(prhs[1]);
	ndata = mxGetPr(prhs[1]);
  nCount = mxGetNumberOfElements(prhs[1]);

  /* plhs[0] is first output */
	if(nCount == 1) {
		noutdims = ndims;
		outdims = dims;
	} else {
    /* count the non-unit dimensions of p and n */
    noutdims = 0;
    for(i=0;i<ndims;i++) if(dims[i] > 1) noutdims++;
    for(i=0;i<ndims2;i++) if(dims2[i] > 1) noutdims++;
		outdims = (mwSize*)mxCalloc(noutdims, sizeof(mwSize));
    j = 0;
    for(i=0;i<ndims;i++) if(dims[i] > 1) outdims[j++] = dims[i];
    for(i=0;i<ndims2;i++) if(dims2[i] > 1) outdims[j++] = dims2[i];
	}
  plhs[0] = mxCreateNumericArrayE(noutdims, outdims, mxDOUBLE_CLASS, mxREAL);
  outdata = mxGetPr(plhs[0]);
  if(mxIsSparse(prhs[0]) || mxIsSparse(prhs[1]))
    mexErrMsgTxt("Cannot handle sparse matrices.  Sorry.");

	for(j=0;j<nCount;j++) {
		double nDouble = *ndata++;
		n = (mwSize)nDouble;
		if((double)n != nDouble)
			mexErrMsgTxt("n is not integer or out of 64-bit range");
		pdata = mxGetPr(prhs[0]);
		for(i=0;i<pCount;i++) {
			*outdata++ = BinoRand(*pdata++, n);
		}
	}
}

