# distutils: language = c++
# distutils: sources = correspondPixels.cc

from libc.stdlib cimport free
from cpython cimport PyObject, Py_INCREF
import cython, time

import numpy as np
cimport numpy as np
np.import_array()

cdef extern from "edge_eval.hh" namespace "edge_eval":
    cdef struct s_returnVal:
        double* out1
        double* out2
        double cost
        double oc

    ctypedef s_returnVal returnVal

    cdef cppclass Eval:
        returnVal correspondPixels(double* bmap1, double* bmap2, int rows, int cols,
    double maxDist, double outlierCost)
        returnVal correspondPixels(double* bmap1, double* bmap2, int rows, int cols,
    double maxDist)
        returnVal correspondPixels(double* bmap1, double* bmap2, int rows, int cols)

cdef class EdgeEval:

    cdef Eval *_thisptr

    def __cinit__(self):
        self._thisptr = new Eval()
        if self._thisptr == NULL:
            raise MemoryError()

    def __dealloc__(self):
        if self._thisptr != NULL:
            del self._thisptr

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef double* numpyToC(self, np.ndarray[double, ndim=2, mode="c"] input):
        return &input[0,0]

    cpdef double getCost(self, bmap1, bmap2):
        assert bmap1.shape == bmap2.shape
        rows, cols = bmap1.shape
        # bmap1, bmap2 = bmap1.flatten(), bmap2.flatten()
        cdef N = rows * cols
        cdef double* bmap1_c = self.numpyToC(bmap1)
        cdef double* bmap2_c = self.numpyToC(bmap2)
        result = self._thisptr.correspondPixels(bmap1_c, bmap2_c, rows, cols).cost
        # free (bmap1_c)
        # free (bmap2_c)
        return result

    cpdef double getOutlierCost(self, bmap1, bmap2):
        assert bmap1.shape == bmap2.shape
        rows, cols = bmap1.shape
        cdef N = rows * cols
        cdef double* bmap1_c = self.numpyToC(bmap1)
        cdef double* bmap2_c = self.numpyToC(bmap2)
        result = self._thisptr.correspondPixels(bmap1_c, bmap2_c, rows, cols).oc
        # free (bmap1_c)
        # free (bmap2_c)
        return result

    cpdef np.ndarray[double, mode="c"] getOut1(self, bmap1, bmap2):
        assert bmap1.shape == bmap2.shape
        rows, cols = bmap1.shape
        cdef N = rows * cols
        cdef double* bmap1_c = self.numpyToC(bmap1)
        cdef double* bmap2_c = self.numpyToC(bmap2)
        cdef result = np.ndarray(dtype=np.double, shape=bmap1.shape)
        result_c = self._thisptr.correspondPixels(bmap1_c, bmap2_c, rows, cols).out1
        cdef np.npy_intp dims[2]
        dims[0] = rows
        dims[1] = cols
        result = np.PyArray_SimpleNewFromData(2, dims, np.NPY_DOUBLE, <void*> result_c)
        return result

    cpdef np.ndarray[double, mode="c"] getOut2(self, bmap1, bmap2):
        assert bmap1.shape == bmap2.shape
        rows, cols = bmap2.shape
        cdef N = rows * cols
        cdef double* bmap1_c = self.numpyToC(bmap1)
        cdef double* bmap2_c = self.numpyToC(bmap2)
        cdef result = np.ndarray(dtype=np.double, shape=bmap2.shape)
        result_c = self._thisptr.correspondPixels(bmap1_c, bmap2_c, rows, cols).out2
        cdef np.npy_intp dims[2]
        dims[0] = rows
        dims[1] = cols
        result = np.PyArray_SimpleNewFromData(2, dims, np.NPY_DOUBLE, <void*> result_c)
        return result
