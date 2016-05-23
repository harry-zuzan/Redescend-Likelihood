import pywt
import numpy 
cimport numpy

from libc.math cimport exp, sqrt
from libc.math cimport M_PI

# Using gaussian distributions

def redescend_residuals_normal_2d(
		numpy.ndarray[numpy.float64_t,ndim=2] resids, double cval):
	return get_redescend_weights_normal_2d(resids, cval)*resids


def redescend_residuals_normal_1d(
		numpy.ndarray[numpy.float64_t,ndim=2] resids, double cval):
	return get_redescend_weights_normal_1d(resids, cval)*resids


def get_redescend_weights_normal_2d(
		numpy.ndarray[numpy.float64_t,ndim=2] resids, double cval):

	cdef double stdev = resids.std()

	cdef numpy.ndarray[numpy.float64_t,ndim=2] prob_vec1 = \
			prob_normal_vec_2d(resids,stdev)

	cdef numpy.ndarray[numpy.float64_t,ndim=2] prob_vec2 = \
			prob_normal_vec_2d(resids,cval*stdev)

	cdef numpy.ndarray[numpy.float64_t,ndim=2] likelihood_weights = \
		prob_vec1/(prob_vec1 + prob_vec2)

	cdef double prob1 = prob_normal_scalar(0.0, stdev)
	cdef double prob2 = prob_normal_scalar(0.0, cval*stdev)

	likelihood_weights /= prob1/(prob1 + prob2)

	return likelihood_weights



def get_redescend_weights_normal_1d(
		numpy.ndarray[numpy.float64_t,ndim=1] resids, double cval):

	cdef double stdev = resids.std()

	cdef numpy.ndarray[numpy.float64_t,ndim=1] prob_vec1 = \
			prob_normal_vec_1d(resids,stdev)

	cdef numpy.ndarray[numpy.float64_t,ndim=1] prob_vec2 = \
			prob_normal_vec_1d(resids,cval*stdev)

	cdef numpy.ndarray[numpy.float64_t,ndim=1] likelihood_weights = \
		prob_vec1/(prob_vec1 + prob_vec2)

	cdef double prob1 = prob_normal_scalar(0.0, stdev)
	cdef double prob2 = prob_normal_scalar(0.0, cval*stdev)

	likelihood_weights /= prob1/(prob1 + prob2)

	return likelihood_weights



cdef numpy.ndarray[numpy.float64_t,ndim=2] \
	prob_normal_vec_2d(numpy.ndarray[numpy.float64_t,ndim=2] vec, double s):

	cdef numpy.ndarray[numpy.float64_t,ndim=2] prob_vec = \
			numpy.exp(-0.5*(vec/s)**2)

	return prob_vec/(s*sqrt(2.0*M_PI))


cdef numpy.ndarray[numpy.float64_t,ndim=1] \
	prob_normal_vec_1d(numpy.ndarray[numpy.float64_t,ndim=1] vec, double s):

	cdef numpy.ndarray[numpy.float64_t,ndim=1] prob_vec = \
			numpy.exp(-0.5*(vec/s)**2)

	return prob_vec/(s*sqrt(2.0*M_PI))


cdef double prob_normal_scalar(double val, double s):
	cdef double prob = numpy.exp(-0.5*(val/s)**2)

	return prob/(s*sqrt(2.0*M_PI))



# Using logistic distributions

def redescend_residuals_logistic_2d(
		numpy.ndarray[numpy.float64_t,ndim=2] resids, double cval):
	return get_redescend_weights_logistic_2d(resids, cval)*resids


def get_redescend_weights_logistic_2d(
		numpy.ndarray[numpy.float64_t,ndim=2] resids,double cval):

	cdef double sval = resids.std()*sqrt(3.0)/M_PI

	cdef numpy.ndarray[numpy.float64_t,ndim=2] prob_vec1 = \
			prob_logistic_vec_2d(resids,sval)

	cdef numpy.ndarray[numpy.float64_t,ndim=2] prob_vec2 = \
			prob_logistic_vec_2d(resids,cval*sval)

	cdef numpy.ndarray[numpy.float64_t,ndim=2] likelihood_weights = \
		prob_vec1/(prob_vec1 + prob_vec2)

	cdef double prob1 = prob_logistic_scalar(0.0, sval)
	cdef double prob2 = prob_logistic_scalar(0.0, cval*sval)

	likelihood_weights /= prob1/(prob1 + prob2)

	return likelihood_weights


cdef double prob_logistic_scalar(double val, double s):
	cdef double numerator = exp(-val/s)
	cdef double denominator = 1.0 + numerator
	denominator *= s*denominator

	return numerator/denominator


cdef numpy.ndarray[numpy.float64_t,ndim=1] prob_logistic_vec_1d(
		numpy.ndarray[numpy.float64_t,ndim=1] vec, double s):

	cdef numpy.ndarray[numpy.float64_t,ndim=1] numerator = exp(-vec/s)
	cdef numpy.ndarray[numpy.float64_t,ndim=1] denominator = \
		1.0 + numerator
	denominator *= s*denominator

	return numerator/denominator


cdef numpy.ndarray[numpy.float64_t,ndim=2] \
	prob_logistic_vec_2d(numpy.ndarray[numpy.float64_t,ndim=2] vec, double s):
	cdef numpy.ndarray[numpy.float64_t,ndim=2] numerator = numpy.exp(-vec/s)
	cdef numpy.ndarray[numpy.float64_t,ndim=2] denominator = \
		1.0 + numerator
	denominator *= s*denominator

	return numerator/denominator

