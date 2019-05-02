
cdef class TemplateFit:

	cdef object logger
	cdef object config
	cdef object lines

	cdef public double zmin
	cdef public double zmax

	cdef double lam_start, lam_end, lam_step, inv_lam_step, conv_limit
	cdef int res, nlines, ntemplates

	cdef double [:] rest_wavelengths
	cdef double [:,:] templates
	cdef double [:] priors

	cdef int _compute_probz(self,
						double [:] precomp_gauss,
						double [:,:] temp,
						double [:] amp_temp, 
						double [:] flux,
						double [:] invnoisevar,
						double z,
						int width,
						int width_hr
						) nogil

	cdef double [:] _precompute_gaussian(self, double sigma, int *width, int *width_hr)
	
	cpdef double [:,:] template_fit(self, 
				double [:] z,
				double [:] flux,
				double [:] invnoisevar,
				double radius,
				double [:] pz)
