
cdef inline double max2(double [:,:] x) nogil:
	""" Return the maximum element in a 2D array. """
	cdef int i, j;
	cdef double m;
	m = x[0,0]
	for i in range(x.shape[0]):
		for j in range(x.shape[1]):
			if x[i,j] > m:
				m = x[i,j]
	return m


cdef class TemplateFit:

	cdef object logger
	cdef object config
	cdef object line_list
	cdef object lines

	cdef public double zmin
	cdef public double zmax

	cdef double lam_start, lam_end, lam_step, inv_lam_step, conv_limit
	cdef int res, nlines, ntemplates

	cdef public double [:] redshift_grid
	cdef double [:] _pz
	cdef public double [:,:] loglike

	cdef double [:] rest_wavelengths
	cdef double [:,:] templates
	cdef double [:] prior_template
	cdef double [:] prior_z_grid
	cdef double [:,:] temp
	cdef double [:] amp


	cdef int _compute_probz(self,
						double [:] precomp_gauss,
						double [:] amp_temp,
						double [:] flux,
						double [:] invnoisevar,
						double z,
						int width,
						int width_hr
						) nogil

	cdef double [:] _precompute_gaussian(self, double sigma, int *width, int *width_hr)

	cpdef double template_fit(self,
							double [:] flux,
							double [:] invnoisevar,
							double sigma)

	cpdef double template_fit_at_z(self,
							double z,
							double [:] flux,
							double [:] invvar,
							double sigma)

	cpdef double[:] pz(self)
	cpdef double[:] combine_pz(self, TemplateFit fit2)
