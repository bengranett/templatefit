import os
import logging

import numpy as np
cimport numpy as np

from scipy import interpolate

cimport libc.math as math

from pypelid.utils import consts

import utils

DEF EXP_LIMIT = -10
DEF BIGNEGNUM = -1e10

cdef class TemplateFit:
	
	def __init__(self, wavelength_scale,
					redshift_grid,
					lines=None,
					conv_limit=3,
					res=2,
					template_file=None,
					prior_z=None,
					):
		""" """
		self.logger = logging.getLogger(self.__class__.__name__)

		self.redshift_grid = redshift_grid

		self.res = res
		self.conv_limit = conv_limit

		self.init_wavelength_scale(wavelength_scale)

		if lines is None:
			lines = consts.line_list.keys()
			lines.sort()
		self.lines = lines

		self.init_templates(filename = template_file)

		self.init_prior_z(prior_z)

		self.temp = np.zeros((self.ntemplates, wavelength_scale.shape[0]), dtype=np.float)
		self.amp = np.zeros(self.ntemplates, dtype=np.float)
		self.loglike = np.zeros((redshift_grid.shape[0], self.ntemplates), dtype=np.float)


	def init_wavelength_scale(self, wavelength_scale):
		""" """
		self.lam_step = wavelength_scale[1] - wavelength_scale[0]
		if self.lam_step > 0:
			self.inv_lam_step = 1./self.lam_step
		else:
			self.inv_lam_step = 0
		self.lam_start = wavelength_scale[0]
		self.lam_end = wavelength_scale[wavelength_scale.shape[0]-1]

		try:
			assert self.lam_step > 0
			assert self.inv_lam_step > 0
			assert np.isfinite(self.inv_lam_step)
			assert self.lam_start < self.lam_end
		except AssertionError:
			raise ValueError("Wavelength scale must be increasing (got %s)", str(wavelength_scale[:10]))

	def init_templates(self, templates_data=None, priors=None, filename=None):
		""" """
		if filename is not None:
			templates_data, priors = utils.load_template_file(filename)

		lines = []
		templates = []

		for line in self.lines:
			if not line in templates_data.dtype.names:
				logging.warning("Line missing from template file `%s`: %s", filename, line)
				continue
			lines.append(line)
			templates.append(templates_data[line])

		self.lines = lines

		ntempl = len(priors)
		if ntempl > 1:
			templates = np.transpose(templates)
		else:
			templates = np.array([templates])

		self.logger.debug("Got %s", str(templates))
		self.logger.debug("Loaded %i templates", ntempl)


		self.logger.debug("Loaded spectral templates from file %s - found %i templates with %i lines", filename, templates.shape[0], templates.shape[1])
		self.logger.debug("Line list for z measurement: %s", ", ".join(self.lines))

		self.templates = templates
		self.ntemplates = ntempl
		self.prior_template = priors

		self.rest_wavelengths = self._rest_wavelengths()
		self.nlines = self.rest_wavelengths.shape[0]

		# determine redshift limits of template fit
		self.zmin = self.lam_start / np.max(self.rest_wavelengths) - 1
		self.zmax = self.lam_end / np.min(self.rest_wavelengths) - 1
		if self.zmin < 0:
			self.zmin = 0

		logging.info("template fit start %f end %f",self.lam_start, self.lam_end)
		logging.info("template fit zmin %f zmax %f",self.zmin, self.zmax)


	def init_prior_z(self, prior_z):
		""" """
		cdef int i
		if prior_z is None:
			self.prior_z_grid = np.ones(self.redshift_grid.shape[0])
			return
		z, pz = prior_z
		logging.info("redshift prior z-min %f z-max %f z-peak %f mean z %f", z.min(), z.max(),z[np.argmax(pz)], np.sum(z*pz)/np.sum(pz))
		pz_func = interpolate.interp1d(z, pz, bounds_error=False, fill_value=0)
		self.prior_z_grid = pz_func(self.redshift_grid)

	def _rest_wavelengths(self):
		""" """
		rest_wavelengths = []
		for line in self.lines:
			if not line in consts.line_list:
				raise MeasureError("Line named '%s' is not in consts.line_list!", line)
			rest_wavelengths.append(consts.line_list[line])
			self.logger.debug("Line %s: %f A", line, consts.line_list[line])
		return np.array(rest_wavelengths)

	cdef int _compute_probz(self,
						double [:] precomp_gauss,
						double [:] amp_temp, 
						double [:] flux,
						double [:] invnoisevar,
						double z,
						int width,
						int width_hr
						) nogil:
		""" Compute the redshift probability marginalizing
		over the template set.

		Parameters
		----------
		flux : numpy.ndarray
			Array of spectral flux values
		invnoisevar : numpy.ndarray
			Array of noise values (variance of flux)
		z : double
			Redshift to test
		sigma : double
			Line width
		Returns
		-------
		double
		"""
		cdef int i, pix, tempi, window_low, window_high
		cdef double obs_wave, norm, amp, pix_f
		
		cdef int n = flux.shape[0]
		cdef int got_a_line = 0

		# make local views of the arrays (cython requirement)
		cdef double [:] rest_wavelengths = self.rest_wavelengths
		cdef double [:,:] templates = self.templates
		cdef double [:,:] temp = self.temp


		if self.res <= 0:
			with gil:
				raise ValueError("res=%s, but must be > 0", self.res)

		if width <= 0:
			with gil:
				raise ValueError("width=%s, but must be > 0", width)

		if width_hr <= 0:
			with gil:
				raise ValueError("width_hr=%s, but must be > 0", width_hr)


		window_low = n
		window_high = 0

		for tempi in range(self.ntemplates):
			for i in range(n):
				temp[tempi,i] = 0

		for linei in range(self.nlines):
			obs_wave = rest_wavelengths[linei] * (1 + z)
			if obs_wave < self.lam_start:
				continue
			if obs_wave > self.lam_end:
				continue

			pix_f = (obs_wave - self.lam_start) * self.inv_lam_step
			pix = <int> pix_f

			if pix - width < window_low:
				window_low = pix - width
			if pix + width > window_high:
				window_high = pix + width

			for tempi in range(self.ntemplates):

				for i in range(2*width_hr+1):
					j = <int> ((<double>i - width_hr)/self.res + 0.5 + pix_f)
					if j < 0:
						continue
					if j >= n:
						continue
					temp[tempi, j] = temp[tempi, j] + templates[tempi, linei] * precomp_gauss[i]

			got_a_line = 1

		if not got_a_line:
			return got_a_line

		if window_low < 0:
			window_low = 0

		if window_high > n:
			window_high = n

		for tempi in range(self.ntemplates):
			norm = 0.
			amp = 0.
			for i in range(window_low, window_high):
				if temp[tempi, i] > 0:
					amp += flux[i] * temp[tempi, i] * invnoisevar[i]
					norm += temp[tempi, i] * temp[tempi, i] * invnoisevar[i]

			if norm > 0:
				amp_temp[tempi] = 0.5 * amp * amp / norm
			else:
				amp_temp[tempi] = 0

		return got_a_line

	cdef double [:] _precompute_gaussian(self, double sigma, int *width, int *width_hr):
		""" Precompute the gaussian fuction given sigma.

		Parameters
		----------
		sigma : double
			std deviaton of gaussian in pixel units

		Returns
		-------
		array, sigp
		"""
		cdef int i, j

		cdef double invsigma = 1./(sigma*self.res)
		cdef double invgaussvar = 1./2.*invsigma*invsigma

		width[0] = <int> (sigma * self.conv_limit)
		width_hr[0] = <int> (sigma * self.conv_limit * self.res)

		if width[0] < 1:
			width[0] = 1
			width_hr[0] = self.res

		cdef double[:] x = np.zeros(2*width_hr[0]+1, dtype=np.float)

		for i in range(x.shape[0]):
			j = i - width_hr[0]
			x[i] = math.exp(-j*j*invgaussvar)

		return x

	cpdef double template_fit(self, 
				double [:] flux,
				double [:] invvar,
				double sigma):
		""" Run template fit

		Parameters
		----------
		zgrid : numpy.ndarray
			array of redshift values to try
		flux : numpy.ndarray
			spectral flux (arbitrary units)
		invvar : numpy.ndarray
			inverse of variance on the flux
		sigma: double
			line width in stddev (pixel units)

		Returns
		-------
		numpy.ndarray : p(z) values (un-normalized)
		"""
		cdef int i, j, n, nz, success
		cdef double d, max_amp
		cdef int width=0, width_hr=0
		cdef double[:] precomp_gauss

		n = flux.shape[0]
		nz = self.redshift_grid.shape[0]

		cdef double [:] amp = self.amp
		cdef double [:,:] loglike = self.loglike

		cdef double [:] zgrid = self.redshift_grid

		precomp_gauss = self._precompute_gaussian(sigma, &width, &width_hr)

		self.logger.debug("gauss %s", str(np.array(precomp_gauss)))

		max_amp = 0

		with nogil:
			for i in range(nz):

				success = self._compute_probz(precomp_gauss, amp, flux, invvar, zgrid[i], width, width_hr)

				if not success:
					continue

				for j in range(self.ntemplates):
					if amp[j] == 0:
						continue
					loglike[i, j] = amp[j]

					if amp[j] > max_amp:
						max_amp = amp[j]

		return max_amp

	cpdef double[:] pz(self):
		""" """
		cdef int i, j
		cdef double max_amp

		cdef double[:] pz = np.zeros(self.redshift_grid.shape[0])

		cdef double [:] prior_z_grid = self.prior_z_grid
		cdef double [:] prior_template = self.prior_template

		cdef double [:,:] loglike = self.loglike

		with nogil:
			max_amp = max2(loglike)

			for i in range(loglike.shape[0]):
				if prior_z_grid[i] == 0:
					pz[i] = 0
					continue
				for j in range(loglike.shape[1]):
					pz[i] += math.exp(loglike[i,j] - max_amp) * prior_template[j] * prior_z_grid[i]
		return pz

	cpdef double[:] combine_pz(self, TemplateFit fit2):
		""" """
		cdef int i, j
		cdef double max_amp
		
		cdef double[:] pz = np.zeros(self.redshift_grid.shape[0])


		cdef double [:] prior_z_grid = self.prior_z_grid
		cdef double [:] prior_template = self.prior_template

		cdef double [:,:] loglike = self.loglike
		cdef double [:,:] loglike2 = fit2.loglike
		cdef double [:,:] combine = np.zeros((loglike.shape[0], loglike.shape[1]), dtype=np.float)

		with nogil:
			max_amp = -math.INFINITY
			for i in range(loglike.shape[0]):
				for j in range(loglike.shape[1]):
					combine[i,j] = loglike[i,j] + loglike2[i,j]
					if combine[i,j] > max_amp:
						max_amp = combine[i,j]


			for i in range(loglike.shape[0]):
				if prior_z_grid[i] == 0:
					pz[i] = 0
					continue
				for j in range(loglike.shape[1]):
					pz[i] += math.exp(combine[i,j] - max_amp) * prior_template[j] * prior_z_grid[i]
		return pz



class MeasureError(Exception):
	pass
