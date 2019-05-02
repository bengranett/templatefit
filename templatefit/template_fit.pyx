#cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True
import os
import logging

import numpy as np
cimport numpy as np

cimport libc.math as math
# from cython.parallel import prange

from pypelid.utils import filetools
from pypelid.utils import consts

DEF EXP_LIMIT = -10
DEF BIGNEGNUM = -1e10

cdef class TemplateFit:
	
	def __init__(self,
					optics,
					config,
					lines=None,
					conv_limit=3,
					res=2,
					**kwargs
					):
		""" """
		self.logger = logging.getLogger(self.__class__.__name__)
		self.config = config

		if lines is None:
			lines = consts.line_list.keys()
			lines.sort()
		self.lines = lines

		self.res = res
##

#		print len(optics)
		for i in range(len(optics)):
			self.lam_start, self.lam_end = optics[0].config['lambda_range']
			if optics[i].config['lambda_range'][0] < self.lam_start:
				self.lam_start = optics[i].config['lambda_range'][0]
			if optics[i].config['lambda_range'][1] > self.lam_end:
				self.lam_end = optics[i].config['lambda_range'][1]

##
#		self.lam_start, self.lam_end = optics[0].config['lambda_range']
#		self.lam_start, self.lam_end = optics['lambda_range']
		self.lam_step = optics[0].config['pix_disp'][1]
#		self.lam_step = optics['pix_disp'][1]

		self._load_template_file()

		self.rest_wavelengths = self._rest_wavelengths()

		self.inv_lam_step = 1./self.lam_step
		assert np.isfinite(self.inv_lam_step)
		self.conv_limit = conv_limit
		self.nlines = self.rest_wavelengths.shape[0]
		self.ntemplates = self.templates.shape[0]

		# determine redshift limits of template fit
		self.zmin = self.lam_start / np.max(self.rest_wavelengths) - 1
		self.zmax = self.lam_end / np.min(self.rest_wavelengths) - 1
		if self.zmin < 0:
			self.zmin = 0

		logging.info("template fit start %f end %f",self.lam_start, self.lam_end)
		logging.info("template fit zmin %f zmax %f",self.zmin, self.zmax)

	cpdef _load_template_file(self):
		""" """
		filename = filetools.get_path(self.config['in_dir'], self.config['zmeas_template_file'])
		logging.debug("filename %s",filename)
		if not os.path.exists(filename):
			raise IOError("File does not exist: %s"%filename)

		templates_data = np.genfromtxt(filename, delimiter=",", names=True)

		priors = templates_data['prior']
		try:
			ntempl = len(priors)
		except TypeError:
			ntempl = 1
			priors = np.array([priors])

		templates = []
		lines = []

		for line in self.lines:
			if not line in templates_data.dtype.names:
				logging.warning("Line missing from template file `%s`: %s", filename, line)
				continue
			lines.append(line)
			templates.append(templates_data[line])

		self.logger.debug("got %s", templates)

		self.lines = lines
		if ntempl > 1:
			templates = np.transpose(templates)
		else:
			templates = np.array([templates])

		self.logger.debug("Loaded spectral templates from file %s - found %i templates with %i lines", filename, templates.shape[0], templates.shape[1])
		self.logger.debug("Line list for z measurement: %s", ", ".join(self.lines))

		self.templates = templates
		self.priors = priors

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
						double [:,:] temp,
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

				for i in range(2*width_hr):
					j = <int> ((<double>i - width_hr)/self.res + pix_f)
					# with gil:
						# logging.debug("i j %f %f %f %f %f",i,j, width_hr, self.res, pix_f)
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

		# with gil:
			# logging.debug("template %f",np.mean(temp))

		for tempi in range(self.ntemplates):
			norm = 0.
			amp = 0.
			for i in range(window_low, window_high):
				if temp[tempi, i] > 0:
					amp += flux[i] * temp[tempi, i] * invnoisevar[i]
					norm += temp[tempi, i] * temp[tempi, i] * invnoisevar[i]

			if norm > 0:
				amp_temp[tempi] = amp * amp / norm
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

		cdef double[:] x = np.zeros(2*width_hr[0], dtype=np.float)

		for i in range(2*width_hr[0]):
			j = i - width_hr[0]
			x[i] = math.exp(-j*j*invgaussvar)

		return x

	cpdef double template_fit(self, 
				double [:] z,
				double [:] flux,
				double [:] invnoisevar,
				double radius,
				double [:] pz):
		""" Run template fit

		Parameters
		----------
		z : numpy.ndarray
			array of redshift values to try
		flux : numpy.ndarray
			spectral flux
		invnoisevar : numpy.ndarray
			spectral noise (variance of the flux)
		sigma: double
			line width

		Returns
		-------
		numpy.ndarray : p(z) values (unormalized)

		"""
		cdef int i, j, n, nz, success
		cdef double d, sigma, max_amp
		cdef int width=0, width_hr=0
		cdef double[:] precomp_gauss

		cdef double [:] priors = self.priors

		n = flux.shape[0]
		nz = z.shape[0]
		cdef double [:,:] temp = np.zeros((self.ntemplates, n), dtype=np.float)
		cdef double [:] amp = np.zeros(self.ntemplates, dtype=np.float)
		cdef double [:,:] amp_all = np.zeros((nz, self.ntemplates), dtype=np.float)

		sigma = radius * consts.RADIUS_TO_SIGMA

		precomp_gauss = self._precompute_gaussian(sigma, &width, &width_hr)

		# logging.debug("widths %f %f", width, width_hr)
		# logging.debug("gauss %s",str(np.array(precomp_gauss)))

		max_amp = BIGNEGNUM

		for i in range(nz):
			pz[i] = 0
			success = self._compute_probz(precomp_gauss, temp, amp, flux, invnoisevar, z[i], width, width_hr)

			if not success:
				continue

			# logging.debug("z %f success %i flux %s noise %s amp %f", z[i], success, np.mean(flux), np.mean(invnoisevar), amp[0])

			for j in range(self.ntemplates):
				if amp[j] == 0:
					continue

				amp_all[i, j] = amp[j]
				if amp[j] > max_amp:
					max_amp = amp[j]

		logging.info("max_amp %f", max_amp)

		if nz > 1:
			for i in range(nz):
				for j in range(self.ntemplates):
					d = 0.5 * (amp_all[i, j] - max_amp)
					assert d <= 0
					if d < EXP_LIMIT:
						continue
					pz[i] += math.exp(d) * priors[j]

		return max_amp

class MeasureError(Exception):
	pass
