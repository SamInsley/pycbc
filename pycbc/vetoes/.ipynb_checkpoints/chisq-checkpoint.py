# Copyright (C) 2012  Alex Nitz
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.


#
# =============================================================================
#
#                                   Preamble
#
# =============================================================================
#
import numpy, logging, math, pycbc.fft

from pycbc.types import zeros, real_same_precision_as, TimeSeries, FrequencySeries, complex_same_precision_as
from pycbc.filter import sigmasq_series, make_frequency_series, matched_filter_core, get_cutoff_indices
from pycbc.scheme import schemed
import pycbc.pnutils

BACKEND_PREFIX="pycbc.vetoes.chisq_"

def power_chisq_bins_from_sigmasq_series(sigmasq_series, num_bins, kmin, kmax):
    """Returns bins of equal power for use with the chisq functions

    Parameters
    ----------

    sigmasq_series: FrequencySeries
        A frequency series containing the cumulative power of a filter template
        preweighted by a psd.
    num_bins: int
        The number of chisq bins to calculate.
    kmin: int
        DOCUMENTME
    kmax: int
        DOCUMENTME

    Returns
    -------

    bins: List of ints
        A list of the edges of the chisq bins is returned.
    """
    sigmasq = sigmasq_series[kmax - 1]
    edge_vec = numpy.arange(0, num_bins) * sigmasq / num_bins
    bins = numpy.searchsorted(sigmasq_series[kmin:kmax], edge_vec, side='right')
    bins += kmin
    return numpy.append(bins, kmax)


def power_chisq_bins(htilde, num_bins, psd, low_frequency_cutoff=None,
                     high_frequency_cutoff=None):
    """Returns bins of equal power for use with the chisq functions

    Parameters
    ----------

    htilde: FrequencySeries
        A frequency series containing the template waveform
    num_bins: int
        The number of chisq bins to calculate.
    psd: FrequencySeries
        A frequency series containing the psd. Its length must be commensurate
        with the template waveform.
    low_frequency_cutoff: {None, float}, optional
        The low frequency cutoff to apply
    high_frequency_cutoff: {None, float}, optional
        The high frequency cutoff to apply

    Returns
    -------

    bins: List of ints
        A list of the edges of the chisq bins is returned.
    """
    sigma_vec = sigmasq_series(htilde, psd, low_frequency_cutoff,
                               high_frequency_cutoff).numpy()
    kmin, kmax = get_cutoff_indices(low_frequency_cutoff,
                                    high_frequency_cutoff,
                                    htilde.delta_f,
                                    (len(htilde)-1)*2)
    return power_chisq_bins_from_sigmasq_series(sigma_vec, num_bins, kmin, kmax)


@schemed(BACKEND_PREFIX)
def chisq_accum_bin(chisq, q):
    err_msg = "This function is a stub that should be overridden using the "
    err_msg += "scheme. You shouldn't be seeing this error!"
    raise ValueError(err_msg)

@schemed(BACKEND_PREFIX)
def shift_sum(v1, shifts, bins):
    """ Calculate the time shifted sum of the FrequencySeries
    """
    err_msg = "This function is a stub that should be overridden using the "
    err_msg += "scheme. You shouldn't be seeing this error!"
    raise ValueError(err_msg)

@schemed(BACKEND_PREFIX)
def shift_multibin(v1, shifts, scales, bins):
    """ Calculate the time shifted sum of the FrequencySeries
    """
    err_msg = "This function is a stub that should be overridden using the "
    err_msg += "scheme. You shouldn't be seeing this error!"
    raise ValueError(err_msg)

def power_chisq_at_points_from_precomputed(corr, snr, snr_norm, bins, indices):
    """Calculate the chisq timeseries from precomputed values for only select points.

    This function calculates the chisq at each point by explicitly time shifting
    and summing each bin. No FFT is involved.

    Parameters
    ----------
    corr: FrequencySeries
        The product of the template and data in the frequency domain.
    snr: numpy.ndarray
        The unnormalized array of snr values at only the selected points in `indices`.
    snr_norm: float
        The normalization of the snr (EXPLAINME : refer to Findchirp paper?)
    bins: List of integers
        The edges of the equal power bins
    indices: Array
        The indices where we will calculate the chisq. These must be relative
        to the given `corr` series.

    Returns
    -------
    chisq: Array
        An array containing only the chisq at the selected points.
    """
    num_bins = len(bins) - 1
    chisq = shift_sum(corr, indices, bins) # pylint:disable=assignment-from-no-return
    return (chisq * num_bins - (snr.conj() * snr).real) * (snr_norm ** 2.0)

_q_l = None
_qtilde_l = None
_chisq_l = None
def power_chisq_from_precomputed(corr, snr, snr_norm, bins, indices=None, return_bins=False):
    """Calculate the chisq timeseries from precomputed values.

    This function calculates the chisq at all times by performing an
    inverse FFT of each bin.

    Parameters
    ----------

    corr: FrequencySeries
        The produce of the template and data in the frequency domain.
    snr: TimeSeries
        The unnormalized snr time series.
    snr_norm:
        The snr normalization factor (true snr = snr * snr_norm) EXPLAINME - define 'true snr'?
    bins: List of integers
        The edges of the chisq bins.
    indices: {Array, None}, optional
        Index values into snr that indicate where to calculate
        chisq values. If none, calculate chisq for all possible indices.
    return_bins: {boolean, False}, optional
        Return a list of the SNRs for each chisq bin.

    Returns
    -------
    chisq: TimeSeries
    """
    # Get workspace memory
    global _q_l, _qtilde_l, _chisq_l

    bin_snrs = []

    if _q_l is None or len(_q_l) != len(snr):
        q = zeros(len(snr), dtype=complex_same_precision_as(snr))
        qtilde = zeros(len(snr), dtype=complex_same_precision_as(snr))
        _q_l = q
        _qtilde_l = qtilde
    else:
        q = _q_l
        qtilde = _qtilde_l

    if indices is not None:
        snr = snr.take(indices)

    if _chisq_l is None or len(_chisq_l) < len(snr):
        chisq = zeros(len(snr), dtype=real_same_precision_as(snr))
        _chisq_l = chisq
    else:
        chisq = _chisq_l[0:len(snr)]
        chisq.clear()

    num_bins = len(bins) - 1

    for j in range(num_bins):
        k_min = int(bins[j])
        k_max = int(bins[j+1])

        qtilde[k_min:k_max] = corr[k_min:k_max]
        pycbc.fft.ifft(qtilde, q)
        qtilde[k_min:k_max].clear()

        if return_bins:
            bin_snrs.append(TimeSeries(q  * snr_norm *  num_bins ** 0.5,
                                      delta_t=snr.delta_t,
                                      epoch=snr.start_time))

        if indices is not None:
            chisq_accum_bin(chisq, q.take(indices))
        else:
            chisq_accum_bin(chisq, q)

    chisq = (chisq * num_bins - snr.squared_norm()) * (snr_norm ** 2.0)

    if indices is None:
        chisq = TimeSeries(chisq, delta_t=snr.delta_t, epoch=snr.start_time, copy=False)

    if return_bins:
        return chisq, bin_snrs
    else:
        return chisq


def fastest_power_chisq_at_points(corr, snr, snrv, snr_norm, bins, indices):
    """Calculate the chisq values for only selected points.

    This function looks at the number of points to be evaluated and selects
    the fastest method (FFT, or direct time shift and sum). In either case,
    only the selected points are returned.

    Parameters
    ----------
    corr: FrequencySeries
        The product of the template and data in the frequency domain.
    snr: Array
        The unnormalized snr
    snr_norm: float
        The snr normalization factor  --- EXPLAINME
    bins: List of integers
        The edges of the equal power bins
    indices: Array
        The indices where we will calculate the chisq. These must be relative
        to the given `snr` series.

    Returns
    -------
    chisq: Array
        An array containing only the chisq at the selected points.
    """
    import pycbc.scheme
    if isinstance(pycbc.scheme.mgr.state, pycbc.scheme.CPUScheme):
        # We don't have that many points so do the direct time shift.
        return power_chisq_at_points_from_precomputed(corr, snrv,
                                                      snr_norm, bins, indices)
    else:
        # We have a lot of points so it is faster to use the fourier transform
        return power_chisq_from_precomputed(corr, snr, snr_norm, bins,
                                            indices=indices)


def power_chisq(template, data, num_bins, psd,
                low_frequency_cutoff=None,
                high_frequency_cutoff=None,
                return_bins=False):
    """Calculate the chisq timeseries

    Parameters
    ----------
    template: FrequencySeries or TimeSeries
        A time or frequency series that contains the filter template.
    data: FrequencySeries or TimeSeries
        A time or frequency series that contains the data to filter. The length
        must be commensurate with the template.
        (EXPLAINME - does this mean 'the same as' or something else?)
    num_bins: int
        The number of frequency bins used for chisq. The number of statistical
        degrees of freedom ('dof') is 2*num_bins-2.
    psd: FrequencySeries
        The psd of the data.
    low_frequency_cutoff: {None, float}, optional
        The low frequency cutoff for the filter
    high_frequency_cutoff: {None, float}, optional
        The high frequency cutoff for the filter
    return_bins: {boolean, False}, optional
        Return a list of the individual chisq bins

    Returns
    -------
    chisq: TimeSeries
        TimeSeries containing the chisq values for all times.
    """
    htilde = make_frequency_series(template)
    stilde = make_frequency_series(data)

    bins = power_chisq_bins(htilde, num_bins, psd, low_frequency_cutoff,
                            high_frequency_cutoff)
    corra = zeros((len(htilde) - 1) * 2, dtype=htilde.dtype)
    total_snr, corr, tnorm = matched_filter_core(htilde, stilde, psd,
                           low_frequency_cutoff, high_frequency_cutoff,
                           corr_out=corra)

    return power_chisq_from_precomputed(corr, total_snr, tnorm, bins, return_bins=return_bins)


class SingleDetPowerChisq(object):
    """Class that handles precomputation and memory management for efficiently
    running the power chisq in a single detector inspiral analysis.
    """
    def __init__(self, num_bins=0, snr_threshold=None):
        if not (num_bins == "0" or num_bins == 0):
            self.do = True
            self.column_name = "chisq"
            self.table_dof_name = "chisq_dof"
            self.num_bins = num_bins
        else:
            self.do = False
        self.snr_threshold = snr_threshold

    @staticmethod
    def parse_option(row, arg):
        safe_dict = {'max': max, 'min': min}
        safe_dict.update(row.__dict__)
        safe_dict.update(math.__dict__)
        safe_dict.update(pycbc.pnutils.__dict__)
        return eval(arg, {"__builtins__":None}, safe_dict)

    def cached_chisq_bins(self, template, psd):
        from pycbc.opt import LimitedSizeDict

        key = id(psd)
        if not hasattr(psd, '_chisq_cached_key'):
            psd._chisq_cached_key = {}

        if not hasattr(template, '_bin_cache'):
            template._bin_cache = LimitedSizeDict(size_limit=2**2)

        if key not in template._bin_cache or id(template.params) not in psd._chisq_cached_key:
            psd._chisq_cached_key[id(template.params)] = True
            num_bins = int(self.parse_option(template, self.num_bins))

            if hasattr(psd, 'sigmasq_vec') and \
                    template.approximant in psd.sigmasq_vec:
                kmin = int(template.f_lower / psd.delta_f)
                kmax = template.end_idx
                bins = power_chisq_bins_from_sigmasq_series(
                    psd.sigmasq_vec[template.approximant],
                    num_bins,
                    kmin,
                    kmax
                )
            else:
                bins = power_chisq_bins(template, num_bins, psd, template.f_lower)
            template._bin_cache[key] = bins

        return template._bin_cache[key]

    def values(self, corr, snrv, snr_norm, psd, indices, template):
        """ Calculate the chisq at points given by indices.

        Returns
        -------
        chisq: Array
            Chisq values, one for each sample index, or zero for points below
            the specified SNR threshold

        chisq_dof: Array
            Number of statistical degrees of freedom for the chisq test
            in the given template, equal to 2 * num_bins - 2
        """
        if self.do:
            num_above = len(indices)
            dof = -100
            if self.snr_threshold:
                above = abs(snrv * snr_norm) > self.snr_threshold
                num_above = above.sum()
                logging.info('%s above chisq activation threshold' % num_above)
                above_indices = indices[above]
                above_snrv = snrv[above]
                chisq_out = numpy.zeros(len(indices), dtype=numpy.float32)
            else:
                above_indices = indices
                above_snrv = snrv

            if num_above > 0:
                bins = self.cached_chisq_bins(template, psd)
                # len(bins) is number of bin edges, num_bins = len(bins) - 1
                dof = (len(bins) - 1) * 2 - 2
                _chisq = power_chisq_at_points_from_precomputed(corr,
                                     above_snrv, snr_norm, bins, above_indices)

            if self.snr_threshold:
                if num_above > 0:
                    chisq_out[above] = _chisq
            else:
                if num_above == 0:
                    chisq_out = numpy.zeros(0, dtype=numpy.float32)
                else:
                    chisq_out = _chisq

            return chisq_out, numpy.repeat(dof, len(indices))# dof * numpy.ones_like(indices)
        else:
            return None, None


class SingleDetSkyMaxPowerChisq(SingleDetPowerChisq):
    """Class that handles precomputation and memory management for efficiently
    running the power chisq in a single detector inspiral analysis when
    maximizing analytically over sky location.
    """
    def __init__(self, **kwds):
        super(SingleDetSkyMaxPowerChisq, self).__init__(**kwds)
        self.template_mem = None
        self.corr_mem = None

    def calculate_chisq_bins(self, template, psd):
        """ Obtain the chisq bins for this template and PSD.
        """
        num_bins = int(self.parse_option(template, self.num_bins))
        if hasattr(psd, 'sigmasq_vec') and \
                                       template.approximant in psd.sigmasq_vec:
            kmin = int(template.f_lower / psd.delta_f)
            kmax = template.end_idx
            bins = power_chisq_bins_from_sigmasq_series(
                   psd.sigmasq_vec[template.approximant], num_bins, kmin, kmax)
        else:
            bins = power_chisq_bins(template, num_bins, psd, template.f_lower)
        return bins

    def values(self, corr_plus, corr_cross, snrv, psd,
               indices, template_plus, template_cross, u_vals,
               hplus_cross_corr, hpnorm, hcnorm):
        """ Calculate the chisq at points given by indices.

        Returns
        -------
        chisq: Array
            Chisq values, one for each sample index

        chisq_dof: Array
            Number of statistical degrees of freedom for the chisq test
            in the given template
        """
        if self.do:
            num_above = len(indices)
            if self.snr_threshold:
                above = abs(snrv) > self.snr_threshold
                num_above = above.sum()
                logging.info('%s above chisq activation threshold' % num_above)
                above_indices = indices[above]
                above_snrv = snrv[above]
                u_vals = u_vals[above]
                rchisq = numpy.zeros(len(indices), dtype=numpy.float32)
                dof = -100
            else:
                above_indices = indices
                above_snrv = snrv

            if num_above > 0:
                chisq = []
                curr_tmplt_mult_fac = 0.
                curr_corr_mult_fac = 0.
                if self.template_mem is None or \
                        (not len(self.template_mem) == len(template_plus)):
                    self.template_mem = zeros(len(template_plus),
                                dtype=complex_same_precision_as(corr_plus))
                if self.corr_mem is None or \
                                (not len(self.corr_mem) == len(corr_plus)):
                    self.corr_mem = zeros(len(corr_plus),
                                dtype=complex_same_precision_as(corr_plus))

                tmplt_data = template_cross.data
                corr_data = corr_cross.data
                numpy.copyto(self.template_mem.data, template_cross.data)
                numpy.copyto(self.corr_mem.data, corr_cross.data)
                template_cross._data = self.template_mem.data
                corr_cross._data = self.corr_mem.data

                for lidx, index in enumerate(above_indices):
                    above_local_indices = numpy.array([index])
                    above_local_snr = numpy.array([above_snrv[lidx]])
                    local_u_val = u_vals[lidx]
                    # Construct template from _plus and _cross
                    # Note that this modifies in place, so we store that and
                    # revert on the next pass.
                    template = template_cross.multiply_and_add(template_plus,
                                               local_u_val-curr_tmplt_mult_fac)
                    curr_tmplt_mult_fac = local_u_val

                    template.f_lower = template_plus.f_lower
                    template.params = template_plus.params
                    # Construct the corr vector
                    norm_fac = local_u_val*local_u_val + 1
                    norm_fac += 2 * local_u_val * hplus_cross_corr
                    norm_fac = hcnorm / (norm_fac**0.5)
                    hp_fac = local_u_val * hpnorm / hcnorm
                    corr = corr_cross.multiply_and_add(corr_plus,
                                                   hp_fac - curr_corr_mult_fac)
                    curr_corr_mult_fac = hp_fac

                    bins = self.calculate_chisq_bins(template, psd)
                    dof = (len(bins) - 1) * 2 - 2
                    curr_chisq = power_chisq_at_points_from_precomputed(corr,
                                          above_local_snr/ norm_fac, norm_fac,
                                          bins, above_local_indices)
                    chisq.append(curr_chisq[0])
                chisq = numpy.array(chisq)
                # Must reset corr and template to original values!
                template_cross._data = tmplt_data
                corr_cross._data = corr_data

            if self.snr_threshold:
                if num_above > 0:
                    rchisq[above] = chisq
            else:
                rchisq = chisq

            return rchisq, numpy.repeat(dof, len(indices))# dof * numpy.ones_like(indices)
        else:
            return None, None


class SingleDetTHAPowerChisq(SingleDetPowerChisq):
    """Class that handles precomputation and memory management for efficiently
    running the power chisq in a single detector inspiral analysis when
    maximizing analytically over sky location.

    The comparison filter is reconstructed at each trigger's best-fit
    (theta_jn, alpha0, psi) marginalization grid point (see
    pycbc.filter.tha_marginalize.marginalize_for_psd's return_bestfit),
    using that grid point's own analytic/geometric harmonic-combination
    coefficients (see _bestfit_M_row) rather than the SNR-weighted
    harmonic-sum reconstruction described in arXiv:2305.19270 eq. 40
    (h = sum_k <h_k^perp|s> h_k^perp). By linearity of correlation, this
    only ever needs the already-computed per-harmonic correlations
    `corrs` -- no fresh matched filter and no lalsimulation call. The
    SNR-weighted-sum method is kept as `reconstruct_template` and used as
    a fallback for triggers whose template has no marginalization grid
    coverage (theta comes through as NaN, ta_idx as -1, in that case --
    see pycbc_inspiral_tha).
    """
    def __init__(self, **kwds):
        super(SingleDetTHAPowerChisq, self).__init__(**kwds)
        self.template_mem = None
        self.corr_mem = None

    def calculate_chisq_bins(self, template):
        """ Obtain the chisq bins for this template and PSD.
        """
        num_bins = int(self.parse_option(template, self.num_bins))
        # Here template must be whitened, so psd=None implies a flat PSD.
        bins = power_chisq_bins(template, num_bins, psd=None,
                                low_frequency_cutoff=template.f_lower,
                                high_frequency_cutoff=None)
        return bins

    def reconstruct_template(self, templates, weights, num_comps):
        """Weighted-sum reconstruction: h = sum_k w_k h_k^perp, i.e. each
        orthonormal harmonic scaled by a (here, internally re-normalized)
        complex weight and summed.

        `weights` used to always be the trigger's own measured
        per-harmonic SNRs (arXiv:2305.19270 eq. 40: w_k = <h_k^perp|s>,
        the noise/data-dependent maximum-likelihood combination). It is
        now instead the best-fit marginalization grid point's analytic
        combination coefficients (see _bestfit_M_row) -- the physical
        harmonic content of the true best-fit-orientation waveform,
        which depends only on the PSD, not on this segment's noise
        realization. The function itself doesn't care which -- it just
        normalizes and sums whatever 5-vector it's given.
        """
        scales = weights / numpy.sum(numpy.abs(weights) ** 2.) ** 0.5
        template = templates[0] * scales[0]
        for i in range(1, num_comps):
            template += templates[i] * scales[i]

        template.f_lower = templates[0].f_lower
        template.params = templates[0].params
        return template, scales

    def _compute_ep_ec(self, grid, raws, segment_psd):
        """Compute Ep/Ec for the WHOLE marginalization grid at once (one
        row per (theta_jn, alpha0) grid point), for this template's raw
        harmonics and this segment's PSD.

        This is the expensive part of what used to be `_bestfit_M_row`
        (a 5x5 Gram-matrix build over every frequency bin, plus a
        (n_grid, 5) x (5, 5) matrix product against the WHOLE grid's
        A/B) -- but it depends only on `raws`/`grid`/`segment_psd`, i.e.
        only on the template and the segment, NOT on any individual
        trigger's winning grid point. So it must be called exactly ONCE
        per values() call (one template, one segment), not once per
        trigger -- see _bestfit_row_from_ep_ec for the actual per-trigger
        work (a cheap single-row lookup + rotation).

        Parameters
        ----------
        grid : dict
            One entry of tha_marginalize.load_raw_grid_file's return
            (the SAME `grid` used for this template's marg_lnl
            computation in pycbc_inspiral_tha) -- needs 'A', 'B', 'df',
            'kmin', 'kmax'.
        raws : list of 5 (n_freq,) complex arrays
            The raw (unwhitened) harmonics for this template -- the SAME
            `raws` passed to tha_marginalize.marginalize_for_psd for this
            segment (bank.curr_tmp.h1..h5, zero-padded to segment_psd's
            length).
        segment_psd : (n_freq,) float array
            The SAME un-scaled PSD passed to marginalize_for_psd for this
            segment (i.e. stilde.psd / DYN_RANGE_FAC**2, NOT the
            DYN_RANGE_FAC-scaled stilde.psd used for ordinary whitening)
            -- using this exact PSD is what makes Ep/Ec reproduce the
            SAME values that picked out each trigger's grid point as the
            best fit in the first place.

        Returns
        -------
        Ep, Ec : (n_grid, 5) complex arrays
            Whole-grid Ep/Ec, as returned by
            tha_marginalize.derive_ep_ec_hh (hh_pp/hh_cc/hh_pc dropped --
            not needed here).
        """
        from pycbc.filter.tha_marginalize import derive_ep_ec_hh

        Ep, Ec, _hh_pp, _hh_cc, _hh_pc = derive_ep_ec_hh(
            raws, grid['A'], grid['B'], segment_psd,
            grid['df'], grid['kmin'], grid['kmax'])
        return Ep, Ec

    def _bestfit_row_from_ep_ec(self, grid, Ep, Ec, ta_idx, psi):
        """The (5,) complex row of the marginalization grid's M matrix
        for one specific (theta_jn, alpha0) grid point (`ta_idx`) and an
        arbitrary `psi`, i.e. the coefficients c_k such that the
        whitened/normalized best-fit-orientation waveform equals
        Sum_k c_k * e_k, where e_k are the SAME 5 orthonormal harmonic
        templates (`templates`) used for ordinary matched filtering --
        this is what values() below feeds into reconstruct_template as
        `weights`.

        Cheap: just a single-row lookup into the already-computed
        whole-grid Ep/Ec (see _compute_ep_ec, called once per template
        per segment, not per trigger) plus a real rotation by psi -- the
        only things here that vary per trigger.

        By linearity of correlation, the SAME coefficients (once
        renormalized by reconstruct_template) combine the
        already-computed per-harmonic correlations `corrs` into the
        correlation of the best-fit-orientation waveform against the
        data (corr(Sum_k c_k*h_k, data) == Sum_k c_k*corr(h_k, data)) --
        see values() below -- without needing a fresh matched filter or
        any lalsimulation call.

        Parameters
        ----------
        grid : dict
            Needs 'dpsi' (per-grid-point psi offset).
        Ep, Ec : (n_grid, 5) complex arrays
            This template/segment's whole-grid Ep/Ec, from
            _compute_ep_ec.
        ta_idx : int
            Index into grid['dpsi']/Ep/Ec for the winning (theta_jn,
            alpha0) grid point (out_vals['marg_ta_idx']). Must not be -1
            (the "no grid coverage" sentinel) -- callers check that
            themselves (see values()).
        psi : float
            The best-fit psi for this trigger (out_vals['marg_psi']) --
            entering only through a smooth cos(2*(psi-dpsi))/
            sin(2*(psi-dpsi)) rotation, so a float32-rounded value is
            fine here (unlike ta_idx, which must be exact).

        Returns
        -------
        M_row : (5,) complex array
            c_k coefficients, ordered to match `templates`/`corrs`.
        """
        dpsi_i = grid['dpsi'][ta_idx]
        delta = psi - dpsi_i
        c2, s2 = numpy.cos(2. * delta), numpy.sin(2. * delta)
        return c2 * Ep[ta_idx].conj() - s2 * Ec[ta_idx].conj()

    def values(self, corrs, snrv, snr_norm, indices, templates,
               grid, raws, segment_psd, ta_idx, psi):
        """ Calculate the chisq at points given by indices.

        Same overall structure as the original implementation (build a
        comparison template + chisq bins per trigger, then combine the
        already-computed per-harmonic correlations `corrs` with the same
        weights used to build that template): the only thing that has
        changed is where the per-harmonic weights fed into
        reconstruct_template come from -- the best-fit marginalization
        grid point's analytic coefficients (_bestfit_row_from_ep_ec),
        instead of the trigger's own measured per-harmonic SNRs.

        Parameters
        ----------
        corrs : list of (FrequencySeries or None)
            The 5 per-harmonic correlation FrequencySeries (frequency
            domain, conj(harmonic)*data) -- the SAME `corrs` returned by
            matched_filter_and_cluster and already used for ordinary
            matched filtering. Combined directly with the grid point's
            weights -- no fresh correlation is ever performed here.
        snrv : Array
            SNR values at the trigger locations (only used for
            self.snr_threshold gating -- no longer used inside the chisq
            formula itself, see below).
        snr_norm : float
            SNR normalization (matched_filter_and_cluster's `norm`).
        indices : Array
            Indices (relative to the correlation vectors) to compute
            chisq at.
        templates : list of (FrequencySeries or None)
            The 5 whitened/orthogonalized harmonic templates, combined
            with the grid point's weights (via reconstruct_template) to
            get the best-fit-orientation waveform's frequency content
            for calculate_chisq_bins.
        grid : dict
            This template's PSD-independent marginalization grid data
            (the SAME `grid` used for marg_lnl in pycbc_inspiral_tha).
        raws : list of 5 (n_freq,) complex arrays
            This template's live raw harmonics (see _compute_ep_ec).
        segment_psd : (n_freq,) float array
            This segment's un-scaled PSD (see _compute_ep_ec).
        ta_idx, psi : Array
            Per-trigger best-fit grid-point values
            (out_vals['marg_ta_idx']/['marg_psi'] from
            pycbc_inspiral_tha), aligned with snrv/indices. ta_idx is -1
            for triggers with no marginalization grid coverage -- those
            triggers get chisq = NaN (there is no other reconstruction
            to fall back to any more).

        Returns
        -------
        chisq: Array
            Chisq values, one for each sample index

        chisq_dof: Array
            Number of statistical degrees of freedom for the chisq test
            in the given template
        """
        if self.do:
            num_above = len(indices)
            if self.snr_threshold:
                above = abs(snrv) > self.snr_threshold
                num_above = above.sum()
                logging.info('%s above chisq activation threshold' % num_above)
                above_indices = indices[above]
                above_ta_idx = ta_idx[above]
                above_psi = psi[above]
                rchisq = numpy.zeros(len(indices), dtype=numpy.float32)
                dof = -100
            else:
                above_indices = indices
                above_ta_idx = ta_idx
                above_psi = psi

            if num_above > 0:
                bins = []
                scales = []
                covered = []
                num_comps = len([t for t in templates if t is not None])

                # Whole-grid Ep/Ec depends only on raws/grid/segment_psd
                # -- the same for every trigger of this template in this
                # segment -- so it's computed exactly ONCE here, outside
                # the per-trigger loop below, instead of being
                # recalculated (Gram matrix + full (n_grid, 5) matrix
                # products) once per trigger. Only computed at all if
                # there's a grid for this template; if grid is None,
                # every ta_idx below will be -1 (see pycbc_inspiral_tha)
                # and Ep/Ec are never read.
                if grid is not None:
                    Ep, Ec = self._compute_ep_ec(grid, raws, segment_psd)
                else:
                    Ep = Ec = None

                for lidx, index in enumerate(above_indices):
                    ta = int(above_ta_idx[lidx])
                    if ta == -1:
                        # No marginalization grid coverage for this
                        # trigger's template -- no best-fit orientation
                        # to reconstruct from, so no chisq bins to build
                        # either. Handled below (chisq forced to NaN)
                        # rather than falling back to any other
                        # reconstruction.
                        covered.append(False)
                        bins += [numpy.zeros(self.parse_option(
                            templates[0], self.num_bins) + 1, dtype=int)]
                        scales += [numpy.zeros(num_comps, dtype=complex)]
                        continue
                    covered.append(True)
                    m_row = self._bestfit_row_from_ep_ec(
                        grid, Ep, Ec, ta, above_psi[lidx])
                    # m_row is M[ta_idx,:] (same M as tha_marginalize's
                    # dh = M @ z), NOT the physical combination
                    # coefficients directly: <h_recon|data> = Sum_k
                    # conj(c_k)*<h_k|data>, matched against dh = Sum_k
                    # M[k]*z_k term-by-term gives c_k = conj(M[k]). So
                    # the template needs conj(m_row) (reconstruct_template
                    # uses its `weights` argument un-conjugated, exactly
                    # like the old code passed raw local_snrs = c_k for
                    # the SNR-weighted method) -- and the corr-combining
                    # weight below then needs conj(c_k) = m_row, i.e.
                    # sc.conj() undoes reconstruct_template's internal
                    # conj(m_row) back to m_row. This is the same
                    # template-uses-c_k / corr-uses-conj(c_k) asymmetry
                    # the original SNR-weighted method had (there,
                    # c_k = local_snrs, un-conjugated for the template,
                    # conj(local_snrs) for the corr combine).
                    template, sc = self.reconstruct_template(
                        templates, m_row.conj(), num_comps)
                    bins += [self.calculate_chisq_bins(template)]
                    scales += [sc.conj()]

                bins = numpy.stack(bins, axis=0)
                scales = numpy.stack(scales, axis=0)
                covered = numpy.array(covered)

                dof = (bins.shape[1] - 1) * 2 - 2
                chisq = numpy.zeros(
                    (len(above_indices), bins.shape[1] - 1),
                    dtype=complex_same_precision_as(corrs[0])
                )

                for i in range(num_comps):
                    chisq += shift_multibin(corrs[i], above_indices, scales[:, i], bins)

                # Reference total power for the chisq identity must come
                # from summing THESE SAME bins (this reconstruction's own
                # whole-band response), not from the trigger's separately
                # -measured combined-harmonic SNR (above_snrv) -- that's
                # the response of a DIFFERENT filter (the SNR-maximizing
                # combination; above_snrv = sqrt(Sum|snr_k|^2) is, by
                # Cauchy-Schwarz, the LARGEST total any unit-normalized
                # combination of the 5 harmonics can reach, strictly
                # greater than this filter's actual total for generic
                # grid weights) -- reusing it here would make this
                # chisq come out spuriously negative for ordinary,
                # well-matched signals.
                #
                # With the conjugate convention above correct, this
                # total is provably identical to out_vals['marg_snr']
                # for this same trigger, up to a phase (both a
                # consequence, not a coincidence): total = Sum_i
                # sc.conj()[i]*z_i = Sum_i (m_row[i]/||m_row||)*z_i =
                # dh/||m_row|| (dh = M@z from tha_marginalize, using the
                # SAME m_row row), and ||m_row||^2 = Sum_k|m_row_k|^2 =
                # hh (h_recon's own norm, since the 5 harmonics are
                # orthonormal) -- so |total| = |dh|/sqrt(hh) = marg_snr
                # exactly. Recomputed here from `chisq`'s own bins
                # (rather than reusing the already-stored marg_snr)
                # purely for robustness: it stays correct even if
                # marg_snr's own computation ever drifted out of sync
                # with this one.
                total = numpy.sum(chisq, axis=1)
                chisq = numpy.sum((chisq * chisq.conj()).real, axis=1) * (bins.shape[1] - 1)
                chisq = (chisq - (total.conj() * total).real) * (snr_norm ** 2.0)
                chisq[~covered] = numpy.nan

            if self.snr_threshold:
                if num_above > 0:
                    rchisq[above] = chisq
            else:
                rchisq = chisq

            return rchisq, numpy.repeat(dof, len(indices))
        else:
            return None, None