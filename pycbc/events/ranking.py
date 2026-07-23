""" This module contains functions for calculating single-ifo ranking
statistic values
"""
import logging
from types import SimpleNamespace

import numpy
import h5py
from .ml_stat_conditional import MLStatistic
from pycbc.waveform.bank import TemplateBank, compute_beta

logger = logging.getLogger('pycbc.events.ranking')

# Cache harmonic statistics so the HDF file is only read once
_harmonic_stats_cache = {}
_conditional_flow_cache = {}
_template_beta_cache = {}


def _load_harmonic_stats(harmonic_stats_file):
    """
    Load harmonic means and covariance matrices from an HDF file.

    Results are cached so repeated calls to the ranking function do not
    repeatedly read the file or invert the covariance matrices.
    """
    if harmonic_stats_file not in _harmonic_stats_cache:
        with h5py.File(harmonic_stats_file, "r") as f:
            harmonic_means = numpy.asarray(
                f["means"][:],
                dtype=numpy.float64
            )
            harmonic_covs = numpy.asarray(
                f["covariances"][:],
                dtype=numpy.float64
            )

        harmonic_inv_covs = numpy.linalg.pinv(harmonic_covs)

        _harmonic_stats_cache[harmonic_stats_file] = (
            harmonic_means,
            harmonic_inv_covs
        )

    return _harmonic_stats_cache[harmonic_stats_file]

def mahalanobis_weighted_snr(
        trigs,
        harmonic_means,
        harmonic_inv_covs,
        distance_threshold=2.65,
        **kwargs):

        def get_field(name):
            try:
                return numpy.asarray(trigs[name])
            except (KeyError, TypeError, IndexError):
                return numpy.asarray(getattr(trigs, name))

        snr = get_field("snr").astype(numpy.float64)
        comp1 = get_field("snr_comp_1").astype(numpy.float64)
        comp2 = get_field("snr_comp_2").astype(numpy.float64)
        comp3 = get_field("snr_comp_3").astype(numpy.float64)

        with numpy.errstate(divide="ignore", invalid="ignore"):
            x = numpy.column_stack([
                numpy.log(comp2 / comp1),
                numpy.log(comp3 / comp1),
            ])

        try:
            template_ids = get_field("template_num").astype(numpy.int64)
        except (KeyError, AttributeError):
            template_ids = get_field("template_id").astype(numpy.int64)

        if template_ids.ndim == 0:
            template_ids = numpy.full(
                snr.shape,
                int(template_ids),
                dtype=numpy.int64
            )

        means = numpy.asarray(
            harmonic_means[template_ids, 1:],
            dtype=numpy.float64
        )

        inv_covs = numpy.asarray(
            harmonic_inv_covs[template_ids],
            dtype=numpy.float64
        ).reshape(-1, 2, 2)

        delta = x - means

        d_squared = numpy.einsum(
            "ni,nij,nj->n",
            delta,
            inv_covs,
            delta
        )

        d_squared = numpy.maximum(d_squared, 0.0)
        weights = numpy.ones_like(snr)
        weights = -0.5 * d_squared

        return snr + weights

def _load_conditional_flow(flow_file):
    """Load and cache the conditional normalizing flow."""
    if flow_file not in _conditional_flow_cache:
        _conditional_flow_cache[flow_file] = MLStatistic.from_file(flow_file)

    return _conditional_flow_cache[flow_file]


def _load_template_beta(bank_file):
    """Calculate, fold and cache beta for every template in the bank."""
    if bank_file not in _template_beta_cache:
        bank = TemplateBank(bank_file)
        beta = numpy.empty(len(bank), dtype=numpy.float64)

        for index, template in enumerate(bank.table):
            # compute_beta expects the low-frequency cutoff as ``flow``,
            # while template-bank rows store it as ``f_lower``.
            beta_template = SimpleNamespace(
                mass1=template.mass1,
                mass2=template.mass2,
                spin1x=template.spin1x,
                spin1y=template.spin1y,
                spin1z=template.spin1z,
                spin2x=template.spin2x,
                spin2y=template.spin2y,
                spin2z=template.spin2z,
                flow=template.f_lower,
            )
            beta[index] = compute_beta(beta_template)

        # Fold beta about pi / 2 so the condition lies in [0, pi / 2].
        beta = numpy.where(
            beta > numpy.pi / 2.0,
            numpy.pi - beta,
            beta
        )
        _template_beta_cache[bank_file] = beta[:, None]

    return _template_beta_cache[bank_file]


def conditional_flow_weighted_snr(
        trigs,
        conditional_flow,
        template_beta,
        num_comps=3,
        batch_size=1000000,
        **kwargs):
    """
    Add the beta-conditional flow log probability of the harmonic ratios
    to the trigger SNR.

    The flow sample has ``num_comps - 1`` entries:
    ``log(snr_comp_i / snr_comp_1)`` for harmonics 2 through ``num_comps``.
    The condition is the folded template beta value.
    """

    def get_field(name):
        try:
            return numpy.asarray(trigs[name])
        except (KeyError, TypeError, IndexError):
            return numpy.asarray(getattr(trigs, name))

    if num_comps < 2:
        raise ValueError("num_comps must be at least 2")

    snr = numpy.array(
        get_field("snr"),
        ndmin=1,
        dtype=numpy.float64
    )

    components = [
        numpy.array(
            get_field(f"snr_comp_{index}"),
            ndmin=1,
            dtype=numpy.float64
        )
        for index in range(1, num_comps + 1)
    ]

    for index, component in enumerate(components, start=1):
        if component.size == 1 and snr.size > 1:
            components[index - 1] = numpy.full(
                snr.shape,
                component.item(),
                dtype=numpy.float64
            )
        elif component.size != snr.size:
            raise ValueError(
                f"Got {component.size} values for snr_comp_{index} "
                f"and {snr.size} triggers"
            )

    try:
        template_ids = numpy.array(
            get_field("template_id"),
            ndmin=1,
            dtype=numpy.int64
        )
    except (KeyError, AttributeError, TypeError, IndexError):
        template_ids = numpy.array(
            get_field("template_num"),
            ndmin=1,
            dtype=numpy.int64
        )

    # ReadByTemplate may provide one template number for all triggers.
    if template_ids.size == 1 and snr.size > 1:
        template_ids = numpy.full(
            snr.shape,
            template_ids.item(),
            dtype=numpy.int64
        )

    if template_ids.size != snr.size:
        raise ValueError(
            f"Got {template_ids.size} template IDs for {snr.size} triggers"
        )

    comp1 = components[0]
    with numpy.errstate(divide="ignore", invalid="ignore"):
        ratios = numpy.column_stack([
            numpy.log(component / comp1)
            for component in components[1:]
        ])

    log_prob = numpy.empty(len(ratios), dtype=numpy.float64)

    for start in range(0, len(ratios), batch_size):
        end = min(start + batch_size, len(ratios))

        batch_conditions = numpy.asarray(
            template_beta[template_ids[start:end]],
            dtype=numpy.float64
        )

        log_prob[start:end] = conditional_flow.log_prob(
            ratios[start:end],
            conditional=batch_conditions
        )

    log_prob[~numpy.isfinite(log_prob)] = -numpy.inf

    return snr + log_prob

def effsnr(snr, reduced_x2, fac=250.,
           **kwargs):  # pylint:disable=unused-argument
    """Calculate the effective SNR statistic. See (S5y1 paper) for definition.
    """
    snr = numpy.array(snr, ndmin=1, dtype=numpy.float64)
    rchisq = numpy.array(reduced_x2, ndmin=1, dtype=numpy.float64)
    esnr = snr / (1 + snr ** 2 / fac) ** 0.25 / rchisq ** 0.25

    # If snr input is float, return a float. Otherwise return numpy array.
    if hasattr(snr, '__len__'):
        return esnr
    else:
        return esnr[0]


def newsnr(snr, reduced_x2, q=6., n=2.,
           **kwargs):  # pylint:disable=unused-argument
    """Calculate the re-weighted SNR statistic ('newSNR') from given SNR and
    reduced chi-squared values. See http://arxiv.org/abs/1208.3491 for
    definition. Previous implementation in glue/ligolw/lsctables.py
    """
    nsnr = numpy.array(snr, ndmin=1, dtype=numpy.float64)
    reduced_x2 = numpy.array(reduced_x2, ndmin=1, dtype=numpy.float64)

    # newsnr is only different from snr if reduced chisq > 1
    ind = numpy.where(reduced_x2 > 1.)[0]
    nsnr[ind] *= (0.5 * (1. + reduced_x2[ind] ** (q/n))) ** (-1./q)

    # If snr input is float, return a float. Otherwise return numpy array.
    if hasattr(snr, '__len__'):
        return nsnr
    else:
        return nsnr[0]


def newsnr_sgveto(snr, brchisq, sgchisq, **kwargs):
    """ Combined SNR derived from NewSNR and Sine-Gaussian Chisq"""
    nsnr = numpy.array(
        newsnr(
            snr,
            brchisq,
            **kwargs),
        ndmin=1)
    sgchisq = numpy.array(sgchisq, ndmin=1)
    t = numpy.array(sgchisq > 4, ndmin=1)
    if len(t):
        nsnr[t] = nsnr[t] / (sgchisq[t] / 4.0) ** 0.5

    # If snr input is float, return a float. Otherwise return numpy array.
    if hasattr(snr, '__len__'):
        return nsnr
    else:
        return nsnr[0]


def newsnr_sgveto_psdvar(snr, brchisq, sgchisq, psd_var_val,
                         min_expected_psdvar=0.65,
                         **kwargs):
    """ Combined SNR derived from SNR, reduced Allen chisq, sine-Gaussian chisq and
    PSD variation statistic"""
    # If PSD var is lower than the 'minimum usually expected value' stop this
    # being used in the statistic. This low value might arise because a
    # significant fraction of the "short" PSD period was gated (for instance).
    psd_var_val = numpy.array(psd_var_val, copy=True)
    psd_var_val[psd_var_val < min_expected_psdvar] = 1.
    scaled_snr = snr * (psd_var_val ** -0.5)
    scaled_brchisq = brchisq * (psd_var_val ** -1.)
    nsnr = newsnr_sgveto(
        scaled_snr,
        scaled_brchisq,
        sgchisq,
        **kwargs
    )

    # If snr input is float, return a float. Otherwise return numpy array.
    if hasattr(snr, '__len__'):
        return nsnr
    else:
        return nsnr[0]


def newsnr_sgveto_psdvar_threshold(snr, brchisq, sgchisq, psd_var_val,
                                   min_expected_psdvar=0.65,
                                   brchisq_threshold=10.0,
                                   psd_var_val_threshold=10.0,
                                   **kwargs):
    """ newsnr_sgveto_psdvar with thresholds applied.

    This is the newsnr_sgveto_psdvar statistic with additional options
    to threshold on chi-squared or PSD variation.
    """
    nsnr = newsnr_sgveto_psdvar(
        snr,
        brchisq,
        sgchisq,
        psd_var_val,
        min_expected_psdvar=min_expected_psdvar,
        **kwargs
    )
    nsnr = numpy.array(nsnr, ndmin=1)
    nsnr[brchisq > brchisq_threshold] = 1.
    nsnr[psd_var_val > psd_var_val_threshold] = 1.

    # If snr input is float, return a float. Otherwise return numpy array.
    if hasattr(snr, '__len__'):
        return nsnr
    else:
        return nsnr[0]


def newsnr_sgveto_psdvar_scaled(snr, brchisq, sgchisq, psd_var_val,
                                scaling=0.33, min_expected_psdvar=0.65,
                                **kwargs):
    """ Combined SNR derived from NewSNR, Sine-Gaussian Chisq and scaled PSD
    variation statistic. """
    nsnr = numpy.array(
        newsnr_sgveto(
            snr,
            brchisq,
            sgchisq,
            **kwargs),
        ndmin=1)
    psd_var_val = numpy.array(psd_var_val, ndmin=1, copy=True)
    psd_var_val[psd_var_val < min_expected_psdvar] = 1.

    # Default scale is 0.33 as tuned from analysis of data from O2 chunks
    nsnr = nsnr / psd_var_val ** scaling

    # If snr input is float, return a float. Otherwise return numpy array.
    if hasattr(snr, '__len__'):
        return nsnr
    else:
        return nsnr[0]


def newsnr_sgveto_psdvar_scaled_threshold(snr, bchisq, sgchisq, psd_var_val,
                                          threshold=2.0,
                                          **kwargs):
    """ Combined SNR derived from NewSNR and Sine-Gaussian Chisq, and
    scaled psd variation.
    """
    nsnr = newsnr_sgveto_psdvar_scaled(
        snr,
        bchisq,
        sgchisq,
        psd_var_val,
        **kwargs
    )
    nsnr = numpy.array(nsnr, ndmin=1)
    nsnr[bchisq > threshold] = 1.

    # If snr input is float, return a float. Otherwise return numpy array.
    if hasattr(snr, '__len__'):
        return nsnr
    else:
        return nsnr[0]


def get_snr(trigs, **kwargs):  # pylint:disable=unused-argument
    """
    Return SNR from a trigs/dictionary object

    Parameters
    ----------
    trigs: dict of numpy.ndarrays, h5py group (or similar dict-like object)
        Dictionary-like object holding single detector trigger information.
        'snr' is a required key

    Returns
    -------
    numpy.ndarray
        Array of snr values
    """
    return numpy.array(trigs['snr'][:], ndmin=1, dtype=numpy.float32)


def get_newsnr(trigs, **kwargs):
    """
    Calculate newsnr ('reweighted SNR') for a trigs/dictionary object

    Parameters
    ----------
    trigs: dict of numpy.ndarrays, h5py group (or similar dict-like object)
        Dictionary-like object holding single detector trigger information.
        'chisq_dof', 'snr', and 'chisq' are required keys

    Returns
    -------
    numpy.ndarray
        Array of newsnr values
    """
    dof = 2. * trigs['chisq_dof'][:] - 2.
    nsnr = newsnr(
        trigs['snr'][:],
        trigs['chisq'][:] / dof,
        **kwargs
    )
    return numpy.array(nsnr, ndmin=1, dtype=numpy.float32)


def get_newsnr_sgveto(trigs, **kwargs):
    """
    Calculate newsnr re-weigthed by the sine-gaussian veto

    Parameters
    ----------
    trigs: dict of numpy.ndarrays, h5py group (or similar dict-like object)
        Dictionary-like object holding single detector trigger information.
        'chisq_dof', 'snr', 'sg_chisq' and 'chisq' are required keys

    Returns
    -------
    numpy.ndarray
        Array of newsnr values
    """
    dof = 2. * trigs['chisq_dof'][:] - 2.
    nsnr_sg = newsnr_sgveto(
        trigs['snr'][:],
        trigs['chisq'][:] / dof,
        trigs['sg_chisq'][:],
        **kwargs
    )
    return numpy.array(nsnr_sg, ndmin=1, dtype=numpy.float32)


def get_newsnr_sgveto_psdvar(trigs, **kwargs):
    """
    Calculate snr re-weighted by Allen chisq, sine-gaussian veto and
    psd variation statistic

    Parameters
    ----------
    trigs: dict of numpy.ndarrays
        Dictionary holding single detector trigger information.
    'chisq_dof', 'snr', 'chisq' and 'psd_var_val' are required keys

    Returns
    -------
     numpy.ndarray
        Array of newsnr values
    """
    dof = 2. * trigs['chisq_dof'][:] - 2.
    nsnr_sg_psd = newsnr_sgveto_psdvar(
        trigs['snr'][:],
        trigs['chisq'][:] / dof,
        trigs['sg_chisq'][:],
        trigs['psd_var_val'][:],
        **kwargs
    )
    return numpy.array(nsnr_sg_psd, ndmin=1, dtype=numpy.float32)


def get_newsnr_sgveto_psdvar_threshold(trigs, **kwargs):
    """
    Calculate newsnr re-weighted by the sine-gaussian veto and scaled
    psd variation statistic

    Parameters
    ----------
    trigs: dict of numpy.ndarrays
        Dictionary holding single detector trigger information.
    'chisq_dof', 'snr', 'chisq' and 'psd_var_val' are required keys

    Returns
    -------
     numpy.ndarray
        Array of newsnr values
    """
    dof = 2. * trigs['chisq_dof'][:] - 2.
    nsnr_sg_psdt = newsnr_sgveto_psdvar_threshold(
        trigs['snr'][:], trigs['chisq'][:] / dof,
        trigs['sg_chisq'][:],
        trigs['psd_var_val'][:],
        **kwargs
    )
    return numpy.array(nsnr_sg_psdt, ndmin=1, dtype=numpy.float32)


def get_newsnr_sgveto_psdvar_scaled(trigs, **kwargs):
    """
    Calculate newsnr re-weighted by the sine-gaussian veto and scaled
    psd variation statistic

    Parameters
    ----------
    trigs: dict of numpy.ndarrays
        Dictionary holding single detector trigger information.
    'chisq_dof', 'snr', 'chisq' and 'psd_var_val' are required keys

    Returns
    -------
     numpy.ndarray
        Array of newsnr values
    """
    dof = 2. * trigs['chisq_dof'][:] - 2.
    nsnr_sg_psdscale = newsnr_sgveto_psdvar_scaled(
        trigs['snr'][:],
        trigs['chisq'][:] / dof,
        trigs['sg_chisq'][:],
        trigs['psd_var_val'][:],
        **kwargs
    )
    return numpy.array(nsnr_sg_psdscale, ndmin=1, dtype=numpy.float32)


def get_newsnr_sgveto_psdvar_scaled_threshold(trigs, **kwargs):
    """
    Calculate newsnr re-weighted by the sine-gaussian veto and scaled
    psd variation statistic. A further threshold is applied to the
    reduced chisq.

    Parameters
    ----------
    trigs: dict of numpy.ndarrays
        Dictionary holding single detector trigger information.
    'chisq_dof', 'snr', 'chisq' and 'psd_var_val' are required keys

    Returns
    -------
     numpy.ndarray
        Array of newsnr values
    """
    dof = 2. * trigs['chisq_dof'][:] - 2.
    nsnr_sg_psdt = newsnr_sgveto_psdvar_scaled_threshold(
        trigs['snr'][:],
        trigs['chisq'][:] / dof,
        trigs['sg_chisq'][:],
        trigs['psd_var_val'][:],
        **kwargs
    )
    return numpy.array(nsnr_sg_psdt, ndmin=1, dtype=numpy.float32)

def get_newsnr_sgveto_psdvar_threshold_mahalanobis(
        trigs,
        harmonic_stats_file=None,
        distance_threshold=2.65,
        **kwargs):
    """
    Calculate newsnr re-weighted by the sine-gaussian veto, PSD variation,
    and thresholds, after first applying Mahalanobis weighting to the SNR.
    """

    if harmonic_stats_file is None:
        raise ValueError(
            "newsnr_sgveto_psdvar_threshold_mahalanobis requires "
            "harmonic_stats_file"
        )

    harmonic_means, harmonic_inv_covs = _load_harmonic_stats(
        harmonic_stats_file
    )

    weighted_snr = mahalanobis_weighted_snr(
        trigs,
        harmonic_means=harmonic_means,
        harmonic_inv_covs=harmonic_inv_covs,
        distance_threshold=distance_threshold
    )

    dof = 2. * trigs['chisq_dof'][:] - 2.

    nsnr_sg_psdt = newsnr_sgveto_psdvar_threshold(
        weighted_snr,
        trigs['chisq'][:] / dof,
        trigs['sg_chisq'][:],
        trigs['psd_var_val'][:],
        **kwargs
    )

    return numpy.array(
        nsnr_sg_psdt,
        ndmin=1,
        dtype=numpy.float32
    )

def get_newsnr_sgveto_psdvar_threshold_conditional_flow(
        trigs,
        conditional_flow_file=None,
        template_bank_file=None,
        num_comps=3,
        batch_size=1000000,
        **kwargs):
    """
    Calculate newsnr with the conditional normalizing-flow log probability
    added to the SNR before the standard chi-squared, SG-veto and PSD
    variation reweighting.
    """

    if conditional_flow_file is None:
        raise ValueError(
            "newsnr_sgveto_psdvar_threshold_conditional_flow requires "
            "conditional_flow_file"
        )

    if template_bank_file is None:
        raise ValueError(
            "newsnr_sgveto_psdvar_threshold_conditional_flow requires "
            "template_bank_file"
        )

    conditional_flow = _load_conditional_flow(
        conditional_flow_file
    )

    template_beta = _load_template_beta(
        template_bank_file
    )

    weighted_snr = conditional_flow_weighted_snr(
        trigs,
        conditional_flow=conditional_flow,
        template_beta=template_beta,
        num_comps=num_comps,
        batch_size=batch_size
    )

    dof = 2. * trigs["chisq_dof"][:] - 2.

    nsnr_sg_psdt = newsnr_sgveto_psdvar_threshold(
        weighted_snr,
        trigs["chisq"][:] / dof,
        trigs["sg_chisq"][:],
        trigs["psd_var_val"][:],
        **kwargs
    )

    return numpy.array(
        nsnr_sg_psdt,
        ndmin=1,
        dtype=numpy.float32
    )

sngls_ranking_function_dict = {
    'snr': get_snr,
    'newsnr': get_newsnr,
    'new_snr': get_newsnr,
    'newsnr_sgveto': get_newsnr_sgveto,
    'newsnr_sgveto_psdvar': get_newsnr_sgveto_psdvar,
    'newsnr_sgveto_psdvar_threshold': get_newsnr_sgveto_psdvar_threshold,
    'newsnr_sgveto_psdvar_scaled': get_newsnr_sgveto_psdvar_scaled,
    'newsnr_sgveto_psdvar_scaled_threshold':
    get_newsnr_sgveto_psdvar_scaled_threshold,
    'newsnr_sgveto_psdvar_threshold_mahalanobis':
    get_newsnr_sgveto_psdvar_threshold_mahalanobis,
    'newsnr_sgveto_psdvar_threshold_conditional_flow':
    get_newsnr_sgveto_psdvar_threshold_conditional_flow,
}

# Lists of datasets required in the trigs object for each function
reqd_datasets = {}
reqd_datasets['snr'] = ['snr']
reqd_datasets['newsnr'] = reqd_datasets['snr'] + ['chisq', 'chisq_dof']
reqd_datasets['new_snr'] = reqd_datasets['newsnr']
reqd_datasets['newsnr_sgveto'] = reqd_datasets['newsnr'] + ['sg_chisq']
reqd_datasets['newsnr_sgveto_psdvar'] = \
    reqd_datasets['newsnr_sgveto'] + ['psd_var_val']
reqd_datasets['newsnr_sgveto_psdvar_threshold'] = \
    reqd_datasets['newsnr_sgveto_psdvar']
reqd_datasets['newsnr_sgveto_psdvar_scaled'] = \
    reqd_datasets['newsnr_sgveto_psdvar']
reqd_datasets['newsnr_sgveto_psdvar_scaled_threshold'] = \
    reqd_datasets['newsnr_sgveto_psdvar']
reqd_datasets['newsnr_sgveto_psdvar_threshold_mahalanobis'] = \
    reqd_datasets['newsnr_sgveto_psdvar_threshold'] + [
        'snr_comp_1',
        'snr_comp_2',
        'snr_comp_3',
        'template_id'
    ]
reqd_datasets[
        'newsnr_sgveto_psdvar_threshold_conditional_flow'
    ] = reqd_datasets['newsnr_sgveto_psdvar_threshold'] + [
        'snr_comp_1',
        'snr_comp_2',
        'snr_comp_3',
        'snr_comp_4',
        'snr_comp_5',
        'template_id'
    ]


def get_sngls_ranking_from_trigs(trigs, statname, **kwargs):
    """
    Return ranking for all trigs given a statname.

    Compute the single-detector ranking for a list of input triggers for a
    specific statname.

    Parameters
    -----------
    trigs: dict of numpy.ndarrays, SingleDetTriggers or ReadByTemplate
        Dictionary holding single detector trigger information.
    statname:
        The statistic to use.
    """
    # Identify correct function
    try:
        sngl_func = sngls_ranking_function_dict[statname]
    except KeyError as exc:
        err_msg = 'Single-detector ranking {} not recognized'.format(statname)
        raise ValueError(err_msg) from exc

    # NOTE: In the sngl_funcs all the kwargs are explicitly stated, so any
    #       kwargs sent here must be known to the function.
    return sngl_func(trigs, **kwargs)