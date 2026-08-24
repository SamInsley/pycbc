"""
Offline, PSD-independent precompute of the per-template data needed for
the 5-harmonic precessing-search marginalized-SNR statistic
(``pycbc.filter.tha_marginalize``).

This is the expensive, lalsimulation-dependent half of the pipeline: for
each template, generate the 5 raw (unwhitened) harmonics and, for a grid
of (theta_jn, alpha0) orientations, the exact expansion coefficients of
that orientation's h+/hx in terms of the raw harmonics. Both are
properties of the intrinsic waveform alone and do not depend on any
noise PSD, so they only need computing once per template regardless of
how many different PSDs are later encountered at filtering time (see
pycbc.filter.tha_marginalize.derive_ep_ec_hh for the cheap, PSD-specific
re-derivation done at filter time).

Used by the ``pycbc_make_tha_marginalization_grid`` executable.
See THA_MARG_INFO.md for the full derivation and rationale.
"""
import numpy as np

from pycbc.waveform.bank import PhenomXPTemplate
from pycbc.filter.tha_marginalize import raw_gram_matrix, weighted_correlation


def _shifted_hp_hc(tmpl, theta_jn, alpha0, phi0, df, f_final):
    """gen_hp_hc, wrapped with the same cyclic time shift bank.py's
    gen_harmonics_comp applies (see compute_raw_and_ab's docstring note
    on why this is needed), reusing _PhenomTemplate's own convention
    helper rather than reimplementing it."""
    hp, hc = tmpl.gen_hp_hc(theta_jn, alpha0, phi0, df, f_final)
    return (tmpl._shift_convention(hp.data.data, hp.epoch, df),
            tmpl._shift_convention(hc.data.data, hc.epoch, df))


def get_interpolated_hp_hc(tmpl, theta_jn, alpha0, phi0, df, f_final):
    """
    Same "generate at coarse df, then upsample" shortcut as bank.py's
    get_interpolated_harmonic_comp (interp=True), applied to h+/hx
    separately instead of to the already psi-combined harmonic (there
    is no separate-polarization equivalent in bank.py, since it only
    ever needs the combined harmonic). Reuses _PhenomTemplate's own
    get_interp_df_min/_interpolate_up so the coarse df_min and
    padding/offset conventions are guaranteed to match
    get_interpolated_harmonic_comp exactly rather than being
    reimplemented here.
    """
    df_min = tmpl.get_interp_df_min()
    hp_small, hc_small = _shifted_hp_hc(tmpl, theta_jn, alpha0, phi0, df_min, f_final)
    return tmpl._interpolate_up(hp_small, df), tmpl._interpolate_up(hc_small, df)


def make_theta_alpha_grid(n_theta, n_alpha):
    """
    theta_jn grid: uniform in cos(theta_jn) in (-1, 1) (isotropic-
    orientation prior), midpoint rule.
    alpha0 grid: uniform over the circle [0, 2*pi), midpoint rule.
    Both carry uniform quadrature weight (the prior density has already
    been absorbed into the choice of variable).
    """
    cos_edges = np.linspace(-1., 1., n_theta + 1)
    cos_mid = 0.5 * (cos_edges[:-1] + cos_edges[1:])
    theta_jn = np.arccos(cos_mid)

    alpha0 = (np.arange(n_alpha) + 0.5) * (2 * np.pi / n_alpha)

    theta_grid, alpha_grid = np.meshgrid(theta_jn, alpha0, indexing='ij')
    weights = np.full(theta_grid.size, 1. / theta_grid.size)
    return theta_grid.ravel(), alpha_grid.ravel(), weights


def get_raw_harmonics(tmpl, df, f_final, kmin, kmax, interp=False):
    """
    Return the 5 raw (unwhitened, unnormalized) harmonic frequency
    series as plain complex numpy arrays, band-limited to [kmin, kmax)
    (zeroed elsewhere), in reverse_flag-adjusted order (matching
    pycbc.waveform.bank's hcomp1..hcomp5 ordering before whitening).

    interp must match whatever the grid points (compute_raw_and_ab) were
    built with. interp=False gives ~machine-precision (1e-14) hp_grid =
    A . raws reconstruction fidelity; interp=True gives ~1e-4 relative
    fidelity (still far below any level that matters for a search, but
    not exact) -- see THA_MARG_INFO.md for why, and for why an earlier,
    confounded measurement in this codebase's history over-attributed a
    much larger (500-1000x) discrepancy to interp that actually came
    from a separate, since-fixed bug (grid points missing the
    cyclic_time_shift correction). Note bank.py's actual search-time
    filtering (get_whitened_normalized_comps) hardcodes interp=True
    with no override; using interp=False here is still an excellent
    (~1e-6, per direct measurement) approximation of those real filters
    while giving much better internal reconstruction fidelity, so it
    remains the recommended default. interp=True is offered because
    it is much cheaper to generate and its self-consistency, while not
    exact, is more than good enough for search purposes.
    """
    comps = tmpl.compute_waveform_five_comps(df, f_final, interp=interp)
    if tmpl.reverse_flag:
        comps = comps[::-1]
    raws = []
    for h in comps:
        arr = np.zeros(len(h), dtype=np.complex128)
        arr[kmin:kmax] = h.data[kmin:kmax]
        raws.append(arr)
    return raws


def compute_raw_and_ab(template_params, sample_rate, f_lower, df, f_final,
                        theta_grid, alpha_grid, kmin, kmax, ref_psd,
                        interp=False):
    """
    PSD-independent (expensive) precompute for one template: raw
    harmonics v_1..v_5, and the (n_grid, 5) coefficient matrices A, B
    expressing hp_grid_i, hc_grid_i exactly as linear combinations of
    v_1..v_5.

    Parameters
    ----------
    template_params : row-like object
        Must provide mass1, mass2, spin{1,2}{x,y,z}, latitude, longitude,
        inclination, polarization, orbital_phase, reverse_flag -- i.e.
        a row of a pycbc.waveform.bank.TemplateBank's .table.
    ref_psd : (n_freq,) float array
        Used only as the internal metric to solve for A, B; the result
        is (to the same tolerance as the 5-harmonic reconstruction
        fidelity, typically ~1e-7 or better) independent of this choice.
    interp : bool
        Must be applied identically to both the raw harmonics and the
        grid-point h+/hx generation -- see get_raw_harmonics docstring.

    Returns
    -------
    tmpl : PhenomXPTemplate
    raws : list of 5 (n_freq,) complex arrays
    A, B : (n_grid, 5) complex arrays
    """
    tmpl = PhenomXPTemplate(template_params, sample_rate, f_lower)
    raws = get_raw_harmonics(tmpl, df, f_final, kmin, kmax, interp=interp)
    Graw_ref = raw_gram_matrix(raws, ref_psd, df, kmin, kmax)

    n_grid = len(theta_grid)
    A = np.zeros((n_grid, 5), dtype=np.complex128)
    B = np.zeros((n_grid, 5), dtype=np.complex128)
    n_freq = len(raws[0])
    for i, (theta_jn, alpha0) in enumerate(zip(theta_grid, alpha_grid)):
        # Phenom's FD output does not place the merger at the end of the
        # array (a T-domain convention leaking into the F-domain output);
        # bank.py's gen_harmonics_comp corrects this with a cyclic time
        # shift when building h1..h5. The grid-point waveforms need the
        # exact same correction, or they end up in a different time/phase
        # convention than h1..h5 and the "hp_grid = A . raws" expansion
        # silently fails (same magnitude per harmonic, wrong relative
        # phase -- see THA_MARG_INFO.md).
        if interp:
            hp_fs, hc_fs = get_interpolated_hp_hc(tmpl, theta_jn, alpha0, 0., df, f_final)
        else:
            hp_fs, hc_fs = _shifted_hp_hc(tmpl, theta_jn, alpha0, 0., df, f_final)
        hp_raw = np.zeros(n_freq, dtype=np.complex128)
        hc_raw = np.zeros(n_freq, dtype=np.complex128)
        hp_raw[kmin:kmax] = hp_fs.data[kmin:kmax]
        hc_raw[kmin:kmax] = hc_fs.data[kmin:kmax]

        c_p = np.array([weighted_correlation(v, hp_raw, ref_psd, df, kmin, kmax)
                        for v in raws])
        c_c = np.array([weighted_correlation(v, hc_raw, ref_psd, df, kmin, kmax)
                        for v in raws])
        A[i, :] = np.linalg.solve(Graw_ref, c_p)
        B[i, :] = np.linalg.solve(Graw_ref, c_c)

    return tmpl, raws, A, B
