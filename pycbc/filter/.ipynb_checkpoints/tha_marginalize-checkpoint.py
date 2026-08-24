"""
Runtime support for the 5-harmonic precessing-search marginalized-SNR
statistic (marginalizing over sky-orientation-like nuisance parameters
theta_jn, alpha0, psi, and analytically over distance and phase).

This module intentionally has minimal dependencies (numpy/scipy only,
no lalsimulation) so it can be imported and called cheaply inside
pycbc_inspiral_tha's per-segment filtering loop.

Two kinds of per-template data are involved:

* PSD-independent (computed once, offline, by
  ``pycbc_make_tha_marginalization_grid`` / ``pycbc.waveform.tha_marg_grid``,
  which does need lalsimulation): the raw (unwhitened) harmonics
  ``h1_raw..h5_raw`` and the ``(n_grid, 5)`` coefficient matrices ``A``,
  ``B`` expressing the (theta_jn, alpha0) orientation grid's h+/hx
  exactly as linear combinations of the raw harmonics. These are valid
  for the template regardless of noise PSD and are loaded here via
  ``load_raw_grid_file``.

* PSD-dependent (cheap, recomputed here on the fly for whatever PSD the
  current segment actually has -- no lalsimulation involved, only 5x5
  linear algebra plus small matrix products): the actual orthonormal
  filters' projection coefficients ``Ep``/``Ec`` and self/cross powers
  ``hh_pp``/``hh_cc``/``hh_pc``, folded together with a psi quadrature
  into the final ``M``/``hh``/``ln_weight`` consumed by
  ``marginalize_segment``.

See THA_MARG_INFO.md for the full derivation and rationale.
"""
from pathlib import Path
import numpy as np
import scipy.special
from scipy.integrate import quad
from scipy.interpolate import RectBivariateSpline
from scipy.special import logsumexp
import h5py

D_LUMINOSITY_MAX = 1.5e4  # Mpc
CACHE_DIR = Path(__file__).parent / '.tha_lookup_table_cache'


def euclidean_distance_prior(d_luminosity):
    """Distance prior uniform in luminosity volume (Mpc^3)."""
    return 4 * np.pi * d_luminosity**2


class LookupTable:
    """
    Marginalize the likelihood over distance via a cached interpolation
    table. Ported from cogwheel's
    cogwheel.likelihood.marginalization.lookup_table.LookupTable,
    stripped of the utils.JSONMixin/numba/IPython dependencies pulled in
    by importing the full cogwheel package, and of the 'comoving'
    distance prior (not needed here).
    """
    REFERENCE_DISTANCE = 1.  # Mpc, luminosity distance at which h is defined.
    _Z0 = 10.
    _SIGMAS = 10.

    def __init__(self, d_luminosity_max=D_LUMINOSITY_MAX, shape=(256, 128)):
        self.d_luminosity_prior = euclidean_distance_prior
        self.d_luminosity_max = d_luminosity_max
        self.shape = shape

        self._inverse_volume = 1 / quad(self.d_luminosity_prior,
                                         0, self.d_luminosity_max)[0]

        x_arr = np.linspace(-self._SIGMAS, 0, shape[0])
        y_arr = np.linspace(self._compactify(-self._SIGMAS / self._Z0),
                             1 - 1e-8, shape[1])
        x_grid, y_grid = np.meshgrid(x_arr, y_arr, indexing='ij')

        dh_grid, hh_grid = self._get_dh_hh(x_grid, y_grid)
        table = self._get_table(dh_grid, hh_grid)
        self._interpolated_table = RectBivariateSpline(x_arr, y_arr, table)

    def _cache_key(self):
        return (f"{self.__class__.__name__}_{self.d_luminosity_max}_"
                f"{self.shape[0]}x{self.shape[1]}")

    def _get_table(self, dh_grid, hh_grid):
        CACHE_DIR.mkdir(exist_ok=True)
        cache_file = CACHE_DIR / f"{self._cache_key()}.npy"
        if cache_file.exists():
            return np.load(cache_file)
        table = np.vectorize(self._function)(dh_grid, hh_grid)
        np.save(cache_file, table)
        return table

    def __call__(self, d_h, h_h):
        """log(evidence) - d_h**2 / h_h / 2, evidence marginalized over distance."""
        return self._interpolated_table(*self._get_x_y(d_h, h_h), grid=False)[()]

    def _get_distance_bounds(self, d_h, h_h, sigmas=5.):
        u_peak = d_h / (self.REFERENCE_DISTANCE * h_h)
        delta_u = sigmas / np.sqrt(h_h)
        return np.array([self.REFERENCE_DISTANCE / (u_peak + delta_u),
                          self.REFERENCE_DISTANCE / (u_peak - delta_u)])

    def lnlike_marginalized(self, d_h, h_h):
        """d_h, h_h: inner products (d|h), (h|h) at REFERENCE_DISTANCE (scalars/arrays)."""
        return self(d_h, h_h) + d_h**2 / h_h / 2

    def _function(self, d_h, h_h):
        return np.log(quad(self._function_integrand, 0, self.d_luminosity_max,
                            args=(d_h, h_h),
                            points=self._get_distance_bounds(d_h, h_h))[0]
                       + 1e-100)

    def _function_integrand(self, d_luminosity, d_h, h_h):
        norm_h = np.sqrt(h_h)
        return (self.d_luminosity_prior(d_luminosity) * self._inverse_volume
                * np.exp(-(norm_h * self.REFERENCE_DISTANCE / d_luminosity
                           - d_h / norm_h)**2 / 2))

    def _get_x_y(self, d_h, h_h):
        norm_h = np.sqrt(h_h)
        overlap = d_h / norm_h
        x = np.log(norm_h / (self.d_luminosity_max
                              * (self._SIGMAS + np.abs(overlap))))
        y = self._compactify(overlap / self._Z0)
        return x, y

    def _get_dh_hh(self, x, y):
        overlap = self._uncompactify(y) * self._Z0
        norm_h = (np.exp(x) * self.d_luminosity_max
                  * (self._SIGMAS + np.abs(overlap)))
        d_h = overlap * norm_h
        h_h = norm_h**2
        return d_h, h_h

    @staticmethod
    def _compactify(value):
        return value / (1 + np.abs(value))

    @staticmethod
    def _uncompactify(value):
        return value / (1 - np.abs(value))


class LookupTableMarginalizedPhase22(LookupTable):
    """
    Like LookupTable but additionally marginalizes analytically over
    orbital phase for |m|=2-dominated radiation. d_h is the ABSOLUTE
    VALUE of the complex (d|h).
    """
    def _function_integrand(self, d_luminosity, d_h, h_h):
        return (super()._function_integrand(d_luminosity, d_h, h_h)
                * scipy.special.i0e(d_h * self.REFERENCE_DISTANCE / d_luminosity))


_LOOKUP_TABLE = None


def get_lookup_table():
    """Lazily construct (and cache) the module-level LookupTableMarginalizedPhase22."""
    global _LOOKUP_TABLE
    if _LOOKUP_TABLE is None:
        _LOOKUP_TABLE = LookupTableMarginalizedPhase22()
    return _LOOKUP_TABLE


def marginalize_segment(M, hh, ln_weight, z, lookup_table=None):
    """
    Compute the log-likelihood marginalized over (theta_jn, alpha0, psi,
    distance, phase) for a chunk of per-harmonic complex SNR values.

    Parameters
    ----------
    M : (n_grid, 5) complex array
        Per-grid-point reconstruction coefficients (bank-time precomputed).
    hh : (n_grid,) float array
        Per-grid-point (h|h), independent of the data (bank-time precomputed).
    ln_weight : (n_grid,) float array
        Log of the (theta_jn, alpha0, psi) quadrature weights (shared
        across templates for a fixed grid).
    z : (5,) or (5, n_time) complex array
        snr_comp_1..5 at the sample(s) to evaluate.
    lookup_table : LookupTableMarginalizedPhase22, optional
        Defaults to the shared module-level instance.

    Returns
    -------
    lnl_marg : float or (n_time,) float array
    """
    lookup_table = lookup_table or get_lookup_table()
    z = np.asarray(z)
    scalar_input = (z.ndim == 1)
    if scalar_input:
        z = z[:, None]

    dh = M @ z  # (n_grid, n_time)
    lnl = lookup_table.lnlike_marginalized(np.abs(dh), hh[:, None])
    lnl_marg = logsumexp(ln_weight[:, None] + lnl, axis=0)

    return lnl_marg[0] if scalar_input else lnl_marg


def weighted_correlation(a, b, psd, df, kmin, kmax):
    """
    <a|b>_psd = 4*df*sum(conj(a)*b/psd), for raw (unwhitened) arrays a, b.
    Restricted to [kmin, kmax) to avoid dividing by psd==0 out of band
    (a, b are assumed already zero there, but 0/0 would still give NaN).
    """
    return 4. * df * np.sum(a[kmin:kmax].conj() * b[kmin:kmax] / psd[kmin:kmax])


def raw_gram_matrix(raws, psd, df, kmin, kmax):
    """5x5 Hermitian Gram matrix Graw[j,k] = <v_j|v_k>_psd."""
    n = len(raws)
    G = np.zeros((n, n), dtype=np.complex128)
    for j in range(n):
        for k in range(j, n):
            val = weighted_correlation(raws[j], raws[k], psd, df, kmin, kmax)
            G[j, k] = val
            G[k, j] = val.conjugate()
    return G


def derive_gram_schmidt_transform(Graw, rel_floor=1e-10):
    """
    Derive (T, sigma) such that, with ĥ_j = v_j/(asd*sigma_j),
    e_k = sum_j T[k,j] * ĥ_j reproduces exactly the same modified
    Gram-Schmidt procedure as pycbc.waveform.bank._PhenomTemplate's
    whiten_and_normalize()+orthogonalize() (first vector unmodified,
    sequential orthogonalize-then-renormalize), operating purely on the
    5x5 Gram matrix -- no frequency arrays, no lalsimulation.

    Weakly-precessing templates can have one or more of the 5 raw
    harmonics carry ~zero power under a given PSD (this is exactly why
    the search truncates them via num_comps in the first place; here we
    always keep all 5 slots, so the zero-power case must be handled
    explicitly rather than relying on truncation to avoid it). Any
    harmonic (or intermediate Gram-Schmidt residual) with norm below
    ``rel_floor`` times the largest one is treated as exactly zero
    (decoupled, contributing nothing) instead of raising a
    divide-by-zero.
    """
    n = Graw.shape[0]
    sigma = np.sqrt(np.abs(Graw.diagonal().real))
    tiny = sigma.max() * rel_floor if sigma.max() > 0 else 0.
    usable = sigma > tiny
    sigma_safe = np.where(usable, sigma, 1.)
    Gn = Graw / np.outer(sigma_safe, sigma_safe)
    Gn[~usable, :] = 0.
    Gn[:, ~usable] = 0.

    C = np.eye(n, dtype=np.complex128)  # C[j,:]: coeffs of "arrs[j]" in terms of ĥ_1..ĥ_n

    def gram_of(C):
        return C.conj() @ Gn @ C.T  # M[i,j] = <arrs[i]|arrs[j]>

    M = gram_of(C)
    for i in range(n):
        for j in range(i + 1, n):
            corr = M[i, j]
            C[j, :] = C[j, :] - corr * C[i, :]
        M = gram_of(C)
        for j in range(i + 1, n):
            norm = M[j, j].real
            if norm > tiny ** 2:
                C[j, :] = C[j, :] / norm ** 0.5
            else:
                C[j, :] = 0.
        M = gram_of(C)

    return C, sigma_safe  # T = C; sigma_safe never introduces a NaN downstream


def derive_ep_ec_hh(raws, A, B, psd, df, kmin, kmax):
    """
    Cheap re-derivation of Ep, Ec, hh_pp, hh_cc, hh_pc for an arbitrary
    PSD, from the PSD-independent raws/A/B (see
    pycbc.waveform.tha_marg_grid). No lalsimulation calls; only 5x5
    linear algebra plus (n_grid x 5) matrix products.
    """
    Graw = raw_gram_matrix(raws, psd, df, kmin, kmax)
    T, sigma = derive_gram_schmidt_transform(Graw)
    Gn = Graw / np.outer(sigma, sigma)

    M = (T.conj() @ Gn) * sigma[None, :]  # M[k,j] = <e_k|v_j>_psd

    Ep = A @ M.T
    Ec = B @ M.T

    hh_pp = np.einsum('ij,jk,ik->i', A.conj(), Graw, A).real
    hh_cc = np.einsum('ij,jk,ik->i', B.conj(), Graw, B).real
    hh_pc = np.einsum('ij,jk,ik->i', A.conj(), Graw, B)

    return Ep, Ec, hh_pp, hh_cc, hh_pc


def fold_psi(Ep, Ec, hh_pp, hh_cc, hh_pc, dpsi, weights_ta, n_psi):
    """
    Fold a psi quadrature (psi enters only via a real rotation of two
    already-computed complex numbers -- no new waveform generation) into
    the (theta_jn, alpha0) grid's Ep/Ec/hh, producing the combined
    (n_grid*n_psi, 5) M matrix, (n_grid*n_psi,) hh vector, and
    (n_grid*n_psi,) log-weight vector consumed by marginalize_segment.
    """
    psi_grid = (np.arange(n_psi) + 0.5) * (np.pi / n_psi)
    weight_psi = np.full(n_psi, 1. / n_psi)

    n_ta = Ep.shape[0]
    M = np.zeros((n_ta, n_psi, 5), dtype=np.complex128)
    hh = np.zeros((n_ta, n_psi))
    ln_weight = np.zeros((n_ta, n_psi))
    for p, psi in enumerate(psi_grid):
        delta = psi - dpsi
        c2, s2 = np.cos(2 * delta), np.sin(2 * delta)
        M[:, p, :] = c2[:, None] * Ep.conj() - s2[:, None] * Ec.conj()
        hh[:, p] = (c2 ** 2) * hh_pp - 2 * c2 * s2 * hh_pc.real + (s2 ** 2) * hh_cc
        ln_weight[:, p] = np.log(weights_ta) + np.log(weight_psi[p])

    return M.reshape(-1, 5), hh.reshape(-1), ln_weight.reshape(-1)


def marginalize_for_psd(template_grid, psd, z, n_psi=16, lookup_table=None):
    """
    All-in-one per-segment call: given a template's PSD-independent grid
    data (as returned by one entry of ``load_raw_grid_file``) and the
    segment's actual PSD, cheaply re-derive the orientation/psi grid for
    that PSD and marginalize the given snr_comp_1..5 sample(s).

    Parameters
    ----------
    template_grid : dict
        One value from the dict returned by ``load_raw_grid_file``, with
        keys 'raws', 'A', 'B', 'dpsi', 'weights_ta'.
    psd : (n_freq,) float array
    z : (5,) or (5, n_time) complex array
        snr_comp_1..5.
    n_psi : int
        Psi quadrature resolution.
    lookup_table : LookupTableMarginalizedPhase22, optional

    Returns
    -------
    lnl_marg : float or (n_time,) float array
    """
    Ep, Ec, hh_pp, hh_cc, hh_pc = derive_ep_ec_hh(
        template_grid['raws'], template_grid['A'], template_grid['B'],
        psd, template_grid['df'], template_grid['kmin'], template_grid['kmax'])
    M, hh, ln_weight = fold_psi(Ep, Ec, hh_pp, hh_cc, hh_pc,
                                template_grid['dpsi'], template_grid['weights_ta'],
                                n_psi)
    return marginalize_segment(M, hh, ln_weight, z, lookup_table=lookup_table)


def load_raw_grid_file(path):
    """
    Load an auxiliary per-template PSD-independent grid file (as written
    by ``pycbc_make_tha_marginalization_grid`` /
    ``pycbc.waveform.tha_marg_grid.build_raw_grid_file``).

    Returns
    -------
    grid_by_hash : dict
        Maps int(template_hash) -> dict with keys 'raws' (list of 5
        complex arrays), 'A', 'B' ((n_grid, 5) complex arrays), 'dpsi'
        ((n_grid,) float array), 'weights_ta' ((n_grid,) float array),
        'df', 'kmin', 'kmax' (shared metadata, duplicated per template
        for convenience).
    """
    grid_by_hash = {}
    with h5py.File(path, 'r') as f:
        df = float(f.attrs['df'])
        kmin = int(f.attrs['kmin'])
        kmax = int(f.attrs['kmax'])
        theta_grid = f['theta_grid'][:]
        alpha_grid = f['alpha_grid'][:]
        weights_ta = f['weights_ta'][:]
        template_hash = f['template_hash'][:]
        raws_all = f['raws'][:]      # (n_templates, 5, n_freq) complex
        A_all = f['A'][:]            # (n_templates, n_grid, 5) complex
        B_all = f['B'][:]
        beta_all = f['beta'][:]      # (n_templates,) float

        for i, h in enumerate(template_hash):
            dpsi = _dpsi_grid(theta_grid, alpha_grid, beta_all[i])
            grid_by_hash[int(h)] = dict(
                raws=[raws_all[i, k] for k in range(5)],
                A=A_all[i], B=B_all[i], dpsi=dpsi, weights_ta=weights_ta,
                df=df, kmin=kmin, kmax=kmax)
    return grid_by_hash


def load_raw_grid_file_attrs(path):
    """
    Read just the generation-provenance attrs of a marginalization grid
    file (see pycbc_make_tha_marginalization_grid), without loading any
    of the (potentially large) per-template datasets. Used by
    pycbc_inspiral_tha to check the grid file's generation options
    (sample_rate, delta_f/df, low_frequency_cutoff, interp) match this
    run's before using it.
    """
    with h5py.File(path, 'r') as f:
        return dict(f.attrs)


def _dpsi_grid(theta_grid, alpha_grid, beta):
    """Vectorized-by-loop wrapper around pycbc.waveform.bank._dpsi."""
    from pycbc.waveform.bank import _dpsi
    return np.array([_dpsi(t, a, beta) for t, a in zip(theta_grid, alpha_grid)])
