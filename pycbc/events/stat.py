# Copyright (C) 2016 Alex Nitz
#
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
"""
This module contains functions for calculating coincident ranking statistic
values.
"""
import logging
from hashlib import sha1
from datetime import datetime as dt
import numpy
import h5py

from . import ranking
from . import coinc_rate
from .eventmgr_cython import logsignalrateinternals_computepsignalbins
from .eventmgr_cython import logsignalrateinternals_compute2detrate
from .eventmgr_cython import marglogsignalrateinternals_computepsignalbins

logger = logging.getLogger("pycbc.events.stat")

_allowed_statistic_features = [
    "phasetd",
    "kde",
    "dq",
    "chirp_mass",
    "sensitive_volume",
    "normalize_fit_rate",
]


class Stat(object):
    """Base class which should be extended to provide a statistic"""

    def __init__(self, sngl_ranking, files=None, ifos=None, **kwargs):
        """
        Create a statistic class instance

        Parameters
        ----------
        sngl_ranking: str
            The name of the ranking to use for the single-detector triggers.
        files: list of strs, needed for some statistics
            A list containing the filenames of hdf format files used to help
            construct the coincident statistics. The files must have a 'stat'
            attribute which is used to associate them with the appropriate
            statistic class.
        ifos: list of strs, needed for some statistics
            The list of detector names
        """

        self.files = {}
        files = files or []
        for filename in files:
            with h5py.File(filename, "r") as f:
                stat = f.attrs["stat"]
            if hasattr(stat, "decode"):
                stat = stat.decode()
            if stat in self.files:
                raise RuntimeError(
                    "We already have one file with stat attr = "
                    "%s. Can't provide more than one!" % stat
                )
            logger.info("Found file %s for stat %s", filename, stat)
            self.files[stat] = filename
        # Keep track of when stat files hashes so it can be
        # reloaded if it has changed
        self.file_hashes = self.get_file_hashes()

        # Provide the dtype of the single detector method's output
        # This is used by background estimation codes that need to maintain
        # a buffer of such values.
        self.single_dtype = numpy.float32
        # True if a larger single detector statistic will produce a larger
        # coincident statistic
        self.single_increasing = True

        self.ifos = ifos or []

        self.sngl_ranking = sngl_ranking
        self.sngl_ranking_kwargs = {}
        self.kwargs = {}
        for key, value in kwargs.items():
            if key.startswith("sngl_ranking_"):
                # Note that all the sngl_ranking keywords are floats,
                # so this conversion is safe - if that changes in the future,
                # we may need something more sophisticated
                self.sngl_ranking_kwargs[key[13:]] = float(value)
            else:
                self.kwargs[key] = value

    def get_file_hashes(self):
        """
        Get sha1 hashes for all the files
        """
        logger.debug(
            "Getting file hashes"
        )
        start = dt.now()
        file_hashes = {}
        for stat, filename in self.files.items():
            with open(filename, 'rb') as file_binary:
                file_hashes[stat] = sha1(file_binary.read()).hexdigest()
        logger.debug(
            "Got file hashes for %d files, took %.3es",
            len(self.files),
            (dt.now() - start).total_seconds()
        )
        return file_hashes

    def files_changed(self):
        """
        Compare hashes of files now with the ones we have cached
        """
        changed_file_hashes = self.get_file_hashes()
        for stat, old_hash in self.file_hashes.items():
            if changed_file_hashes[stat] != old_hash:
                logger.info(
                    "%s statistic file %s has changed",
                    ''.join(self.ifos),
                    stat,
                )
            else:
                # Remove the dataset from the dictionary of hashes
                del changed_file_hashes[stat]

        if changed_file_hashes == {}:
            logger.debug(
                "No %s statistic files have changed",
                ''.join(self.ifos)
            )

        return list(changed_file_hashes.keys())

    def check_update_files(self):
        """
        Check whether files associated with the statistic need updated,
        then do so for each file which needs changing
        """
        files_changed = self.files_changed()
        for file_key in files_changed:
            self.update_file(file_key)
        self.file_hashes = self.get_file_hashes()

    def update_file(self, key):
        """
        Update file used in this statistic referenced by key.
        """
        err_msg = "This function is a stub that should be overridden by the "
        err_msg += "sub-classes. You shouldn't be seeing this error!"
        raise NotImplementedError(err_msg)

    def get_sngl_ranking(self, trigs):
        """
        Returns the ranking for the single detector triggers.

        Parameters
        ----------
        trigs: dict of numpy.ndarrays, h5py group or similar dict-like object
            Object holding single detector trigger information.

        Returns
        -------
        numpy.ndarray
            The array of single detector values
        """
        return ranking.get_sngls_ranking_from_trigs(
            trigs,
            self.sngl_ranking,
            **self.sngl_ranking_kwargs
        )

    def single(self, trigs):  # pylint:disable=unused-argument
        """
        Calculate the necessary single detector information

        Parameters
        ----------
        trigs: dict of numpy.ndarrays, h5py group or similar dict-like object
            Object holding single detector trigger information.

        Returns
        -------
        numpy.ndarray
            The array of single detector values
        """
        err_msg = "This function is a stub that should be overridden by the "
        err_msg += "sub-classes. You shouldn't be seeing this error!"
        raise NotImplementedError(err_msg)

    def rank_stat_single(self, single_info):  # pylint:disable=unused-argument
        """
        Calculate the statistic for a single detector candidate

        Parameters
        ----------
        single_info: tuple
            Tuple containing two values. The first is the ifo (str) and the
            second is the single detector triggers.

        Returns
        -------
        numpy.ndarray
            The array of single detector statistics
        """
        err_msg = "This function is a stub that should be overridden by the "
        err_msg += "sub-classes. You shouldn't be seeing this error!"
        raise NotImplementedError(err_msg)

    def rank_stat_coinc(
        self, s, slide, step, to_shift, **kwargs
    ):  # pylint:disable=unused-argument
        """
        Calculate the coincident detection statistic.
        """
        err_msg = "This function is a stub that should be overridden by the "
        err_msg += "sub-classes. You shouldn't be seeing this error!"
        raise NotImplementedError(err_msg)

    def _check_coinc_lim_subclass(self, allowed_names):
        """
        Check that we are not using coinc_lim_for_thresh when not valid.

        coinc_lim_for_thresh is only defined for the statistic it is present
        in. If we subclass, we must check explicitly that it is still valid and
        indicate this in the code. If the code does not have this explicit
        check you will see the failure message here.

        Parameters
        -----------
        allowed_names : list
            list of allowed classes for the specific sub-classed method.
        """
        if type(self).__name__ not in allowed_names:
            err_msg = "This is being called from a subclass which has not "
            err_msg += "been checked for validity with this method. If it is "
            err_msg += "valid for the subclass to come here, include in the "
            err_msg += "list of allowed_names above."
            raise NotImplementedError(err_msg)

    def coinc_lim_for_thresh(
        self, s, thresh, limifo, **kwargs
    ):  # pylint:disable=unused-argument
        """
        Optimization function to identify coincs too quiet to be of interest

        Calculate the required single detector statistic to exceed
        the threshold for each of the input triggers.
        """

        err_msg = "This function is a stub that should be overridden by the "
        err_msg += "sub-classes. You shouldn't be seeing this error!"
        raise NotImplementedError(err_msg)


class QuadratureSumStatistic(Stat):
    """Calculate the quadrature sum coincident detection statistic"""

    def single(self, trigs):
        """
        Calculate the necessary single detector information

        Here just the ranking is computed and returned.

        Parameters
        ----------
        trigs: dict of numpy.ndarrays, h5py group or similar dict-like object
            Object holding single detector trigger information.

        Returns
        -------
        numpy.ndarray
            The array of single detector values
        """
        return self.get_sngl_ranking(trigs)

    def rank_stat_single(self, single_info):
        """
        Calculate the statistic for a single detector candidate

        For this statistic this is just passing through the
        single value, which will be the second entry in the tuple.

        Parameters
        ----------
        single_info: tuple
            Tuple containing two values. The first is the ifo (str) and the
            second is the single detector triggers.

        Returns
        -------
        numpy.ndarray
            The array of single detector statistics
        """
        return single_info[1]

    def rank_stat_coinc(
        self, sngls_list, slide, step, to_shift, **kwargs
    ):  # pylint:disable=unused-argument
        """
        Calculate the coincident detection statistic.

        Parameters
        ----------
        sngls_list: list
            List of (ifo, single detector statistic) tuples
        slide: (unused in this statistic)
        step: (unused in this statistic)
        to_shift: list
            List of integers indicating what multiples of the time shift will
            be applied (unused in this statistic)

        Returns
        -------
        numpy.ndarray
            Array of coincident ranking statistic values
        """
        cstat = sum(sngl[1] ** 2. for sngl in sngls_list) ** 0.5
        # For single-detector "cuts" the single ranking is set to -1
        for sngls in sngls_list:
            cstat[sngls == -1] = 0
        return cstat

    def coinc_lim_for_thresh(
        self, s, thresh, limifo, **kwargs
    ):  # pylint:disable=unused-argument
        """
        Optimization function to identify coincs too quiet to be of interest

        Calculate the required single detector statistic to exceed
        the threshold for each of the input triggers.

        Parameters
        ----------
        s: list
            List of (ifo, single detector statistic) tuples for all detectors
            except limifo.
        thresh: float
            The threshold on the coincident statistic.
        limifo: string
            The ifo for which the limit is to be found.

        Returns
        -------
        numpy.ndarray
            Array of limits on the limifo single statistic to
            exceed thresh.
        """
        # Safety against subclassing and not rethinking this
        allowed_names = ["QuadratureSumStatistic"]
        self._check_coinc_lim_subclass(allowed_names)

        s0 = thresh ** 2. - sum(sngl[1] ** 2. for sngl in s)
        s0[s0 < 0] = 0
        return s0 ** 0.5


class PhaseTDStatistic(QuadratureSumStatistic):
    """
    Statistic that re-weights combined newsnr using coinc parameters.

    The weighting is based on the PDF of time delays, phase differences and
    amplitude ratios between triggers in different ifos.
    """

    def __init__(
        self,
        sngl_ranking,
        files=None,
        ifos=None,
        pregenerate_hist=True,
        **kwargs,
    ):
        """
        Parameters
        ----------
        sngl_ranking: str
            The name of the ranking to use for the single-detector triggers.

        files: list of strs, unused here
            A list containing the filenames of hdf format files used to help
            construct the coincident statistics. The files must have a 'stat'
            attribute which is used to associate them with the appropriate
            statistic class.

        ifos: list of strs, needed here
            The list of detector names

        pregenerate_hist: bool, optional
            If False, do not pregenerate histogram on class instantiation.
            Default is True.
        """
        QuadratureSumStatistic.__init__(
            self, sngl_ranking, files=files, ifos=ifos, **kwargs
        )

        self.single_dtype = [
            ("snglstat", numpy.float32),
            ("coa_phase", numpy.float32),
            ("end_time", numpy.float64),
            ("sigmasq", numpy.float32),
            ("snr", numpy.float32),
        ]

        # Assign attribute so that it can be replaced with other functions
        self.has_hist = False
        self.hist_ifos = None
        self.ref_snr = 5.
        self.relsense = {}
        self.swidth = self.pwidth = self.twidth = None
        self.srbmin = self.srbmax = None
        self.max_penalty = None
        self.pdtype = []
        self.weights = {}
        self.param_bin = {}
        self.two_det_flag = len(ifos) == 2
        self.two_det_weights = {}
        # Some memory
        self.pdif = numpy.zeros(128, dtype=numpy.float64)
        self.tdif = numpy.zeros(128, dtype=numpy.float64)
        self.sdif = numpy.zeros(128, dtype=numpy.float64)
        self.tbin = numpy.zeros(128, dtype=numpy.int32)
        self.pbin = numpy.zeros(128, dtype=numpy.int32)
        self.sbin = numpy.zeros(128, dtype=numpy.int32)

        # Is the histogram needed to be pre-generated?
        hist_needed = pregenerate_hist
        hist_needed &= not len(ifos) == 1
        hist_needed &= (type(self).__name__ == "PhaseTD" or self.kwargs["phasetd"])

        if hist_needed:
            self.get_hist()
        elif len(ifos) == 1:
            # remove all phasetd files from self.files and self.file_hashes,
            # as they are not needed
            for k in list(self.files.keys()):
                if 'phasetd_newsnr' in k:
                    del self.files[k]
                    del self.file_hashes[k]

    def get_hist(self, ifos=None):
        """
        Read in a signal density file for the ifo combination

        Parameters
        ----------
        ifos: list
            The list of ifos. Needed if not given when initializing the class
            instance.
        """
        ifos = ifos or self.ifos

        selected = None
        for name in self.files:
            # Pick out the statistic files that provide phase / time/ amp
            # relationships and match to the ifos in use
            if "phasetd_newsnr" in name:
                ifokey = name.split("_")[2]
                num = len(ifokey) / 2
                if num != len(ifos):
                    continue

                match = [ifo in ifokey for ifo in ifos]
                if False in match:
                    continue
                selected = name
                break

        # If there are other phasetd_newsnr files, they aren't needed.
        # So tidy them out of the self.files dictionary
        rejected = [key for key in self.files.keys()
                    if 'phasetd_newsnr' in key and not key == selected]
        for k in rejected:
            del self.files[k]
            del self.file_hashes[k]

        if selected is None and len(ifos) > 1:
            raise RuntimeError("Couldn't figure out which stat file to use")
        if len(ifos) == 1:
            # We dont need the histogram file, but we are trying to get one
            # just skip it in this case
            return

        logger.info("Using signal histogram %s for ifos %s", selected, ifos)
        weights = {}
        param = {}

        with h5py.File(self.files[selected], "r") as histfile:
            self.hist_ifos = histfile.attrs["ifos"]

            # Patch for pre-hdf5=3.0 histogram files
            try:
                logger.info("Decoding hist ifos ..")
                self.hist_ifos = [i.decode("UTF-8") for i in self.hist_ifos]
            except (UnicodeDecodeError, AttributeError):
                pass

            # Histogram bin attributes
            self.twidth = histfile.attrs["twidth"]
            self.pwidth = histfile.attrs["pwidth"]
            self.swidth = histfile.attrs["swidth"]
            self.srbmin = histfile.attrs["srbmin"]
            self.srbmax = histfile.attrs["srbmax"]
            relfac = histfile.attrs["sensitivity_ratios"]

            for ifo in self.hist_ifos:
                weights[ifo] = histfile[ifo]["weights"][:]
                param[ifo] = histfile[ifo]["param_bin"][:]

        n_ifos = len(self.hist_ifos)

        bin_volume = (self.twidth * self.pwidth * self.swidth) ** (n_ifos - 1)
        self.hist_max = -1. * numpy.inf

        # Read histogram for each ifo, to use if that ifo has smallest SNR in
        # the coinc
        for ifo in self.hist_ifos:

            # renormalise to PDF
            self.weights[ifo] = \
                (weights[ifo] / (weights[ifo].sum() * bin_volume))
            self.weights[ifo] = self.weights[ifo].astype(numpy.float32)

            if param[ifo].dtype == numpy.int8:
                # Older style, incorrectly sorted histogram file
                ncol = param[ifo].shape[1]
                self.pdtype = [
                    ("c%s" % i, param[ifo].dtype) for i in range(ncol)
                ]
                self.param_bin[ifo] = numpy.zeros(
                    len(self.weights[ifo]), dtype=self.pdtype
                )
                for i in range(ncol):
                    self.param_bin[ifo]["c%s" % i] = param[ifo][:, i]

                lsort = self.param_bin[ifo].argsort()
                self.param_bin[ifo] = self.param_bin[ifo][lsort]
                self.weights[ifo] = self.weights[ifo][lsort]
            else:
                # New style, efficient histogram file
                # param bin and weights have already been sorted
                self.param_bin[ifo] = param[ifo]
                self.pdtype = self.param_bin[ifo].dtype

            # Max_penalty is a small number to assigned to any bins without
            # histogram entries. All histograms in a given file have the same
            # min entry by design, so use the min of the last one read in.
            self.max_penalty = self.weights[ifo].min()
            self.hist_max = max(self.hist_max, self.weights[ifo].max())

            if self.two_det_flag:
                # The density of signals is computed as a function of 3 binned
                # parameters: time difference (t), phase difference (p) and
                # SNR ratio (s). These are computed for each combination of
                # detectors, so for detectors 6 differences are needed. However
                # many combinations of these parameters are highly unlikely and
                # no instances of these combinations occurred when generating
                # the statistic files. Rather than storing a bunch of 0s, these
                # values are just not stored at all. This reduces the size of
                # the statistic file, but means we have to identify the correct
                # value to read for every trigger. For 2 detectors we can
                # expand the weights lookup table here, basically adding in all
                # the "0" values. This makes looking up a value in the
                # "weights" table a O(N) rather than O(NlogN) operation. It
                # sacrifices RAM to do this, so is a good tradeoff for 2
                # detectors, but not for 3!
                if not hasattr(self, "c0_size"):
                    self.c0_size = {}
                    self.c1_size = {}
                    self.c2_size = {}

                self.c0_size[ifo] = numpy.int32(
                    2 * (abs(self.param_bin[ifo]["c0"]).max() + 1)
                )
                self.c1_size[ifo] = numpy.int32(
                    2 * (abs(self.param_bin[ifo]["c1"]).max() + 1)
                )
                self.c2_size[ifo] = numpy.int32(
                    2 * (abs(self.param_bin[ifo]["c2"]).max() + 1)
                )

                array_size = [
                    self.c0_size[ifo],
                    self.c1_size[ifo],
                    self.c2_size[ifo],
                ]
                dtypec = self.weights[ifo].dtype
                self.two_det_weights[ifo] = (
                    numpy.zeros(array_size, dtype=dtypec) + self.max_penalty
                )
                id0 = (
                    self.param_bin[ifo]["c0"].astype(numpy.int32)
                    + self.c0_size[ifo] // 2
                )
                id1 = (
                    self.param_bin[ifo]["c1"].astype(numpy.int32)
                    + self.c1_size[ifo] // 2
                )
                id2 = (
                    self.param_bin[ifo]["c2"].astype(numpy.int32)
                    + self.c2_size[ifo] // 2
                )
                self.two_det_weights[ifo][id0, id1, id2] = self.weights[ifo]

        for ifo, sense in zip(self.hist_ifos, relfac):
            self.relsense[ifo] = sense

        self.has_hist = True

    def update_file(self, key):
        """
        Update file used in this statistic.
        If others are used (i.e. this statistic is inherited), they will
        need updated separately
        """
        if 'phasetd_newsnr' in key and not len(self.ifos) == 1:
            if ''.join(sorted(self.ifos)) not in key:
                logger.debug(
                    "%s file is not used for %s statistic",
                    key,
                    ''.join(self.ifos)
                )
                return False
            logger.info(
                "Updating %s statistic %s file",
                ''.join(self.ifos),
                key
            )
            # This is a PhaseTDStatistic file which needs updating
            self.get_hist()
            return True
        return False

    def logsignalrate(self, stats, shift, to_shift):
        """
        Calculate the normalized log rate density of coinc signals via lookup

        Parameters
        ----------
        stats: dict of dicts
            Single-detector quantities for each detector
        shift: numpy array of float
            Time shift vector for each coinc to be ranked
        to_shift: list of ints
            Multiple of the time shift to apply, ordered as self.ifos

        Returns
        -------
        value: log of coinc signal rate density for the given single-ifo
            triggers and time shifts
        """
        # Convert time shift vector to dict, as hist ifos and self.ifos may
        # not be in same order
        to_shift = {ifo: s for ifo, s in zip(self.ifos, to_shift)}

        if not self.has_hist:
            self.get_hist()

        # Figure out which ifo of the contributing ifos has the smallest SNR,
        # to use as reference for choosing the signal histogram.
        snrs = numpy.array(
            [numpy.array(stats[ifo]["snr"], ndmin=1) for ifo in self.ifos]
        )
        smin = snrs.argmin(axis=0)
        # Store a list of the triggers using each ifo as reference
        rtypes = {
            ifo: numpy.where(smin == j)[0] for j, ifo in enumerate(self.ifos)
        }

        # Get reference ifo information
        rate = numpy.zeros(len(shift), dtype=numpy.float32)
        ps = {ifo: numpy.array(stats[ifo]['coa_phase'],
                               dtype=numpy.float32, ndmin=1)
              for ifo in self.ifos}
        ts = {ifo: numpy.array(stats[ifo]['end_time'],
                               dtype=numpy.float64, ndmin=1)
              for ifo in self.ifos}
        ss = {ifo: numpy.array(stats[ifo]['snr'],
                               dtype=numpy.float32, ndmin=1)
              for ifo in self.ifos}
        sigs = {ifo: numpy.array(stats[ifo]['sigmasq'],
                                 dtype=numpy.float32, ndmin=1)
                for ifo in self.ifos}
        for ref_ifo in self.ifos:
            rtype = rtypes[ref_ifo]
            pref = ps[ref_ifo]
            tref = ts[ref_ifo]
            sref = ss[ref_ifo]
            sigref = sigs[ref_ifo]
            senseref = self.relsense[self.hist_ifos[0]]

            binned = []
            other_ifos = [ifo for ifo in self.ifos if ifo != ref_ifo]
            for ifo in other_ifos:
                # Assign cached memory
                length = len(rtype)
                while length > len(self.pdif):
                    newlen = len(self.pdif) * 2
                    self.pdif = numpy.zeros(newlen, dtype=numpy.float64)
                    self.tdif = numpy.zeros(newlen, dtype=numpy.float64)
                    self.sdif = numpy.zeros(newlen, dtype=numpy.float64)
                    self.pbin = numpy.zeros(newlen, dtype=numpy.int32)
                    self.tbin = numpy.zeros(newlen, dtype=numpy.int32)
                    self.sbin = numpy.zeros(newlen, dtype=numpy.int32)

                # Calculate differences
                logsignalrateinternals_computepsignalbins(
                    self.pdif,
                    self.tdif,
                    self.sdif,
                    self.pbin,
                    self.tbin,
                    self.sbin,
                    ps[ifo],
                    ts[ifo],
                    ss[ifo],
                    sigs[ifo],
                    pref,
                    tref,
                    sref,
                    sigref,
                    shift,
                    rtype,
                    self.relsense[ifo],
                    senseref,
                    self.twidth,
                    self.pwidth,
                    self.swidth,
                    to_shift[ref_ifo],
                    to_shift[ifo],
                    length,
                )

                binned += [
                    self.tbin[:length],
                    self.pbin[:length],
                    self.sbin[:length],
                ]

            # Read signal weight from precalculated histogram
            if self.two_det_flag:
                # High-RAM, low-CPU option for two-det
                logsignalrateinternals_compute2detrate(
                    binned[0],
                    binned[1],
                    binned[2],
                    self.c0_size[ref_ifo],
                    self.c1_size[ref_ifo],
                    self.c2_size[ref_ifo],
                    rate,
                    rtype,
                    sref,
                    self.two_det_weights[ref_ifo],
                    self.max_penalty,
                    self.ref_snr,
                    len(rtype),
                )
            else:
                # Low[er]-RAM, high[er]-CPU option for >two det

                # Convert binned to same dtype as stored in hist
                nbinned = numpy.zeros(len(binned[1]), dtype=self.pdtype)
                for i, b in enumerate(binned):
                    nbinned[f"c{i}"] = b

                loc = numpy.searchsorted(self.param_bin[ref_ifo], nbinned)
                loc[loc == len(self.weights[ref_ifo])] = 0
                rate[rtype] = self.weights[ref_ifo][loc]

                # These weren't in our histogram so give them max penalty
                # instead of random value
                missed = numpy.where(
                    self.param_bin[ref_ifo][loc] != nbinned
                )[0]
                rate[rtype[missed]] = self.max_penalty
                # Scale by signal population SNR
                rate[rtype] *= (sref[rtype] / self.ref_snr) ** -4.

        return numpy.log(rate)

    def single(self, trigs):
        """
        Calculate the necessary single detector information

        Here the ranking as well as phase, endtime and sigma-squared values.

        Parameters
        ----------
        trigs: dict of numpy.ndarrays, h5py group or similar dict-like object
            Object holding single detector trigger information. 'snr', 'chisq',
            'chisq_dof', 'coa_phase', 'end_time', and 'sigmasq' are required
            keys.

        Returns
        -------
        numpy.ndarray
            Array of single detector parameter values
        """
        sngl_stat = self.get_sngl_ranking(trigs)
        singles = numpy.zeros(len(sngl_stat), dtype=self.single_dtype)
        singles["snglstat"] = sngl_stat
        singles["coa_phase"] = trigs["coa_phase"][:]
        singles["end_time"] = trigs["end_time"][:]
        singles["sigmasq"] = trigs["sigmasq"][:]
        singles["snr"] = trigs["snr"][:]
        return numpy.array(singles, ndmin=1)

    def rank_stat_single(self, single_info):
        """
        Calculate the statistic for a single detector candidate

        For this statistic this is just passing through the
        single value, which will be the second entry in the tuple.

        Parameters
        ----------
        single_info: tuple
            Tuple containing two values. The first is the ifo (str) and the
            second is the single detector triggers.

        Returns
        -------
        numpy.ndarray
            The array of single detector statistics
        """
        return single_info[1]["snglstat"]

    def rank_stat_coinc(
        self, sngls_list, slide, step, to_shift, **kwargs
    ):  # pylint:disable=unused-argument
        """
        Calculate the coincident detection statistic, defined in Eq 2 of
        [Nitz et al, 2017](https://doi.org/10.3847/1538-4357/aa8f50).
        """
        rstat = sum(s[1]["snglstat"] ** 2 for s in sngls_list)
        cstat = rstat + 2. * self.logsignalrate(
            dict(sngls_list), slide * step, to_shift
        )
        cstat[cstat < 0] = 0
        return cstat**0.5

    def coinc_lim_for_thresh(
        self, sngls_list, thresh, limifo, **kwargs
    ):  # pylint:disable=unused-argument
        """
        Optimization function to identify coincs too quiet to be of interest.
        Calculate the required single detector statistic to exceed the
        threshold for each of the input triggers.
        """
        # Safety against subclassing and not rethinking this
        allowed_names = ["PhaseTDStatistic"]
        self._check_coinc_lim_subclass(allowed_names)

        if not self.has_hist:
            self.get_hist()

        fixed_stat_sq = sum(
            [b["snglstat"] ** 2 for a, b in sngls_list if a != limifo]
        )
        s1 = thresh ** 2. - fixed_stat_sq
        # Assume best case scenario and use maximum signal rate
        s1 -= 2. * self.hist_max
        s1[s1 < 0] = 0
        return s1**0.5


class ExpFitStatistic(PhaseTDStatistic):
    """
    Detection statistic using an exponential falloff noise model.

    Statistic approximates the negative log noise coinc rate density per
    template over single-ifo newsnr values.

    Extra features can be added by supplying keyword arguments when
    initialising.
    """

    def __init__(self, sngl_ranking, files=None, ifos=None, **kwargs):
        """
        Parameters
        ----------
        sngl_ranking: str
            The name of the ranking to use for the single-detector triggers.

        files: list of strs, needed here
            A list containing the filenames of hdf format files used to help
            construct the coincident statistics. The files must have a 'stat'
            attribute which is used to associate them with the appropriate
            statistic class.

        ifos: list of strs, not used here
            The list of detector names
        kwargs: values and features needed for the statistic
        """
        if not files:
            raise RuntimeError("Files not specified")

        PhaseTDStatistic.__init__(
            self, sngl_ranking, files=files, ifos=ifos, **kwargs
        )

        # Get the single-detector rates fit files
        # the stat file attributes are hard-coded as '%{ifo}-fit_coeffs'
        parsed_attrs = [f.split("-") for f in self.files.keys()]
        self.bg_ifos = [
            at[0]
            for at in parsed_attrs
            if (len(at) == 2 and at[1] == "fit_coeffs")
        ]
        if not len(self.bg_ifos):
            raise RuntimeError(
                "None of the statistic files has the required "
                "attribute called {ifo}-fit_coeffs !"
            )

        # Get the single-detector rates fit values
        self.fits_by_tid = {}
        for i in self.bg_ifos:
            self.fits_by_tid[i] = self.assign_fits(i)
            if self.kwargs["normalize_fit_rate"]:
                self.reassign_rate(i)

        # These are important for the coinc_lim_for_thresh method
        # Is the single threshold a < or > limit?
        self.single_increasing = False

        # Things for calculating the best-case scenario for signal rate
        self.max_sigmasq = -numpy.inf
        self.min_snr = numpy.inf

        # Some modifiers for the statistic to get it into a nice range
        self.benchmark_lograte = float(
            self.kwargs.get("benchmark_lograte", -14.6)
        )
        self.min_stat = float(
            self.kwargs.get("minimum_statistic_cutoff", -30.)
        )

        # Modifier to get a sensible value of the fit slope below threshold
        self.alphabelow = float(
            self.kwargs.get("alpha_below_thresh", numpy.inf)
        )

        # This will be used to keep track of the template number being used
        self.curr_tnum = None

        # Applies a constant offset to all statistic values in a given instance.
        # This can be used to e.g. change relative rankings between different
        # event types. Default is zero offset.
        self.stat_correction = float(
            self.kwargs.get("statistic_correction", 0)
        )

        # Go through the keywords and add class information as needed:
        if self.kwargs["sensitive_volume"]:
            # Add network sensitivity benchmark
            self.single_dtype.append(("benchmark_logvol", numpy.float32))
            # benchmark_logvol is a benchmark sensitivity array
            # over template id
            ref_ifos = self.kwargs.get("reference_ifos", "H1,L1").split(",")
            hl_net_med_sigma = numpy.nanmin(
                [self.fits_by_tid[ifo]["median_sigma"] for ifo in ref_ifos],
                axis=0,
            )
            self.benchmark_logvol = 3. * numpy.log(hl_net_med_sigma)

        if self.kwargs["dq"]:
            # Reweight the noise rate by the dq reweighting factor
            self.dq_rates_by_state = {}
            self.dq_bin_by_tid = {}
            self.dq_state_segments = None
            self.low_latency = False
            self.single_dtype.append(('dq_state', int))

            for ifo in self.ifos:
                key = f"{ifo}-dq_stat_info"
                if key in self.files.keys():
                    self.dq_rates_by_state[ifo] = self.assign_dq_rates(key)
                    self.dq_bin_by_tid[ifo] = self.assign_template_bins(key)
                    self.check_low_latency(key)
                    if not self.low_latency:
                        if self.dq_state_segments is None:
                            self.dq_state_segments = {}
                        self.dq_state_segments[ifo] = self.setup_segments(key)

        if self.kwargs["chirp_mass"]:
            # Reweight the signal rate by the chirp mass of the template
            # This may be stored as a float, so cast just in case
            self.mcm = float(self.kwargs.get("max_chirp_mass", numpy.inf))
            self.curr_mchirp = None

        if self.kwargs["kde"]:
            # Reweight the signal rate by a weighting factor from the KDE of
            # a signal population normalised by expected relative rate of noise
            # triggers for a template
            self.kde_names = []
            self.find_kdes()
            self.kde_by_tid = {}
            for kname in self.kde_names:
                self.assign_kdes(kname)

    def assign_template_bins(self, key):
        """
        Assign bin ID values
        Assign each template id to a bin name based on a
        referenced statistic file.

        Parameters
        ----------
        key: str
            statistic file key string

        Returns
        ---------
        bin_dict: dict of strs
            Dictionary containing the bin name for each template id
        """
        ifo = key.split("-")[0]
        with h5py.File(self.files[key], "r") as dq_file:
            tids = []
            bin_nums = []
            bin_grp = dq_file[f"{ifo}/bins"]
            for bin_name in bin_grp.keys():
                bin_tids = bin_grp[f"{bin_name}/tids"][:]
                tids = list(tids) + list(bin_tids.astype(int))
                bin_nums = list(bin_nums) + list([bin_name] * len(bin_tids))

        bin_dict = dict(zip(tids, bin_nums))
        return bin_dict

    def assign_dq_rates(self, key):
        """
        Assign dq values to each time for every bin based on a
        referenced statistic file.

        Parameters
        ----------
        key: str
            statistic file key string

        Returns
        ---------
        dq_dict: dict of {time: dq_value} dicts for each bin
            Dictionary containing the mapping between the time
            and the dq value for each individual bin.

        """
        ifo = key.split("-")[0]
        with h5py.File(self.files[key], "r") as dq_file:
            bin_grp = dq_file[f"{ifo}/bins"]
            dq_dict = {}
            for bin_name in bin_grp.keys():
                dq_dict[bin_name] = bin_grp[f"{bin_name}/dq_rates"][:]

        return dq_dict

    def setup_segments(self, key):
        """
        Store segments from stat file
        """
        ifo = key.split("-")[0]
        with h5py.File(self.files[key], "r") as dq_file:
            ifo_grp = dq_file[ifo]
            dq_state_segs_dict = {}
            for k in ifo_grp["dq_segments"].keys():
                seg_dict = {}
                seg_dict["start"] = \
                    ifo_grp[f"dq_segments/{k}/segment_starts"][:]
                seg_dict["end"] = ifo_grp[f"dq_segments/{k}/segment_ends"][:]
                dq_state_segs_dict[k] = seg_dict

        return dq_state_segs_dict

    def find_dq_noise_rate(self, trigs, dq_state):
        """Get dq values for a specific ifo and dq states"""

        dq_val = numpy.ones(len(dq_state))

        if self.curr_ifo in self.dq_rates_by_state:
            for i, st in enumerate(dq_state):
                if isinstance(self.curr_tnum, numpy.ndarray):
                    bin_name = self.dq_bin_by_tid[self.curr_ifo][self.curr_tnum[i]]
                else:
                    bin_name = self.dq_bin_by_tid[self.curr_ifo][self.curr_tnum]
                dq_val[i] = self.dq_rates_by_state[self.curr_ifo][bin_name][st]
        return dq_val

    def find_dq_state_by_time(self, ifo, times):
        """Get the dq state for an ifo at times"""
        dq_state = numpy.zeros(len(times), dtype=numpy.uint8)
        if ifo in self.dq_state_segments:
            from pycbc.events.veto import indices_within_times

            for k in self.dq_state_segments[ifo]:
                starts = self.dq_state_segments[ifo][k]["start"]
                ends = self.dq_state_segments[ifo][k]["end"]
                inds = indices_within_times(times, starts, ends)
                # states are named in file as 'dq_state_N', need to extract N
                dq_state[inds] = int(k[9:])
        return dq_state

    def check_low_latency(self, key):
        """
        Check if the statistic file indicates low latency mode.
        Parameters
        ----------
        key: str
            Statistic file key string.
        Returns
        -------
        None
        """
        ifo = key.split('-')[0]
        with h5py.File(self.files[key], 'r') as dq_file:
            ifo_grp = dq_file[ifo]
            if 'dq_segments' not in ifo_grp.keys():
                # if segs are not in file, we must be in LL
                if self.dq_state_segments is not None:
                    raise ValueError(
                        'Either all dq stat files must have segments or none'
                    )
                self.low_latency = True
            elif self.low_latency:
                raise ValueError(
                    'Either all dq stat files must have segments or none'
                )

    def reassign_rate(self, ifo):
        """
        Reassign the rate to be number per time rather than an arbitrarily
        normalised number.

        Parameters
        -----------
        ifo: str
            The ifo to consider.
        """
        with h5py.File(self.files[f'{ifo}-fit_coeffs'], 'r') as coeff_file:
            analysis_time = float(coeff_file.attrs['analysis_time'])
            fbt = 'fit_by_template' in coeff_file

        self.fits_by_tid[ifo]['smoothed_rate_above_thresh'] /= analysis_time
        self.fits_by_tid[ifo]['smoothed_rate_in_template'] /= analysis_time
        # The by-template fits may have been stored in the smoothed fits file
        if fbt:
            self.fits_by_tid[ifo]['fit_by_rate_above_thresh'] /= analysis_time
            self.fits_by_tid[ifo]['fit_by_rate_in_template'] /= analysis_time

    def assign_fits(self, ifo):
        """
        Extract fits from single-detector rate fit files

        Parameters
        -----------
        ifo: str
            The detector to get fits for.

        Returns
        -------
        rate_dict: dict
            A dictionary containing the fit information in the `alpha`, `rate`
            and `thresh` keys.
        """
        coeff_file = h5py.File(self.files[f"{ifo}-fit_coeffs"], "r")
        template_id = coeff_file["template_id"][:]
        # the template_ids and fit coeffs are stored in an arbitrary order
        # create new arrays in template_id order for easier recall
        tid_sort = numpy.argsort(template_id)

        fits_by_tid_dict = {}
        fits_by_tid_dict["smoothed_fit_coeff"] = coeff_file["fit_coeff"][:][
            tid_sort
        ]
        fits_by_tid_dict["smoothed_rate_above_thresh"] = coeff_file[
            "count_above_thresh"
        ][:][tid_sort].astype(float)
        fits_by_tid_dict["smoothed_rate_in_template"] = coeff_file[
            "count_in_template"
        ][:][tid_sort].astype(float)
        if self.kwargs["sensitive_volume"]:
            fits_by_tid_dict["median_sigma"] = coeff_file["median_sigma"][:][
                tid_sort
            ].astype(float)

        # The by-template fits may have been stored in the smoothed fits file
        if "fit_by_template" in coeff_file:
            coeff_fbt = coeff_file["fit_by_template"]
            fits_by_tid_dict["fit_by_fit_coeff"] = coeff_fbt["fit_coeff"][:][
                tid_sort
            ]
            fits_by_tid_dict["fit_by_rate_above_thresh"] = coeff_fbt[
                "count_above_thresh"
            ][:][tid_sort].astype(float)
            fits_by_tid_dict["fit_by_rate_in_template"] = coeff_file[
                "count_in_template"
            ][:][tid_sort].astype(float)


        # Keep the fit threshold in fits_by_tid
        fits_by_tid_dict["thresh"] = coeff_file.attrs["stat_threshold"]

        coeff_file.close()

        return fits_by_tid_dict

    def update_file(self, key):
        """
        Update file used in this statistic.
        If others are used (i.e. this statistic is inherited), they will
        need updated separately
        """
        # First - check if the phasetd file has been updated, this is
        # inherited from the PhaseTDStatistic
        if PhaseTDStatistic.update_file(self, key):
            return True

        if key.endswith('-fit_coeffs'):
            # This is a ExpFitStatistic file which needs updating
            # Which ifo is it?
            ifo = key[:2]
            self.fits_by_tid[ifo] = self.assign_fits(ifo)
            if self.kwargs["normalize_fit_rate"]:
                self.reassign_rate(ifo)
            self.get_ref_vals(ifo)
            logger.info(
                "Updating %s statistic %s file",
                ''.join(self.ifos),
                key
            )
            return True

        # Is the key a KDE statistic file that we update here?
        if key.endswith('kde_file'):
            logger.info(
                "Updating %s statistic %s file",
                ''.join(self.ifos),
                key
            )
            kde_style = key.split('-')[0]
            self.assign_kdes(kde_style)
            return True

        # We also need to check if the DQ files have updated
        if key.endswith('dq_stat_info'):
            ifo = key.split('-')[0]
            logger.info(
                "Updating %s statistic %s file",
                ifo,
                key
            )
            self.dq_rates_by_state[ifo] = self.assign_dq_rates(key)
            self.dq_bin_by_tid[ifo] = self.assign_template_bins(key)
            return True

        return False

    def get_ref_vals(self, ifo):
        """
        Get the largest `alpha` value over all templates for given ifo.

        This is stored in `self.alphamax[ifo]` in the class instance.

        Parameters
        -----------
        ifo: str
            The detector to get fits for.
        """
        self.alphamax[ifo] = self.fits_by_tid[ifo]['smoothed_fit_coeff'].max()

    def find_fits(self, trigs):
        """
        Get fit coeffs for a specific ifo and template id(s)

        Parameters
        ----------
        trigs: dict of numpy.ndarrays, h5py group or similar dict-like object
            Object holding single detector trigger information.
            The coincidence executable will always call this using a bunch of
            trigs from a single template, there template_num is stored as an
            attribute and we just return the single value for all templates.
            If multiple templates are in play we must return arrays.

        Returns
        --------
        alphai: float or numpy array
            The alpha fit value(s)
        ratei: float or numpy array
            The rate fit value(s)
        thresh: float or numpy array
            The thresh fit value(s)
        """
        try:
            ifo = trigs.ifo
        except AttributeError:
            ifo = trigs.get('ifo', None)
            if ifo is None:
                ifo = self.ifos[0]
            assert ifo in self.ifos

        # fits_by_tid is a dictionary of dictionaries of arrays
        # indexed by ifo / coefficient name / template_id
        alphai = self.fits_by_tid[ifo]["smoothed_fit_coeff"][self.curr_tnum]
        ratei = self.fits_by_tid[ifo]["smoothed_rate_above_thresh"][self.curr_tnum]
        thresh = self.fits_by_tid[ifo]["thresh"]

        return alphai, ratei, thresh

    def find_kdes(self):
        """
        Find which associated files are for the KDE reweighting
        """
        # The stat file attributes are hard-coded as 'signal-kde_file'
        # and 'template-kde_file'
        parsed_attrs = [f.split("-") for f in self.files.keys()]
        self.kde_names = [
            at[0]
            for at in parsed_attrs
            if (len(at) == 2 and at[1] == "kde_file")
        ]
        assert sorted(self.kde_names) == ["signal", "template"], (
            "Two KDE stat files are required, they should have stat attr "
            "'signal-kde_file' and 'template-kde_file' respectively"
        )

    def assign_kdes(self, kname):
        """
        Extract values from KDE files

        Parameters
        -----------
        kname: str
            Used to label the kde files.
        """
        with h5py.File(self.files[kname + "-kde_file"], "r") as kde_file:
            self.kde_by_tid[kname + "_kdevals"] = kde_file["data_kde"][:]

    def kde_ratio(self):
        """
        Calculate the weighting factor according to the ratio of the
        signal and template KDE lookup tables
        """
        signal_kde = self.kde_by_tid["signal_kdevals"][self.curr_tnum]
        template_kde = self.kde_by_tid["template_kdevals"][self.curr_tnum]

        return numpy.log(signal_kde / template_kde)

    def lognoiserate(self, trigs):
        """
        Calculate the log noise rate density over single-ifo ranking

        Read in single trigger information, make the sngl_stat
        and rescale by the fitted coefficients alpha and rate

        Parameters
        -----------
        trigs: dict of numpy.ndarrays, h5py group or similar dict-like object
            Object holding single detector trigger information.

        Returns
        ---------
        lognoisel: numpy.array
            Array of log noise rate density for each input trigger.
        """
        # What is the template number currently being used?
        try:
            # exists if accessed via coinc_findtrigs, this is a number
            self.curr_tnum = trigs.template_num
        except AttributeError:
            # exists for SingleDetTriggers & pycbc_live get_coinc,
            # this is a numpy array
            self.curr_tnum = trigs["template_id"]

        alphai, ratei, thresh = self.find_fits(trigs)
        sngl_stat = self.get_sngl_ranking(trigs)
        lognoisel = (
            -alphai * (sngl_stat - thresh)
            + numpy.log(alphai)
            + numpy.log(ratei)
        )

        if not numpy.isinf(self.alphabelow):
            # Above the threshold we use the usual fit coefficient (alphai)
            # below threshold use specified alphabelow
            bt = sngl_stat < thresh
            lognoiselbt = (
                -self.alphabelow * (sngl_stat - thresh)
                + numpy.log(self.alphabelow)
                + numpy.log(ratei)
            )
            lognoisel[bt] = lognoiselbt[bt]

        return numpy.array(lognoisel, ndmin=1, dtype=numpy.float32)

    def single(self, trigs):
        """
        Calculate the necessary single detector information

        In this case the ranking rescaled (see the lognoiserate method here).
        with the phase, end time, SNR values added in.

        Parameters
        ----------
        trigs: dict of numpy.ndarrays, h5py group or similar dict-like object
            Object holding single detector trigger information.

        Returns
        -------
        numpy.ndarray
            The array of single detector values
        """

        try:
            self.curr_ifo = trigs.ifo
        except AttributeError:
            self.curr_ifo = trigs.get('ifo', None)
            if self.curr_ifo is None:
                self.curr_ifo = self.ifos[0]
            assert self.curr_ifo in self.ifos
            # Should be exactly one ifo provided
            assert len(numpy.unique(self.curr_ifo)) == 1

        # single-ifo stat = log of noise rate
        sngl_stat = self.lognoiserate(trigs)
        # populate other fields to calculate phase/time/amp consistency
        singles = numpy.zeros(len(sngl_stat), dtype=self.single_dtype)
        singles["coa_phase"] = trigs["coa_phase"][:]
        singles["end_time"] = trigs["end_time"][:]
        singles["snr"] = trigs["snr"][:]
        # Save info about best-case scenario for use later
        if singles["snr"].size:
            self.min_snr = min(singles["snr"].min(), self.min_snr)

        if self.kwargs["sensitive_volume"]:
            # populate fields to allow sensitive volume factor calculation
            singles["sigmasq"] = trigs["sigmasq"][:]
            # Store benchmark log volume as single-ifo information since
            # the ranking methods do not have access to template id
            singles["benchmark_logvol"] = self.benchmark_logvol[self.curr_tnum]

            # Save info about the best-case scenario for use later:
            if singles["sigmasq"].size:
                max_sigsq = numpy.max(singles["sigmasq"])
                self.max_sigmasq = max(max_sigsq, self.max_sigmasq)

        if self.kwargs["chirp_mass"]:
            from pycbc.conversions import mchirp_from_mass1_mass2

            try:
                mass1 = trigs.param['mass1']
                mass2 = trigs.param['mass2']
            except AttributeError:
                mass1 = trigs['mass1']
                mass2 = trigs['mass2']
            self.curr_mchirp = mchirp_from_mass1_mass2(mass1, mass2)

        if self.kwargs["dq"]:
            if self.low_latency:
                # trigs should already have a dq state assigned
                singles['dq_state'] = trigs['dq_state'][:]
            else:
                singles['dq_state'] = self.find_dq_state_by_time(
                    self.curr_ifo, trigs['end_time'][:]
                )
            dq_rate = self.find_dq_noise_rate(trigs, singles['dq_state'])
            dq_rate = numpy.maximum(dq_rate, 1)
            sngl_stat += numpy.log(dq_rate)

        singles["snglstat"] = sngl_stat
        return numpy.array(singles, ndmin=1)

    def sensitive_volume_factor(self, sngls):
        """
        Determine the factor for different network sensitivities

        Assuming a homogeneous distribution of sources, the signal rate
        should be given by the volume of the sphere to which we are
        sensitive.

        Parameters
        ----------
        sngls: tuple
            Single-detector information, tuples of the (ifo, sngl_object),
            where sngl_object is a ReadByTemplate object or similar.

        Returns
        -------
        network_logvol: numpy.array
            Array of values for the network log-volume factor. This is the
            log of the cube of the sensitive distance (sigma), divided by
            a benchmark volume.
        """
        # Get benchmark log volume as single-ifo information :
        # benchmark_logvol for a given template is not ifo-dependent, so
        # choose the first ifo for convenience
        benchmark_logvol = sngls[0][1]["benchmark_logvol"]

        # Benchmark log volume will be the same for all triggers, so if
        # any are nan, they are all nan
        if any(numpy.isnan(benchmark_logvol)):
            # This can be the case in pycbc live if there are no triggers
            # from this template in the trigger fits file. If so, assume 
            # that sigma for the triggers being ranked is
            # representative of the benchmark network.
            return 0

        # Network sensitivity for a given coinc type is approximately
        # determined by the least sensitive ifo
        network_sigmasq = numpy.amin(
            [sngl[1]["sigmasq"] for sngl in sngls], axis=0
        )
        # Volume \propto sigma^3 or sigmasq^1.5
        network_logvol = 1.5 * numpy.log(network_sigmasq) - benchmark_logvol

        return network_logvol

    def logsignalrate_shared(self, sngls_info):
        """
        Calculate the parts of the log signal rate which are shared by
        both the single and coinc ranking statistics

        Parameters
        ----------
        sngls_info: tuple
            Single-detector information, tuples of the (ifo, sngl_object),
            where sngl_object is a ReadByTemplate object or similar.

        Returns
        -------
        sr_factor: numpy.ndarray
            Array of values to be added to the ranking statistic when
            taking various signal rate factors into account
        """
        # Other features affecting the signal rate
        sr_factor = 0
        if self.kwargs["sensitive_volume"]:
            # Sensitive volume - this is proportional to signal rate
            # assuming a homogeneous universe
            sr_factor += self.sensitive_volume_factor(sngls_info)

        if self.kwargs["chirp_mass"]:
            # chirp mass reweighting
            if isinstance(self.curr_mchirp, numpy.ndarray):
                mchirp = numpy.minimum(self.curr_mchirp, self.mcm)
            else:
                # curr_mchirp will be a number
                mchirp = min(self.curr_mchirp, self.mcm)
            sr_factor += numpy.log((mchirp / 20.) ** (11. / 3.))

        if self.kwargs["kde"]:
            # KDE reweighting
            sr_factor += self.kde_ratio()

        return sr_factor

    def rank_stat_single(self, single_info):
        """
        Calculate the statistic for single detector candidates

        Parameters
        ----------
        single_info: tuple
            Tuple containing two values. The first is the ifo (str) and the
            second is the single detector triggers.

        Returns
        -------
        numpy.ndarray
            The array of single detector statistics
        """
        sngls = single_info[1]

        # Basic noise rate: the exp fit rate from the single statistic
        ln_noise_rate = sngls["snglstat"]
        ln_noise_rate -= self.benchmark_lograte

        # Basic signal rate: snr ** -4
        ln_s = -4 * numpy.log(sngls["snr"] / self.ref_snr)
        # Add in the feature-dependent terms to the signal rate
        ln_s += self.logsignalrate_shared((single_info,))

        # Combine the signal and noise rates
        loglr = ln_s - ln_noise_rate

        # Apply statistic correction
        loglr += self.stat_correction

        # cut off underflowing and very small values
        loglr[loglr < self.min_stat] = self.min_stat
        return loglr

    def rank_stat_coinc(
        self, s, slide, step, to_shift, **kwargs
    ):  # pylint:disable=unused-argument
        """
        Calculate the coincident detection statistic.

        Parameters
        ----------
        s: list
            List of (ifo, single detector statistic) tuples
        slide: numpy array
            The number of steps by which to shift each trigger
        step: float
            The size of each step, to be multiplied by to_shift
        to_shift: list
            List of integers indicating what multiples of the time shift will
            be applied
        kwargs: various
            Key-word arguments to define different features and tunable
            values in the statistic. Only time_addition is used here

        Returns
        -------
        loglr: numpy.ndarray
            The ranking statistic for each trigger based on the supplied
            triggers, requested features and keyword arguments

        """
        # ranking statistic is -ln(expected rate density of noise triggers)
        # plus normalization constant
        sngl_dict = {sngl[0]: sngl[1]["snglstat"] for sngl in s}

        # Basic noise rate: sum of noise rates multiplied by the
        # window they can form coincidences in
        ln_noise_rate = coinc_rate.combination_noise_lograte(
            sngl_dict,
            kwargs["time_addition"],
            dets=kwargs.get('dets', None),
        )

        ln_noise_rate -= self.benchmark_lograte

        # Basic option is not to have any signal-based assumptions,
        # so this is zero to begin with
        ln_s = 0

        if self.kwargs["phasetd"]:
            if not self.has_hist:
                self.get_hist()
            # Find total volume of phase-time-amplitude space occupied by
            # noise coincs, so that the logsignalrate function is properly
            # normalized
            # Extent of time-difference space occupied
            noise_twindow = coinc_rate.multiifo_noise_coincident_area(
                self.hist_ifos,
                kwargs["time_addition"],
                dets=kwargs.get('dets', None),
            )
            # Volume is the allowed time difference window, multiplied by 2pi
            # for each phase difference dimension and by allowed range of SNR
            # ratio for each SNR ratio dimension : there are (n_ifos - 1)
            # dimensions for both phase and SNR
            n_ifos = len(self.hist_ifos)
            snr_range = (self.srbmax - self.srbmin) * self.swidth
            hist_vol = noise_twindow * (2. * numpy.pi * snr_range) ** (
                n_ifos - 1
            )
            # Noise PDF is 1/volume, assuming a uniform distribution of noise
            # coincs
            ln_noise_rate -= numpy.log(hist_vol)

            # What is the signal pdf?
            stat = {ifo: st for ifo, st in s}
            ln_s += self.logsignalrate(stat, slide * step, to_shift)

        # Add in the feature-dependent terms to the signal rate
        ln_s += self.logsignalrate_shared(s)

        # Combine the signal and noise rates
        loglr = ln_s - ln_noise_rate

        # Apply statistic correction
        loglr += self.stat_correction

        # cut off underflowing and very small values
        loglr[loglr < self.min_stat] = self.min_stat

        return loglr

    def coinc_lim_for_thresh(
        self, s, thresh, limifo, **kwargs
    ):  # pylint:disable=unused-argument
        """
        Optimization function to identify coincs too quiet to be of interest

        We are trying to get rid of as many events here at the point where
        we can be confident that they will not meet ranking statistic
        thresholds.

        The calculation here is "What is the minimum required snglstat in
        the pivot IFO which could possibly pass the threshold?"

        There are a couple of points to be wary of here, e.g. in the signal
        rate calculation, we take the best-case scenario. By using the
        best-case for signal rate in this calculation, some events are kept
        at this point which are hopeless.

        Parameters
        ----------
        s: list
            List of (ifo, single detector statistic) tuples
        thresh: float
            The threshold value to be checked against
        limifo: string
            The pivot detector, i.e. the one in which the sngl stat must
            reach the value output by ths function
        kwargs: various
            Key-word arguments to define different features and tunable
            values in the statistic. Only time_addition is used here

        Returns
        -------
        numpy.array
            The minimum snglstat values required in the 'pivot' detector
            in order to reach the threshold specified
        """
        # Safety against subclassing and not rethinking this
        allowed_names = ["ExpFitStatistic"]
        self._check_coinc_lim_subclass(allowed_names)

        if thresh <= self.min_stat:
            return numpy.ones(len(s[0][1]["snglstat"])) * numpy.inf

        # Modify the sngls so that the pivot ifo snglstats are zero
        sngl_dict = {sngl[0]: sngl[1]["snglstat"] for sngl in s}
        sngl_dict[limifo] = numpy.zeros(len(s[0][1]))

        # Noise rate calculated as normal. Because of the modification
        # above, this is the rank_stat_coinc noise rate calculation
        # minus the pivot ifo's snglstat
        ln_noise_rate = coinc_rate.combination_noise_lograte(
            sngl_dict, kwargs["time_addition"]
        )
        ln_noise_rate -= self.benchmark_lograte

        # Basic option is not to have any signal-based assumptions,
        # so this is zero to begin with
        ln_s = 0

        if self.kwargs["phasetd"]:
            if not self.has_hist:
                self.get_hist()
            # Assume best-case scenario and use maximum signal rate
            ln_s = numpy.log(
                self.hist_max * (self.min_snr / self.ref_snr) ** -4.
            )

        # Shared info is the same as in the coinc calculation
        ln_s += self.logsignalrate_shared(s)

        # Combine the signal and noise rates
        loglr = ln_s - ln_noise_rate

        # Apply statistic correction
        loglr += self.stat_correction

        # From this combined rate, what is the minimum snglstat value
        # in the pivot IFO needed to reach the threshold?
        return loglr - thresh


class ExpFitCombinedSNR(ExpFitStatistic):
    """
    Reworking of ExpFitStatistic designed to resemble network SNR

    Use a monotonic function of the negative log noise rate density which
    approximates combined (new)snr for coincs with similar newsnr in each ifo
    """

    def __init__(self, sngl_ranking, files=None, ifos=None, **kwargs):
        """
        Parameters
        ----------
        sngl_ranking: str
            The name of the ranking to use for the single-detector triggers.

        files: list of strs, needed here
            A list containing the filenames of hdf format files used to help
            construct the coincident statistics. The files must have a 'stat'
            attribute which is used to associate them with the appropriate
            statistic class.

        ifos: list of strs, not used here
            The list of detector names
        """
        ExpFitStatistic.__init__(
            self, sngl_ranking, files=files, ifos=ifos, **kwargs
        )
        # for low-mass templates the exponential slope alpha \approx 6
        self.alpharef = 6.
        self.single_increasing = True
        self.single_dtype = numpy.float32

        # Modifier to get a sensible value of the fit slope below threshold
        self.alphabelow = float(
            self.kwargs.get("alpha_below_thresh", numpy.inf)
        )

    def single(self, trigs):
        """
        Calculate the necessary single detector information

        Parameters
        ----------
        trigs: dict of numpy.ndarrays, h5py group or similar dict-like object
            Object holding single detector trigger information.

        Returns
        -------
        numpy.ndarray
            The array of single detector values
        """
        logr_n = self.lognoiserate(trigs)
        _, _, thresh = self.find_fits(trigs)
        # shift by log of reference slope alpha
        logr_n += -1. * numpy.log(self.alpharef)
        # add threshold and rescale by reference slope
        stat = thresh - (logr_n / self.alpharef)
        return numpy.array(stat, ndmin=1, dtype=numpy.float32)

    def rank_stat_single(self, single_info):
        """
        Calculate the statistic for single detector candidates

        Parameters
        ----------
        single_info: tuple
            Tuple containing two values. The first is the ifo (str) and the
            second is the single detector triggers.

        Returns
        -------
        numpy.ndarray
            The array of single detector statistics
        """
        if self.single_increasing:
            sngl_multiifo = single_info[1]
        else:
            sngl_multiifo = -1. * single_info[1]
        return sngl_multiifo

    def rank_stat_coinc(
        self, s, slide, step, to_shift, **kwargs
    ):  # pylint:disable=unused-argument
        """
        Calculate the coincident detection statistic.

        Parameters
        ----------
        sngls_list: list
            List of (ifo, single detector statistic) tuples
        slide: (unused in this statistic)
        step: (unused in this statistic)
        to_shift: list
            List of integers indicating what multiples of the time shift will
            be applied (unused in this statistic)

        Returns
        -------
        numpy.ndarray
            Array of coincident ranking statistic values
        """
        # scale by 1/sqrt(number of ifos) to resemble network SNR
        return sum(sngl[1] for sngl in s) / (len(s) ** 0.5)

    def coinc_lim_for_thresh(
        self, s, thresh, limifo, **kwargs
    ):  # pylint:disable=unused-argument
        """
        Optimization function to identify coincs too quiet to be of interest

        Calculate the required single detector statistic to exceed
        the threshold for each of the input triggers.

        Parameters
        ----------
        s: list
            List of (ifo, single detector statistic) tuples for all detectors
            except limifo.
        thresh: float
            The threshold on the coincident statistic.
        limifo: string
            The ifo for which the limit is to be found.

        Returns
        -------
        numpy.ndarray
            Array of limits on the limifo single statistic to
            exceed thresh.
        """
        # Safety against subclassing and not rethinking this
        allowed_names = ["ExpFitCombinedSNR"]
        self._check_coinc_lim_subclass(allowed_names)

        return thresh * ((len(s) + 1) ** 0.5) - sum(sngl[1] for sngl in s)


class PhaseTDTHAStatistic(QuadratureSumStatistic):
    """
    Statistic that re-weights combined newsnr using coinc parameters.

    The weighting is based on the PDF of time delays, phase differences and
    amplitude ratios between triggers in different ifos.
    """

    def __init__(self, sngl_ranking, files=None, ifos=None,
                 pregenerate_hist=True, num_comps=5, **kwargs):
        """
        Create a statistic class instance

        Parameters
        ----------
        sngl_ranking: str
            The name of the ranking to use for the single-detector triggers.

        files: list of strs, unused here
            A list containing the filenames of hdf format files used to help
            construct the coincident statistics. The files must have a 'stat'
            attribute which is used to associate them with the appropriate
            statistic class.

        ifos: list of strs, needed here
            The list of detector names

        pregenerate_hist: bool, optional
            If False, do not pregenerate histogram on class instantiation.
            Default is True.
        """

        QuadratureSumStatistic.__init__(self, sngl_ranking, files=files,
                                        ifos=ifos, **kwargs)

        self.single_dtype = [
            ('snglstat', numpy.float32),
            ('coa_phase', numpy.float32),
            ('coa_phase_comp_1', numpy.float32),
            ('coa_phase_comp_2', numpy.float32),
            ('coa_phase_comp_3', numpy.float32),
            ('coa_phase_comp_4', numpy.float32),
            ('coa_phase_comp_5', numpy.float32),
            ('end_time', numpy.float64),
            ('sigmasq', numpy.float32),
            ('snr', numpy.float32),
            ('snr_comp_1', numpy.float32),
            ('snr_comp_2', numpy.float32),
            ('snr_comp_3', numpy.float32),
            ('snr_comp_4', numpy.float32),
            ('snr_comp_5', numpy.float32)
        ]

        # Assign attribute so that it can be replaced with other functions
        self.has_hist = False
        self.hist_ifos = None
        self.hist_comps = None
        self.ref_snr = 5.0
        self.relsense = {}
        self.swidth = self.pwidth = self.twidth = None
        self.srbmin = self.srbmax = None
        self.max_penalty = None
        self.pdtype = []
        self.weights = {}
        self.param_bin = {}
        self.two_det_flag = (len(ifos) == 2)
        self.two_det_weights = {}
        # Some memory
        self.pdif = numpy.zeros(128, dtype=numpy.float64)
        self.tdif = numpy.zeros(128, dtype=numpy.float64)
        self.sdif = numpy.zeros(128, dtype=numpy.float64)
        self.tbin = numpy.zeros(128, dtype=numpy.int32)
        self.pbin = numpy.zeros(128, dtype=numpy.int32)
        self.sbin = numpy.zeros(128, dtype=numpy.int32)

        self.num_comps = int(num_comps)

        if pregenerate_hist and not len(ifos) == 1:
            self.get_hist()

    def get_hist(self, ifos=None, num_comps=None):
        """
        Read in a signal density file for the ifo combination

        Parameters
        ----------
        ifos: list
            The list of ifos. Needed if not given when initializing the class
            instance.
        """

        ifos = ifos or self.ifos
        num_comps = num_comps or self.num_comps

        selected = None
        for name in self.files:
            # Pick out the statistic files that provide phase / time/ amp
            # relationships and match to the ifos in use
            if 'phasetdtha_newsnr' in name:
                ifokey = name.split('_')[2]
                num = len(ifokey) / 2
                if num != len(ifos):
                    continue

                match = [ifo in ifokey for ifo in ifos]
                if False in match:
                    continue
                selected = name
                break

        if selected is None and len(ifos) > 1:
            raise RuntimeError("Couldn't figure out which stat file to use")

        logging.info("Using signal histogram %s for ifos %s", selected, ifos)
        weights = {}
        param = {}

        with h5py.File(self.files[selected], 'r') as histfile:
            self.hist_ifos = histfile.attrs['ifos']
            self.hist_num_comps = histfile.attrs['num_comps']

            # Patch for pre-hdf5=3.0 histogram files
            try:
                logging.info("Decoding hist ifos ..")
                self.hist_ifos = [i.decode('UTF-8') for i in self.hist_ifos]
            except (UnicodeDecodeError, AttributeError):
                pass

            # Histogram bin attributes
            self.twidth = histfile.attrs['twidth']
            self.pwidth = histfile.attrs['pwidth']
            self.swidth = histfile.attrs['swidth']
            self.srbmin = histfile.attrs['srbmin']
            self.srbmax = histfile.attrs['srbmax']
            relfac = histfile.attrs['sensitivity_ratios']

            for ifo in self.hist_ifos:
                for comp in range(1, self.hist_num_comps + 1):
                    rtype = ifo + '/' + str(comp)
                    weights[rtype] = histfile[rtype]['weights'][:]
                    param[rtype] = histfile[rtype]['param_bin'][:]

        n_ifos = len(self.hist_ifos)

        bin_volume = (self.twidth * self.pwidth * self.swidth) ** (n_ifos - 1)
        self.hist_max = - 1. * numpy.inf

        # Read histogram for each ifo and num_comp,
        # to use if that ifo/num_comp has highest SNR in the coinc
        for ifo in self.hist_ifos:

            for comp in range(1, self.hist_num_comps + 1):

                rtype = ifo + '/' + str(comp)

                # renormalise to PDF
                self.weights[rtype] = \
                        weights[rtype] / (weights[rtype].sum() * bin_volume)
                self.weights[rtype] = self.weights[rtype].astype(numpy.float32)

                # New style, efficient histogram file
                # param bin and weights have already been sorted
                self.param_bin[rtype] = param[rtype]
                self.pdtype = self.param_bin[rtype].dtype

                # Max_penalty is a small number to assigned to any bins without
                # histogram entries. All histograms in a given file have the same
                # min entry by design, so use the min of the last one read in.
                self.max_penalty = self.weights[rtype].min()
                self.hist_max = max(self.hist_max, self.weights[rtype].max())

                if self.two_det_flag:
                    # The density of signals is computed as a function of 3 binned
                    # parameters: time difference (t), phase difference (p) and
                    # SNR ratio (s). These are computed for each combination of
                    # detectors, so for detectors 6 differences are needed. However
                    # many combinations of these parameters are highly unlikely and
                    # no instances of these combinations occurred when generating
                    # the statistic files. Rather than storing a bunch of 0s, these
                    # values are just not stored at all. This reduces the size of
                    # the statistic file, but means we have to identify the correct
                    # value to read for every trigger. For 2 detectors we can
                    # expand the weights lookup table here, basically adding in all
                    # the "0" values. This makes looking up a value in the
                    # "weights" table a O(N) rather than O(NlogN) operation. It
                    # sacrifices RAM to do this, so is a good tradeoff for 2
                    # detectors, but not for 3!
                    if not hasattr(self, 'c0_size'):
                        self.c0_size = {}
                        self.c1_size = {}
                        self.c2_size = {}

                    if not ifo in self.c0_size.keys():
                        self.c0_size[ifo] = {}
                        self.c1_size[ifo] = {}
                        self.c2_size[ifo] = {}

                    self.c0_size[rtype] = numpy.int32(
                        2 * (abs(self.param_bin[rtype]['c0']).max() + 1)
                    )
                    self.c1_size[rtype] = numpy.int32(
                        2 * (abs(self.param_bin[rtype]['c1']).max() + 1)
                    )
                    self.c2_size[rtype] = numpy.int32(
                        2 * (abs(self.param_bin[rtype]['c2']).max() + 1)
                    )

                    array_size = [self.c0_size[rtype],
                                  self.c1_size[rtype],
                                  self.c2_size[rtype]]
                    dtypec = self.weights[rtype].dtype
                    self.two_det_weights[rtype] = \
                            numpy.zeros(array_size, dtype=dtypec) + self.max_penalty
                    id0 = self.param_bin[rtype]['c0'].astype(numpy.int32) \
                          + self.c0_size[rtype] // 2
                    id1 = self.param_bin[rtype]['c1'].astype(numpy.int32) \
                          + self.c1_size[rtype] // 2
                    id2 = self.param_bin[rtype]['c2'].astype(numpy.int32) \
                          + self.c2_size[rtype] // 2
                    self.two_det_weights[rtype][id0, id1, id2] = self.weights[rtype]

        for ifo, sense in zip(self.hist_ifos, relfac):
            self.relsense[ifo] = sense

        self.has_hist = True

    def logsignalrate(self, stats, shift, to_shift):
        """
        Calculate the normalized log rate density of signals via lookup

        Parameters
        ----------
        stats: dict of dicts
            Single-detector quantities for each detector
        shift: numpy array of float
            Time shift vector for each coinc to be ranked
        to_shift: list of ints
            Multiple of the time shift to apply, ordered as self.ifos

        Returns
        -------
        value: log of coinc signal rate density for the given single-ifo
            triggers and time shifts
        """
        # Convert time shift vector to dict, as hist ifos and self.ifos may
        # not be in same order
        to_shift = {ifo: s for ifo, s in zip(self.ifos, to_shift)}

        if not self.has_hist:
            self.get_hist()

        # Figure out which ifo/num_comp of the contributing ifos has the highest SNR,
        # to use as reference for choosing the signal histogram.
        ifo_max_snr = {}
        ifo_max_comp = {}
        for ifo in self.ifos:
            snrs = numpy.array([numpy.array(stats[ifo][f'snr_comp_{comp}'], ndmin=1)
                                for comp in range(1, self.num_comps + 1)])

            ifo_max_snr[ifo] = numpy.max(snrs, axis=0)
            ifo_max_comp[ifo] = numpy.argmax(snrs, axis=0)

        snrs = numpy.array([ifo_max_snr[ifo] for ifo in self.ifos])
        smax = numpy.argmax(snrs, axis=0)

        # Store a list of the triggers using each ifo as reference
        idxs = {}
        ps = {}
        ts = {}
        ss = {}
        sigs = {}
        for j, ifo in enumerate(self.ifos):
            for comp in range(1, self.num_comps + 1):
                rtype = ifo + '/' + str(comp)

                lgc = (smax == j) & (ifo_max_comp[ifo] == (comp - 1))
                idxs[rtype] = numpy.where(lgc)[0]

                # Get reference information
                ps[rtype] = numpy.array(stats[ifo][f'coa_phase_comp_{comp}'], ndmin=1)
                ts[rtype] = numpy.array(stats[ifo]['end_time'], ndmin=1)
                ss[rtype] = numpy.array(stats[ifo][f'snr_comp_{comp}'], ndmin=1)
                sigs[rtype] = numpy.array(stats[ifo]['sigmasq'], ndmin=1)

        rate = numpy.zeros(len(shift), dtype=numpy.float32)

        for rtype, idx in idxs.items():
            ref_ifo = rtype.split('/')[0]
            pref = ps[rtype]
            tref = ts[rtype]
            sref = ss[rtype]
            sigref = sigs[rtype]
            senseref = self.relsense[self.hist_ifos[0]]

            binned = []
            other_ifos = [ifo for ifo in self.ifos if ifo != ref_ifo]
            for ifo in other_ifos:
                otype = ifo + '/' + rtype.split('/')[1]
                # Assign cached memory
                length = len(idx)
                while length > len(self.pdif):
                    newlen = len(self.pdif) * 2
                    self.pdif = numpy.zeros(newlen, dtype=numpy.float64)
                    self.tdif = numpy.zeros(newlen, dtype=numpy.float64)
                    self.sdif = numpy.zeros(newlen, dtype=numpy.float64)
                    self.pbin = numpy.zeros(newlen, dtype=numpy.int32)
                    self.tbin = numpy.zeros(newlen, dtype=numpy.int32)
                    self.sbin = numpy.zeros(newlen, dtype=numpy.int32)

                # Calculate differences
                logsignalrateinternals_computepsignalbins(
                    self.pdif,
                    self.tdif,
                    self.sdif,
                    self.pbin,
                    self.tbin,
                    self.sbin,
                    ps[otype],
                    ts[otype],
                    ss[otype],
                    sigs[otype],
                    pref,
                    tref,
                    sref,
                    sigref,
                    shift,
                    idx,
                    self.relsense[ifo],
                    senseref,
                    self.twidth,
                    self.pwidth,
                    self.swidth,
                    to_shift[ref_ifo],
                    to_shift[ifo],
                    length
                )

                binned += [
                    self.tbin[:length],
                    self.pbin[:length],
                    self.sbin[:length]
                ]

            # Read signal weight from precalculated histogram
            if self.two_det_flag:
                # High-RAM, low-CPU option for two-det
                logsignalrateinternals_compute2detrate(
                    binned[0],
                    binned[1],
                    binned[2],
                    self.c0_size[rtype],
                    self.c1_size[rtype],
                    self.c2_size[rtype],
                    rate,
                    idx,
                    sref,
                    self.two_det_weights[rtype],
                    self.max_penalty,
                    self.ref_snr,
                    len(idx)
                )
            else:
                # Low[er]-RAM, high[er]-CPU option for >two det

                # Convert binned to same dtype as stored in hist
                nbinned = numpy.zeros(len(binned[1]), dtype=self.pdtype)
                for i, b in enumerate(binned):
                    nbinned[f'c{i}'] = b

                loc = numpy.searchsorted(self.param_bin[rtype], nbinned)
                loc[loc == len(self.weights[rtype])] = 0
                rate[idx] = self.weights[rtype][loc]

                # These weren't in our histogram so give them max penalty
                # instead of random value
                missed = numpy.where(
                    self.param_bin[rtype][loc] != nbinned
                )[0]
                rate[idx[missed]] = self.max_penalty
                # Scale by signal population SNR
                rate[idx] *= (sref[idx] / self.ref_snr) ** -4.0

        return numpy.log(rate)

    def single(self, trigs):
        """
        Calculate the necessary single detector information

        Here the ranking as well as phase, endtime and sigma-squared values.

        Parameters
        ----------
        trigs: dict of numpy.ndarrays, h5py group or similar dict-like object
            Object holding single detector trigger information. 'snr', 'chisq',
            'chisq_dof', 'coa_phase', 'end_time', and 'sigmasq' are required
            keys.

        Returns
        -------
        numpy.ndarray
            Array of single detector parameter values
        """
        sngl_stat = self.get_sngl_ranking(trigs)
        singles = numpy.zeros(len(sngl_stat), dtype=self.single_dtype)
        singles['snglstat'] = sngl_stat
        singles['coa_phase'] = trigs['coa_phase'][:]
        singles['coa_phase_comp_1'] = trigs['coa_phase_comp_1'][:]
        singles['coa_phase_comp_2'] = trigs['coa_phase_comp_2'][:]
        singles['coa_phase_comp_3'] = trigs['coa_phase_comp_3'][:]
        singles['coa_phase_comp_4'] = trigs['coa_phase_comp_4'][:]
        singles['coa_phase_comp_5'] = trigs['coa_phase_comp_5'][:]
        singles['end_time'] = trigs['end_time'][:]
        singles['sigmasq'] = trigs['sigmasq'][:]
        singles['snr'] = trigs['snr'][:]
        singles['snr_comp_1'] = trigs['snr_comp_1'][:]
        singles['snr_comp_2'] = trigs['snr_comp_2'][:]
        singles['snr_comp_3'] = trigs['snr_comp_3'][:]
        singles['snr_comp_4'] = trigs['snr_comp_4'][:]
        singles['snr_comp_5'] = trigs['snr_comp_5'][:]
        return numpy.array(singles, ndmin=1)

    def rank_stat_single(self, single_info):
        """
        Calculate the statistic for a single detector candidate

        Parameters
        ----------
        single_info: tuple
            Tuple containing two values. The first is the ifo (str) and the
            second is the single detector triggers.

        Returns
        -------
        numpy.ndarray
            The array of single detector statistics
        """
        return self.single(single_info[1])

    def rank_stat_coinc(self, sngls_list, slide, step, to_shift,
                        **kwargs):  # pylint:disable=unused-argument
        """
        Calculate the coincident detection statistic, defined in Eq 2 of
        [Nitz et al, 2017](https://doi.org/10.3847/1538-4357/aa8f50).
        """
        rstat = sum(s[1]['snglstat'] ** 2 for s in sngls_list)
        cstat = rstat + 2. * self.logsignalrate(dict(sngls_list),
                                                slide * step,
                                                to_shift)
        cstat[cstat < 0] = 0
        return cstat ** 0.5

    def coinc_lim_for_thresh(self, sngls_list, thresh, limifo,
                             **kwargs):  # pylint:disable=unused-argument
        """
        Optimization function to identify coincs too quiet to be of interest.
        Calculate the required single detector statistic to exceed the
        threshold for each of the input triggers.
        """
        # Safety against subclassing and not rethinking this
        allowed_names = ['PhaseTDTHAStatistic']
        self._check_coinc_lim_subclass(allowed_names)

        if not self.has_hist:
            self.get_hist()

        lim_stat = [b['snglstat'] for a, b in sngls_list if a == limifo][0]
        s1 = thresh ** 2. - lim_stat ** 2.
        # Assume best case scenario and use maximum signal rate
        s1 -= 2. * self.hist_max
        s1[s1 < 0] = 0
        return s1 ** 0.5


class PhaseTDMARGStatistic(QuadratureSumStatistic):
    """
    Statistic that re-weights combined newsnr using coinc parameters.

    The weighting is based on the PDF of time delays, phase differences,
    amplitude ratios and best-fit (theta_jn, alpha0, psi) grid points between
    triggers in different ifos. Precessing templates use a normalizing-flow
    model conditioned on the template opening angle beta; aligned-spin
    templates (beta = 0 or pi), for which every grid point is equivalent, use
    a separate flow over time, phase and amplitude ratio only.
    """

    def __init__(self, sngl_ranking, files=None, ifos=None,
                 pregenerate_hist=True, num_comps=5, seed=None, **kwargs):
        """
        Create a statistic class instance

        Parameters
        ----------
        sngl_ranking: str
            The name of the ranking to use for the single-detector triggers.

        files: list of strs, unused here
            A list containing the filenames of hdf format files used to help
            construct the coincident statistics. The files must have a 'stat'
            attribute which is used to associate them with the appropriate
            statistic class.

        ifos: list of strs, needed here
            The list of detector names

        pregenerate_hist: bool, optional
            If False, do not pregenerate histogram on class instantiation.
            Default is True.

        seed: int, optional
            Seed for the random jitter used to dequantize the grid points.
        """

        QuadratureSumStatistic.__init__(self, sngl_ranking, files=files,
                                        ifos=ifos, **kwargs)

        self.single_dtype = [
            ('snglstat', numpy.float32),
            ('coa_phase', numpy.float32),
            ('coa_phase_comp_1', numpy.float32),
            ('coa_phase_comp_2', numpy.float32),
            ('coa_phase_comp_3', numpy.float32),
            ('coa_phase_comp_4', numpy.float32),
            ('coa_phase_comp_5', numpy.float32),
            ('end_time', numpy.float64),
            ('sigmasq', numpy.float32),
            ('snr', numpy.float32),
            ('snr_comp_1', numpy.float32),
            ('snr_comp_2', numpy.float32),
            ('snr_comp_3', numpy.float32),
            ('snr_comp_4', numpy.float32),
            ('snr_comp_5', numpy.float32),
            ('marg_lnl', numpy.float32),
            ('marg_snr', numpy.float32),
            ('marg_phase', numpy.float32),
            ('marg_theta', numpy.float32),
            ('marg_alpha', numpy.float32),
            ('marg_psi', numpy.float32),
            ('beta', numpy.float32),
        ]

        # Random jitter used to dequantize the best-fit grid points
        self.rng = numpy.random.default_rng(
            None if seed is None else int(seed))
        self.has_hist = False
        self.hist_ifos = None
        self.ref_snr = 5.0
        self.relsense = {}

        # Grid spacings used for the dequantizing jitter
        self.dalpha = numpy.pi / 12      # 2 pi / n_alpha
        self.dpsi = numpy.pi / 16        # pi / n_psi
        self.dcostheta = 1. / 12         # 2 / n_theta

        # Some memory
        self.pdif = numpy.zeros(128, dtype=numpy.float64)
        self.tdif = numpy.zeros(128, dtype=numpy.float64)
        self.sdif = numpy.zeros(128, dtype=numpy.float64)
        self.adif = numpy.zeros(128, dtype=numpy.float64)
        self.psidif = numpy.zeros(128, dtype=numpy.float64)
        self.thetadif = numpy.zeros(128, dtype=numpy.float64)

        if pregenerate_hist and not len(ifos) == 1:
            self.get_hist()

    @staticmethod
    def _log_prob_chunked(model, x, conditional=None, chunk=20000):
        """model.log_prob in chunks, to bound the flow's memory use"""
        out = numpy.empty(len(x), dtype=numpy.float64)
        for i in range(0, len(x), chunk):
            c = None if conditional is None else conditional[i:i + chunk]
            out[i:i + chunk] = model.log_prob(x[i:i + chunk], conditional=c)
        return out

    @staticmethod
    def is_aligned(beta):
        """Aligned-spin templates have beta = 0 or pi (no in-plane spin)."""
        beta = numpy.asarray(beta, dtype=numpy.float64)
        return (beta < 1e-6) | (beta > numpy.pi - 1e-6)

    @staticmethod
    def conditional_columns(names, beta, stats, ifos, mask):
        """
        The flow conditioning columns for the coincs in mask, in the order
        the flow was trained with (see --conditional-params in
        pycbc_dtphase_tha): 'beta' is one column, 'c' is one column per ifo
        in the model's ifo order, c = marg_snr^2 / 2 - marg_lnl.
        """
        cols = []
        for p in names:
            if p == 'beta':
                cols.append(beta[mask])
            elif p == 'c':
                for ifo in ifos:
                    ms = numpy.array(stats[ifo]['marg_snr'],
                                     dtype=numpy.float64, ndmin=1)
                    ml = numpy.array(stats[ifo]['marg_lnl'],
                                     dtype=numpy.float64, ndmin=1)
                    cols.append((0.5 * ms ** 2 - ml)[mask])
            else:
                raise ValueError('Unknown flow conditional %s' % p)
        return numpy.column_stack(cols).astype(numpy.float32)

    def get_hist(self, ifos=None, num_comps=None):
        """
        Read in the signal density models for the ifo combination

        Three normalizing-flow models are read: 'phasetdtha_newsnr_<IFOS>' for
        precessing templates, 'phasetdtha_aligned_newsnr_<IFOS>' for
        aligned-spin templates, and 'phasetdtha_noise_newsnr_<IFOS>', the
        noise density of the (dpsi, dtheta) features for precessing
        templates.

        Parameters
        ----------
        ifos: list
            The list of ifos. Needed if not given when initializing the class
            instance.
        """

        ifos = ifos or self.ifos

        if len(ifos) == 1:
            # We dont need the histogram file, but we are trying to get one
            # just skip it in this case
            return

        def select(tag):
            # Pick out the statistic file with this tag for the ifos in use
            for name in self.files:
                if tag not in name:
                    continue
                ifokey = name.split('_')[-1]
                if len(ifokey) / 2 != len(ifos):
                    continue
                if all(ifo in ifokey for ifo in ifos):
                    return name
            raise RuntimeError("Couldn't figure out which %s stat file to use"
                               % tag)

        selected = select('phasetdtha_newsnr')
        selected_aligned = select('phasetdtha_aligned_newsnr')
        selected_noise = select('phasetdtha_noise_newsnr')
        logger.info("Using signal models %s (precessing) and %s (aligned), "
                    "and noise model %s, for ifos %s", selected,
                    selected_aligned, selected_noise, ifos)

        from .ml_stat import MLStatistic

        self.model = MLStatistic.from_file(self.files[selected],
                                           group_name="model")
        self.model_aligned = MLStatistic.from_file(self.files[selected_aligned],
                                                   group_name="model")
        # Noise density of (dpsi, dtheta) per ifo pair, conditioned on beta
        self.model_noise = MLStatistic.from_file(self.files[selected_noise],
                                                 group_name="model")
        # What each flow is conditioned on (saved by pycbc_dtphase_tha*);
        # older files were conditioned on beta
        self.cond_prec = list(self.model.metadata.get('conditional',
                                                      ['beta']))
        self.cond_noise = list(self.model_noise.metadata.get('conditional',
                                                             ['beta']))
        self.model_ifos = self.model.metadata.get("ifos")
        self.model_relfac = self.model.metadata.get("relfac")
        # Maximum flow log density over the training samples (log units)
        self.hist_max = float(self.model.metadata.get("hist_max"))
        self.hist_max_aligned = float(
            self.model_aligned.metadata.get("hist_max"))
        self.hist_ifos = self.model_ifos

        for ifo, sense in zip(self.model_ifos, self.model_relfac):
            self.relsense[ifo] = sense

        self.has_hist = True

    def logsignalrate(self, stats, shift, to_shift):
        """
        Calculate the normalized log rate density of signals via lookup

        Parameters
        ----------
        stats: dict of dicts
            Single-detector quantities for each detector
        shift: numpy array of float
            Time shift vector for each coinc to be ranked
        to_shift: list of ints
            Multiple of the time shift to apply, ordered as self.ifos

        Returns
        -------
        value: log of coinc signal rate density for the given single-ifo
            triggers and time shifts
        """
        # Convert time shift vector to dict, as hist ifos and self.ifos may
        # not be in same order
        to_shift = {ifo: s for ifo, s in zip(self.ifos, to_shift)}

        if not self.has_hist:
            self.get_hist()

        def get(ifo, key, dtype=numpy.float32):
            return numpy.array(stats[ifo][key], dtype=dtype, ndmin=1)

        ifos = self.model_ifos
        ref_ifo = ifos[0]
        other_ifos = [ifo for ifo in ifos if ifo != ref_ifo]

        # Single-ifo information. The SNR is sqrt(2 marg_lnl), the same
        # quantity as the single-ifo noise model.
        ps = {ifo: get(ifo, 'marg_phase') for ifo in ifos}
        ts = {ifo: get(ifo, 'end_time', numpy.float64) for ifo in ifos}
        ss = {ifo: numpy.sqrt(2. * numpy.maximum(
                  get(ifo, 'marg_lnl', numpy.float64), 1e-30)
              ).astype(numpy.float32) for ifo in ifos}
        sigs = {ifo: get(ifo, 'sigmasq') for ifo in ifos}
        mal = {ifo: get(ifo, 'marg_alpha') for ifo in ifos}
        mpsi = {ifo: get(ifo, 'marg_psi') for ifo in ifos}
        mthe = {ifo: get(ifo, 'marg_theta') for ifo in ifos}

        # Dequantize the best-fit grid points: uniform jitter within one grid
        # cell (theta_jn is gridded uniformly in cos(theta_jn))
        for ifo in ifos:
            size = mal[ifo].shape
            mal[ifo] = (mal[ifo] + self.rng.uniform(-.5, .5, size=size)
                        * self.dalpha).astype(numpy.float32)
            mpsi[ifo] = (mpsi[ifo] + self.rng.uniform(-.5, .5, size=size)
                         * self.dpsi).astype(numpy.float32)
            costheta = (numpy.cos(mthe[ifo].astype(numpy.float64))
                        + self.rng.uniform(-.5, .5, size=size)
                        * self.dcostheta)
            mthe[ifo] = numpy.arccos(numpy.clip(costheta, -1., 1.)
                                     ).astype(numpy.float32)

        # Differences relative to the reference ifo
        length = len(ts[ref_ifo])
        while length > len(self.pdif):
            newlen = len(self.pdif) * 2
            self.pdif = numpy.zeros(newlen, dtype=numpy.float64)
            self.tdif = numpy.zeros(newlen, dtype=numpy.float64)
            self.sdif = numpy.zeros(newlen, dtype=numpy.float64)
            self.adif = numpy.zeros(newlen, dtype=numpy.float64)
            self.psidif = numpy.zeros(newlen, dtype=numpy.float64)
            self.thetadif = numpy.zeros(newlen, dtype=numpy.float64)

        binned = []
        for ifo in other_ifos:
            marglogsignalrateinternals_computepsignalbins(
                self.pdif,
                self.tdif,
                self.sdif,
                self.adif,
                self.psidif,
                self.thetadif,
                ps[ifo],
                ts[ifo],
                ss[ifo],
                mal[ifo],
                mpsi[ifo],
                mthe[ifo],
                sigs[ifo],
                ps[ref_ifo],
                ts[ref_ifo],
                ss[ref_ifo],
                mal[ref_ifo],
                mpsi[ref_ifo],
                mthe[ref_ifo],
                sigs[ref_ifo],
                shift,
                self.relsense[ifo],
                self.relsense[ref_ifo],
                to_shift[ref_ifo],
                to_shift[ifo],
                length,
            )
            binned += [
                self.tdif[:length].copy(),
                self.pdif[:length].copy(),
                self.sdif[:length].copy(),
                self.adif[:length].copy(),
                self.psidif[:length].copy(),
                self.thetadif[:length].copy(),
            ]
        x = numpy.column_stack(binned)

        # The template (and so beta) is the same in every ifo
        beta = get(ref_ifo, 'beta', numpy.float64)
        aligned = self.is_aligned(beta)
        rate = numpy.zeros(length, dtype=numpy.float64)

        # Precessing templates: all features, conditioned on beta
        prec = ~aligned
        if prec.any():
            xp = x[prec]
            rate_p = self._log_prob_chunked(
                self.model, xp,
                conditional=self.conditional_columns(self.cond_prec, beta,
                                                     stats, ifos, prec))
            # Jacobian from log SNR ratio to SNR ratio
            rate_p -= xp[:, 2::6].sum(axis=1)
            # Correct for each ifo having been marginalized over the
            # orientation grid separately
            margs = numpy.array([get(ifo, 'marg_lnl', numpy.float64)[prec]
                                 for ifo in ifos])
            max_margs = numpy.array([get(ifo, 'marg_snr', numpy.float64)[prec]
                                     for ifo in ifos])
            # rate_p += numpy.min(0.5 * max_margs ** 2 - margs, axis=0)
            # Divide by the noise density of (dpsi, dtheta): noise coincs are
            # not uniform in these (both ifos favour the same grid points of
            # the same template), so they are modelled by their own flow and
            # left out of the uniform noise volume in rank_stat_coinc.
            # Columns 4, 5 of each 6-feature pair, in the noise model's
            # [dpsi, dtheta] order.
            noise_cols = [c for i in range(len(other_ifos))
                          for c in (6 * i + 4, 6 * i + 5)]
            if getattr(self, 'uniform_noise', False):
                # uniform over the noise flow's bounds: density 1/pi^2 per
                # ifo pair
                rate_p += 2. * numpy.log(numpy.pi) * len(other_ifos)
            else:
                rate_p -= self._log_prob_chunked(
                    self.model_noise, xp[:, noise_cols],
                    conditional=self.conditional_columns(self.cond_noise,
                                                         beta, stats, ifos,
                                                         prec))
            rate[prec] = rate_p

        # Aligned-spin templates: time, phase and SNR ratio only. All grid
        # points are equivalent, so use the phase of the first harmonic and
        # no correction term (it is zero).
        if aligned.any():
            ph1 = {ifo: get(ifo, 'coa_phase_comp_1', numpy.float64)[aligned]
                   for ifo in ifos}
            feats = []
            for i, ifo in enumerate(other_ifos):
                pdif = numpy.mod(ph1[ref_ifo] - ph1[ifo], 2. * numpy.pi)
                feats += [x[aligned, 6 * i], pdif, x[aligned, 6 * i + 2]]
            xa = numpy.column_stack(feats)
            rate_a = self._log_prob_chunked(self.model_aligned, xa)
            # Jacobian from log SNR ratio to SNR ratio
            rate_a -= xa[:, 2::3].sum(axis=1)
            rate[aligned] = rate_a

        # Jacobian for the reference SNR and SNR^-4 signal population
        snr_ref = ss[ref_ifo].astype(numpy.float64)
        rate -= numpy.log(snr_ref)
        rate += numpy.log((snr_ref / self.ref_snr) ** -4.)

        return rate

    def single(self, trigs):
        """
        Calculate the necessary single detector information

        Here the ranking as well as phase, endtime and sigma-squared values.

        Parameters
        ----------
        trigs: dict of numpy.ndarrays, h5py group or similar dict-like object
            Object holding single detector trigger information. 'snr', 'chisq',
            'chisq_dof', 'coa_phase', 'end_time', and 'sigmasq' are required
            keys.

        Returns
        -------
        numpy.ndarray
            Array of single detector parameter values
        """
        sngl_stat = self.get_sngl_ranking(trigs)
        singles = numpy.zeros(len(sngl_stat), dtype=self.single_dtype)
        singles['snglstat'] = sngl_stat
        singles['coa_phase'] = trigs['coa_phase'][:]
        singles['coa_phase_comp_1'] = trigs['coa_phase_comp_1'][:]
        singles['coa_phase_comp_2'] = trigs['coa_phase_comp_2'][:]
        singles['coa_phase_comp_3'] = trigs['coa_phase_comp_3'][:]
        singles['coa_phase_comp_4'] = trigs['coa_phase_comp_4'][:]
        singles['coa_phase_comp_5'] = trigs['coa_phase_comp_5'][:]
        singles['end_time'] = trigs['end_time'][:]
        singles['sigmasq'] = trigs['sigmasq'][:]
        singles['snr'] = trigs['snr'][:]
        singles['snr_comp_1'] = trigs['snr_comp_1'][:]
        singles['snr_comp_2'] = trigs['snr_comp_2'][:]
        singles['snr_comp_3'] = trigs['snr_comp_3'][:]
        singles['snr_comp_4'] = trigs['snr_comp_4'][:]
        singles['snr_comp_5'] = trigs['snr_comp_5'][:]
        singles['marg_lnl'] = trigs['marg_lnl'][:]
        singles['marg_snr'] = trigs['marg_snr'][:]
        singles['marg_phase'] = trigs['marg_phase'][:]
        singles['marg_theta'] = trigs['marg_theta'][:]
        singles['marg_alpha'] = trigs['marg_alpha'][:]
        singles['marg_psi'] = trigs['marg_psi'][:]
        singles['beta'] = trigs['beta'][:]
        return numpy.array(singles, ndmin=1)

    def rank_stat_single(self, single_info):
        """
        Calculate the statistic for a single detector candidate

        Parameters
        ----------
        single_info: tuple
            Tuple containing two values. The first is the ifo (str) and the
            second is the single detector triggers.

        Returns
        -------
        numpy.ndarray
            The array of single detector statistics
        """
        return self.single(single_info[1])

    def rank_stat_coinc(self, sngls_list, slide, step, to_shift,
                        **kwargs):  # pylint:disable=unused-argument
        """
        Calculate the coincident detection statistic, defined in Eq 2 of
        [Nitz et al, 2017](https://doi.org/10.3847/1538-4357/aa8f50).
        """
        rstat = sum(s[1]['snglstat'] ** 2 for s in sngls_list)
        cstat = rstat + 2. * self.logsignalrate(dict(sngls_list),
                                                slide * step,
                                                to_shift)
        cstat[cstat < 0] = 0
        return cstat ** 0.5

    def coinc_lim_for_thresh(self, sngls_list, thresh, limifo,
                             **kwargs):  # pylint:disable=unused-argument
        """
        Optimization function to identify coincs too quiet to be of interest.
        Calculate the required single detector statistic to exceed the
        threshold for each of the input triggers.
        """
        # Safety against subclassing and not rethinking this
        allowed_names = ['PhaseTDMARGStatistic']
        self._check_coinc_lim_subclass(allowed_names)

        if not self.has_hist:
            self.get_hist()

        lim_stat = [b['snglstat'] for a, b in sngls_list if a == limifo][0]
        # No reliable upper bound on logsignalrate yet (correction term and
        # SNR^-4 term are not included in hist_max), so do not pre-filter.
        return numpy.zeros_like(lim_stat)

# FIXME: This class (and probably the one above) need properly integrating into
#        the new statistic system. Much duplication present here.
class ExpFitFgBgNormTHAStatistic(PhaseTDTHAStatistic,
                                 ExpFitStatistic):
    """
    Statistic combining PhaseTD, ExpFitBg and additional foreground info.
    """

    def __init__(self, sngl_ranking, files=None, ifos=None,
                 reference_ifos='H1,L1', num_comps=5, **kwargs):
        """
        Create a statistic class instance

        Parameters
        ----------
        sngl_ranking: str
            The name of the ranking to use for the single-detector triggers.
        files: list of strs, needed here
            A list containing the filenames of hdf format files used to help
            construct the coincident statistics. The files must have a 'stat'
            attribute which is used to associate them with the appropriate
            statistic class.
        ifos: list of strs
            The list of detector names
        reference_ifos: string of comma separated ifo prefixes
            Detectors to be used as the reference network for network
            sensitivity comparisons. Each must be in fits_by_tid
        """
        # FIXME: Currently hardcoded to True. This statistic does not support the
        #        statistic features options at present. Do *NOT* give them!!
        kwargs['normalize_fit_rate'] = True

        # read in background fit info and store it
        ExpFitStatistic.__init__(self, sngl_ranking, files=files,
                                 ifos=ifos, **kwargs)
        # if ifos not already set, determine via background fit info
        self.ifos = self.ifos or self.bg_ifos
        self.num_comps = int(num_comps)

        # PhaseTD statistic single_dtype plus network sensitivity benchmark
        PhaseTDTHAStatistic.__init__(self, sngl_ranking, files=files,
                                     ifos=self.ifos, num_comps=num_comps, **kwargs)
        self.single_dtype.append(('benchmark_logvol', numpy.float32))

        for ifo in self.bg_ifos:
            self.assign_median_sigma(ifo)

        ref_ifos = reference_ifos.split(',')

        # benchmark_logvol is a benchmark sensitivity array over template id
        hl_net_med_sigma = numpy.amin([self.fits_by_tid[ifo]['median_sigma']
                                       for ifo in ref_ifos], axis=0)
        self.benchmark_logvol = 3.0 * numpy.log(hl_net_med_sigma)
        self.single_increasing = False

    def assign_median_sigma(self, ifo):
        """
        Read and sort the median_sigma values from input files.

        Parameters
        ----------
        ifo: str
            The ifo to consider.
        """

        with h5py.File(self.files[f'{ifo}-fit_coeffs'], 'r') as coeff_file:
            template_id = coeff_file['template_id'][:]
            tid_sort = numpy.argsort(template_id)
            self.fits_by_tid[ifo]['median_sigma'] = \
                coeff_file['median_sigma'][:][tid_sort]

    def lognoiserate(self, trigs, alphabelow=6):
        """
        Calculate the log noise rate density over single-ifo ranking

        Read in single trigger information, make the newsnr statistic
        and rescale by the fitted coefficients alpha and rate

        Parameters
        -----------
        trigs: dict of numpy.ndarrays, h5py group (or similar dict-like object)
            Dictionary-like object holding single detector trigger information.
        alphabelow: float, default=6
            Use this slope to fit the noise triggers below the point at which
            fits are present in the input files.

        Returns
        ---------
        lognoisel: numpy.array
            Array of log noise rate density for each input trigger.
        """
        # What is the template number currently being used?
        try:
            # exists if accessed via coinc_findtrigs, this is a number
            self.curr_tnum = trigs.template_num
        except AttributeError:
            # exists for SingleDetTriggers & pycbc_live get_coinc,
            # this is a numpy array
            self.curr_tnum = trigs["template_id"]

        alphai, ratei, thresh = self.find_fits(trigs)
        newsnr = self.get_sngl_ranking(trigs)
        # Above the threshold we use the usual fit coefficient (alpha)
        # below threshold use specified alphabelow
        bt = newsnr < thresh
        lognoisel = - alphai * (newsnr - thresh) + numpy.log(alphai) + \
                        numpy.log(ratei)
        lognoiselbt = - alphabelow * (newsnr - thresh) + \
                           numpy.log(alphabelow) + numpy.log(ratei)
        lognoisel[bt] = lognoiselbt[bt]
        return numpy.array(lognoisel, ndmin=1, dtype=numpy.float32)

    def single(self, trigs):
        """
        Calculate the necessary single detector information

        In this case the ranking rescaled (see the lognoiserate method here)
        with the phase, end time, sigma, SNR, template_id and the
        benchmark_logvol values added in.

        Parameters
        ----------
        trigs: dict of numpy.ndarrays, h5py group (or similar dict-like object)
            Dictionary-like object holding single detector trigger information.
        Returns
        -------
        numpy.ndarray
            The array of single detector values
        """

        # single-ifo stat = log of noise rate
        sngl_stat = self.lognoiserate(trigs)
        # populate other fields to calculate phase/time/amp consistency
        # and sigma comparison
        singles = numpy.zeros(len(sngl_stat), dtype=self.single_dtype)
        singles['snglstat'] = sngl_stat
        singles['coa_phase'] = trigs['coa_phase'][:]
        singles['coa_phase_comp_1'] = trigs['coa_phase_comp_1'][:]
        singles['coa_phase_comp_2'] = trigs['coa_phase_comp_2'][:]
        singles['coa_phase_comp_3'] = trigs['coa_phase_comp_3'][:]
        singles['coa_phase_comp_4'] = trigs['coa_phase_comp_4'][:]
        singles['coa_phase_comp_5'] = trigs['coa_phase_comp_5'][:]
        singles['end_time'] = trigs['end_time'][:]
        singles['sigmasq'] = trigs['sigmasq'][:]
        singles['snr'] = trigs['snr'][:]
        singles['snr_comp_1'] = trigs['snr_comp_1'][:]
        singles['snr_comp_2'] = trigs['snr_comp_2'][:]
        singles['snr_comp_3'] = trigs['snr_comp_3'][:]
        singles['snr_comp_4'] = trigs['snr_comp_4'][:]
        singles['snr_comp_5'] = trigs['snr_comp_5'][:]
        try:
            tnum = trigs.template_num  # exists if accessed via coinc_findtrigs
        except AttributeError:
            tnum = trigs['template_id']  # exists for SingleDetTriggers
            # Should only be one ifo fit file provided
            assert len(self.ifos) == 1
        # Store benchmark log volume as single-ifo information since the coinc
        # method does not have access to template id
        singles['benchmark_logvol'] = self.benchmark_logvol[tnum]
        return numpy.array(singles, ndmin=1)

    def rank_stat_single(self, single_info):
        """
        Calculate the statistic for single detector candidates

        Parameters
        ----------
        single_info: tuple
            Tuple containing two values. The first is the ifo (str) and the
            second is the single detector triggers.

        Returns
        -------
        numpy.ndarray
            The array of single detector statistics
        """
        sngls = single_info[1]

        ln_noise_rate = sngls['snglstat']
        ln_noise_rate -= self.benchmark_lograte
        network_sigmasq = sngls['sigmasq']
        network_logvol = 1.5 * numpy.log(network_sigmasq)
        benchmark_logvol = sngls['benchmark_logvol']
        network_logvol -= benchmark_logvol
        ln_s = -4 * numpy.log(sngls['snr'] / self.ref_snr)
        loglr = network_logvol - ln_noise_rate + ln_s
        # cut off underflowing and very small values
        loglr[loglr < -30.] = -30.
        return loglr

    def rank_stat_coinc(self, s, slide, step, to_shift,
                        **kwargs): # pylint:disable=unused-argument
        """
        Calculate the coincident detection statistic.

        Parameters
        ----------
        sngls_list: list
            List of (ifo, single detector statistic) tuples
        slide: (unused in this statistic)
        step: (unused in this statistic)
        to_shift: list
            List of integers indicating what multiples of the time shift will
            be applied (unused in this statistic)

        Returns
        -------
        numpy.ndarray
            Array of coincident ranking statistic values
        """

        sngl_rates = {sngl[0]: sngl[1]['snglstat'] for sngl in s}
        ln_noise_rate = coinc_rate.combination_noise_lograte(
                                  sngl_rates, kwargs['time_addition'])
        ln_noise_rate -= self.benchmark_lograte

        # Network sensitivity for a given coinc type is approximately
        # determined by the least sensitive ifo
        network_sigmasq = numpy.amin([sngl[1]['sigmasq'] for sngl in s],
                                     axis=0)
        # Volume \propto sigma^3 or sigmasq^1.5
        network_logvol = 1.5 * numpy.log(network_sigmasq)
        # Get benchmark log volume as single-ifo information :
        # benchmark_logvol for a given template is not ifo-dependent, so
        # choose the first ifo for convenience
        benchmark_logvol = s[0][1]['benchmark_logvol']
        network_logvol -= benchmark_logvol

        # Use prior histogram to get Bayes factor for signal vs noise
        # given the time, phase and SNR differences between IFOs

        # First get signal PDF logr_s
        stat = {ifo: st for ifo, st in s}
        logr_s = self.logsignalrate(stat, slide * step, to_shift)

        # Find total volume of phase-time-amplitude space occupied by noise
        # coincs
        # Extent of time-difference space occupied
        noise_twindow = coinc_rate.multiifo_noise_coincident_area(
                            self.hist_ifos, kwargs['time_addition'])
        # Volume is the allowed time difference window, multiplied by 2pi for
        # each phase difference dimension and by allowed range of SNR ratio
        # for each SNR ratio dimension : there are (n_ifos - 1) dimensions
        # for both phase and SNR
        n_ifos = len(self.hist_ifos)
        hist_vol = noise_twindow * \
            (2 * numpy.pi * (self.srbmax - self.srbmin) * self.swidth) ** \
            (n_ifos - 1)
        # Noise PDF is 1/volume, assuming a uniform distribution of noise
        # coincs
        logr_n = - numpy.log(hist_vol)

        # Combine to get final statistic: log of
        # ((rate of signals / rate of noise) * PTA Bayes factor)
        loglr = network_logvol - ln_noise_rate + logr_s - logr_n

        # cut off underflowing and very small values
        loglr[loglr < -30.] = -30.
        return loglr

    def coinc_lim_for_thresh(self, s, thresh, limifo,
                             **kwargs): # pylint:disable=unused-argument
        """
        Optimization function to identify coincs too quiet to be of interest

        Calculate the required single detector statistic to exceed
        the threshold for each of the input triggers.

        Parameters
        ----------
        s: list
            List of (ifo, single detector statistic) tuples for all detectors
            except limifo.
        thresh: float
            The threshold on the coincident statistic.
        limifo: string
            The ifo for which the limit is to be found.

        Returns
        -------
        numpy.ndarray
            Array of limits on the limifo single statistic to
            exceed thresh.
        """

        # Safety against subclassing and not rethinking this
        allowed_names = ['ExpFitFgBgNormTHAStatistic',
                         'ExpFitFgBgNormTHABBHStatistic',
                         'DQExpFitFgBgNormTHAStatistic']
        self._check_coinc_lim_subclass(allowed_names)

        if not self.has_hist:
            self.get_hist()
        # if the threshold is below this value all triggers will
        # pass because of rounding in the coinc method
        if thresh <= -30:
            return numpy.ones(len(s[0][1]['snglstat'])) * numpy.inf
        sngl_rates = {sngl[0]: sngl[1]['snglstat'] for sngl in s}
        # Add limifo to singles dict so that overlap time is calculated correctly
        sngl_rates[limifo] = numpy.zeros(len(s[0][1]))
        ln_noise_rate = coinc_rate.combination_noise_lograte(
                                  sngl_rates, kwargs['time_addition'])
        ln_noise_rate -= self.benchmark_lograte

        # Assume best case and use the maximum sigma squared from all triggers
        network_sigmasq = numpy.ones(len(s[0][1])) * kwargs['max_sigmasq']
        # Volume \propto sigma^3 or sigmasq^1.5
        network_logvol = 1.5 * numpy.log(network_sigmasq)
        # Get benchmark log volume as single-ifo information :
        # benchmark_logvol for a given template is not ifo-dependent, so
        # choose the first ifo for convenience
        benchmark_logvol = s[0][1]['benchmark_logvol']
        network_logvol -= benchmark_logvol

        # Assume best case scenario and use maximum signal rate
        logr_s = numpy.log(self.hist_max
                           * (kwargs['min_snr'] / self.ref_snr) ** -4.0)

        # Find total volume of phase-time-amplitude space occupied by noise
        # coincs
        # Extent of time-difference space occupied
        noise_twindow = coinc_rate.multiifo_noise_coincident_area(
                            self.hist_ifos, kwargs['time_addition'])
        # Volume is the allowed time difference window, multiplied by 2pi for
        # each phase difference dimension and by allowed range of SNR ratio
        # for each SNR ratio dimension : there are (n_ifos - 1) dimensions
        # for both phase and SNR
        n_ifos = len(self.hist_ifos)
        hist_vol = noise_twindow * \
            (2 * numpy.pi * (self.srbmax - self.srbmin) * self.swidth) ** \
            (n_ifos - 1)
        # Noise PDF is 1/volume, assuming a uniform distribution of noise
        # coincs
        logr_n = - numpy.log(hist_vol)

        loglr = - thresh + network_logvol - ln_noise_rate + logr_s - logr_n
        return loglr

# FIXME: This class (and probably the one above) need properly integrating into
#        the new statistic system. Much duplication present here.
class ExpFitFgBgNormMARGStatistic(PhaseTDMARGStatistic,
                                 ExpFitStatistic):
    """
    Statistic combining PhaseTD, ExpFitBg and additional foreground info.
    """

    def __init__(self, sngl_ranking, files=None, ifos=None,
                 reference_ifos='H1,L1', num_comps=5, **kwargs):
        """
        Create a statistic class instance

        Parameters
        ----------
        sngl_ranking: str
            The name of the ranking to use for the single-detector triggers.
        files: list of strs, needed here
            A list containing the filenames of hdf format files used to help
            construct the coincident statistics. The files must have a 'stat'
            attribute which is used to associate them with the appropriate
            statistic class.
        ifos: list of strs
            The list of detector names
        reference_ifos: string of comma separated ifo prefixes
            Detectors to be used as the reference network for network
            sensitivity comparisons. Each must be in fits_by_tid
        """
        # FIXME: Currently hardcoded to True. This statistic does not support the
        #        statistic features options at present. Do *NOT* give them!!
        kwargs['normalize_fit_rate'] = True
        # uniform_noise:1 replaces the (dpsi, dtheta) noise flow by a uniform
        # density over its bounds, dpsi in [-pi/4, pi/4), dtheta in
        # [-pi, pi): 1 / pi^2 per ifo pair (a test of how much the noise
        # model matters)
        self.uniform_noise = str(kwargs.pop('uniform_noise', '0')).lower() \
            in ('1', 'true', 'yes', '1.0')

        # read in background fit info and store it
        ExpFitStatistic.__init__(self, sngl_ranking, files=files,
                                 ifos=ifos, **kwargs)
        # if ifos not already set, determine via background fit info
        self.ifos = self.ifos or self.bg_ifos
        self.num_comps = int(num_comps)

        # PhaseTD statistic single_dtype plus network sensitivity benchmark
        PhaseTDMARGStatistic.__init__(self, sngl_ranking, files=files,
                                      ifos=self.ifos, num_comps=num_comps,
                                      **kwargs)
        self.single_dtype.append(('benchmark_logvol', numpy.float32))

        for ifo in self.bg_ifos:
            self.assign_median_sigma(ifo)

        ref_ifos = reference_ifos.split(',')

        # benchmark_logvol is a benchmark sensitivity array over template id
        hl_net_med_sigma = numpy.amin([self.fits_by_tid[ifo]['median_sigma']
                                       for ifo in ref_ifos], axis=0)
        self.benchmark_logvol = 3.0 * numpy.log(hl_net_med_sigma)
        self.single_increasing = False

    def assign_median_sigma(self, ifo):
        """
        Read and sort the median_sigma values from input files.

        Parameters
        ----------
        ifo: str
            The ifo to consider.
        """

        with h5py.File(self.files[f'{ifo}-fit_coeffs'], 'r') as coeff_file:
            template_id = coeff_file['template_id'][:]
            tid_sort = numpy.argsort(template_id)
            self.fits_by_tid[ifo]['median_sigma'] = \
                coeff_file['median_sigma'][:][tid_sort]

    def lognoiserate(self, trigs, alphabelow=6):
        """
        Calculate the log noise rate density over single-ifo ranking

        Read in single trigger information, make the newsnr statistic
        and rescale by the fitted coefficients alpha and rate

        Parameters
        -----------
        trigs: dict of numpy.ndarrays, h5py group (or similar dict-like object)
            Dictionary-like object holding single detector trigger information.
        alphabelow: float, default=6
            Use this slope to fit the noise triggers below the point at which
            fits are present in the input files.

        Returns
        ---------
        lognoisel: numpy.array
            Array of log noise rate density for each input trigger.
        """
        # What is the template number currently being used?
        try:
            # exists if accessed via coinc_findtrigs, this is a number
            self.curr_tnum = trigs.template_num
        except AttributeError:
            # exists for SingleDetTriggers & pycbc_live get_coinc,
            # this is a numpy array
            self.curr_tnum = trigs["template_id"]

        alphai, ratei, thresh = self.find_fits(trigs)
        newsnr = self.get_sngl_ranking(trigs)
        # Above the threshold we use the usual fit coefficient (alpha)
        # below threshold use specified alphabelow
        bt = newsnr < thresh
        lognoisel = - alphai * (newsnr - thresh) + numpy.log(alphai) + \
                        numpy.log(ratei)
        lognoiselbt = - alphabelow * (newsnr - thresh) + \
                           numpy.log(alphabelow) + numpy.log(ratei)
        lognoisel[bt] = lognoiselbt[bt]
        return numpy.array(lognoisel, ndmin=1, dtype=numpy.float32)

    def single(self, trigs):
        """
        Calculate the necessary single detector information

        In this case the ranking rescaled (see the lognoiserate method here)
        with the phase, end time, sigma, SNR, template_id and the
        benchmark_logvol values added in.

        Parameters
        ----------
        trigs: dict of numpy.ndarrays, h5py group (or similar dict-like object)
            Dictionary-like object holding single detector trigger information.
        Returns
        -------
        numpy.ndarray
            The array of single detector values
        """

        # single-ifo stat = log of noise rate
        sngl_stat = self.lognoiserate(trigs)
        # populate other fields to calculate phase/time/amp consistency
        # and sigma comparison
        singles = numpy.zeros(len(sngl_stat), dtype=self.single_dtype)
        singles['snglstat'] = sngl_stat
        singles['coa_phase'] = trigs['coa_phase'][:]
        singles['coa_phase_comp_1'] = trigs['coa_phase_comp_1'][:]
        singles['coa_phase_comp_2'] = trigs['coa_phase_comp_2'][:]
        singles['coa_phase_comp_3'] = trigs['coa_phase_comp_3'][:]
        singles['coa_phase_comp_4'] = trigs['coa_phase_comp_4'][:]
        singles['coa_phase_comp_5'] = trigs['coa_phase_comp_5'][:]
        singles['end_time'] = trigs['end_time'][:]
        singles['sigmasq'] = trigs['sigmasq'][:]
        singles['snr'] = trigs['snr'][:]
        singles['snr_comp_1'] = trigs['snr_comp_1'][:]
        singles['snr_comp_2'] = trigs['snr_comp_2'][:]
        singles['snr_comp_3'] = trigs['snr_comp_3'][:]
        singles['snr_comp_4'] = trigs['snr_comp_4'][:]
        singles['snr_comp_5'] = trigs['snr_comp_5'][:]
        singles['marg_lnl'] = trigs['marg_lnl'][:]
        singles['marg_snr'] = trigs['marg_snr'][:]
        singles['marg_phase'] = trigs['marg_phase'][:]
        singles['marg_theta'] = trigs['marg_theta'][:]
        singles['marg_alpha'] = trigs['marg_alpha'][:]
        singles['marg_psi'] = trigs['marg_psi'][:]
        singles['beta'] = trigs['beta'][:]
        try:
            tnum = trigs.template_num  # exists if accessed via coinc_findtrigs
        except AttributeError:
            tnum = trigs['template_id']  # exists for SingleDetTriggers
            # Should only be one ifo fit file provided
            assert len(self.ifos) == 1
        # Store benchmark log volume as single-ifo information since the coinc
        # method does not have access to template id
        singles['benchmark_logvol'] = self.benchmark_logvol[tnum]
        return numpy.array(singles, ndmin=1)

    def rank_stat_single(self, single_info):
        """
        Calculate the statistic for single detector candidates

        Parameters
        ----------
        single_info: tuple
            Tuple containing two values. The first is the ifo (str) and the
            second is the single detector triggers.

        Returns
        -------
        numpy.ndarray
            The array of single detector statistics
        """
        sngls = single_info[1]

        ln_noise_rate = sngls['snglstat']
        ln_noise_rate -= self.benchmark_lograte
        network_sigmasq = sngls['sigmasq']
        network_logvol = 1.5 * numpy.log(network_sigmasq)
        benchmark_logvol = sngls['benchmark_logvol']
        network_logvol -= benchmark_logvol
        ln_s = -4 * numpy.log(sngls['snr'] / self.ref_snr)
        loglr = network_logvol - ln_noise_rate + ln_s
        # cut off underflowing and very small values
        loglr[loglr < -30.] = -30.
        return loglr

    def rank_stat_coinc(self, s, slide, step, to_shift,
                        **kwargs): # pylint:disable=unused-argument
        """
        Calculate the coincident detection statistic.

        Parameters
        ----------
        sngls_list: list
            List of (ifo, single detector statistic) tuples
        slide: (unused in this statistic)
        step: (unused in this statistic)
        to_shift: list
            List of integers indicating what multiples of the time shift will
            be applied (unused in this statistic)

        Returns
        -------
        numpy.ndarray
            Array of coincident ranking statistic values
        """

        sngl_rates = {sngl[0]: sngl[1]['snglstat'] for sngl in s}
        ln_noise_rate = coinc_rate.combination_noise_lograte(
                                  sngl_rates, kwargs['time_addition'])
        ln_noise_rate -= self.benchmark_lograte

        # Network sensitivity for a given coinc type is approximately
        # determined by the least sensitive ifo
        network_sigmasq = numpy.amin([sngl[1]['sigmasq'] for sngl in s],
                                     axis=0)
        # Volume \propto sigma^3 or sigmasq^1.5
        network_logvol = 1.5 * numpy.log(network_sigmasq)
        # Get benchmark log volume as single-ifo information :
        # benchmark_logvol for a given template is not ifo-dependent, so
        # choose the first ifo for convenience
        benchmark_logvol = s[0][1]['benchmark_logvol']
        network_logvol -= benchmark_logvol

        # Use prior histogram to get Bayes factor for signal vs noise
        # given the time, phase and SNR differences between IFOs

        # First get signal PDF logr_s
        stat = {ifo: st for ifo, st in s}
        logr_s = self.logsignalrate(stat, slide * step, to_shift)

        # Find total volume of phase-time-amplitude space occupied by noise
        # coincs
        # Extent of time-difference space occupied
        noise_twindow = coinc_rate.multiifo_noise_coincident_area(
                            self.hist_ifos, kwargs['time_addition'])
        # Volume is the allowed time difference window, multiplied by 2pi for
        # each phase difference dimension and by allowed range of SNR ratio
        # for each SNR ratio dimension : there are (n_ifos - 1) dimensions
        # for both phase and SNR
        n_ifos = len(self.hist_ifos)
        # (dpsi, dtheta) are not in the uniform volume: their noise density
        # is a flow, divided out in logsignalrate
        hist_vol = noise_twindow * \
            (2 * numpy.pi            # phase difference
             * 2 * numpy.pi          # alpha0 difference
             ) ** (n_ifos - 1)
        # Aligned-spin templates have no orientation features: only the time
        # and phase differences are spanned
        hist_vol_aligned = noise_twindow * (2 * numpy.pi) ** (n_ifos - 1)
        # Noise PDF is 1/volume, assuming a uniform distribution of noise
        # coincs
        aligned = self.is_aligned(s[0][1]['beta'])
        logr_n = numpy.where(aligned, - numpy.log(hist_vol_aligned),
                             - numpy.log(hist_vol))

        # Combine to get final statistic: log of
        # ((rate of signals / rate of noise) * PTA Bayes factor)
        loglr = network_logvol - ln_noise_rate + logr_s - logr_n

        # cut off underflowing and very small values
        loglr[loglr < -30.] = -30.
        return loglr

    def coinc_lim_for_thresh(self, s, thresh, limifo,
                             **kwargs): # pylint:disable=unused-argument
        """
        Optimization function to identify coincs too quiet to be of interest

        Calculate the required single detector statistic to exceed
        the threshold for each of the input triggers.

        Parameters
        ----------
        s: list
            List of (ifo, single detector statistic) tuples for all detectors
            except limifo.
        thresh: float
            The threshold on the coincident statistic.
        limifo: string
            The ifo for which the limit is to be found.

        Returns
        -------
        numpy.ndarray
            Array of limits on the limifo single statistic to
            exceed thresh.
        """

        # Safety against subclassing and not rethinking this
        allowed_names = ['ExpFitFgBgNormMARGStatistic']
        self._check_coinc_lim_subclass(allowed_names)

        if not self.has_hist:
            self.get_hist()
        # if the threshold is below this value all triggers will
        # pass because of rounding in the coinc method
        if thresh <= -30:
            return numpy.ones(len(s[0][1]['snglstat'])) * numpy.inf
        sngl_rates = {sngl[0]: sngl[1]['snglstat'] for sngl in s}
        # Add limifo to singles dict so that overlap time is calculated correctly
        sngl_rates[limifo] = numpy.zeros(len(s[0][1]))
        ln_noise_rate = coinc_rate.combination_noise_lograte(
                                  sngl_rates, kwargs['time_addition'])
        ln_noise_rate -= self.benchmark_lograte

        # Assume best case and use the maximum sigma squared from all triggers
        network_sigmasq = numpy.ones(len(s[0][1])) * kwargs['max_sigmasq']
        # Volume \propto sigma^3 or sigmasq^1.5
        network_logvol = 1.5 * numpy.log(network_sigmasq)
        # Get benchmark log volume as single-ifo information :
        # benchmark_logvol for a given template is not ifo-dependent, so
        # choose the first ifo for convenience
        benchmark_logvol = s[0][1]['benchmark_logvol']
        network_logvol -= benchmark_logvol

        # The template (so beta) is the same for every trigger here
        aligned = self.is_aligned(s[0][1]['beta'])

        # Assume best case scenario and use maximum signal rate: the flow's
        # maximum log density (hist_max is already in log units) for the
        # model this template uses, times the SNR^-4 population factor
        logr_s = (numpy.where(aligned, self.hist_max_aligned, self.hist_max)
                  - 4. * numpy.log(kwargs['min_snr'] / self.ref_snr))
        if getattr(self, 'uniform_noise', False):
            # the uniform (dpsi, dtheta) noise density adds 2 ln(pi) per pair
            logr_s = logr_s + numpy.where(
                aligned, 0., 2. * numpy.log(numpy.pi)
                * (len(self.hist_ifos) - 1))

        # Find total volume of phase-time-amplitude space occupied by noise
        # coincs, as in rank_stat_coinc
        noise_twindow = coinc_rate.multiifo_noise_coincident_area(
                            self.hist_ifos, kwargs['time_addition'])
        n_ifos = len(self.hist_ifos)
        # (dpsi, dtheta) are not in the uniform volume: their noise density
        # is a flow, divided out in logsignalrate
        hist_vol = noise_twindow * \
            (2 * numpy.pi            # phase difference
             * 2 * numpy.pi          # alpha0 difference
             ) ** (n_ifos - 1)
        hist_vol_aligned = noise_twindow * (2 * numpy.pi) ** (n_ifos - 1)
        # Noise PDF is 1/volume, assuming a uniform distribution of noise
        # coincs
        logr_n = numpy.where(aligned, - numpy.log(hist_vol_aligned),
                             - numpy.log(hist_vol))

        loglr = - thresh + network_logvol - ln_noise_rate + logr_s - logr_n
        return loglr

    
    
class _LazyPSD(object):
    """A PSD segment read only if its values are needed (the responses
    usually need just its cached band means)"""
    def __init__(self, owner, ifo, i):
        self.owner, self.key = owner, (ifo, i)
        self._a = None

    def array(self):
        if self._a is None:
            self._a = self.owner._psd(*self.key)
        return self._a

    def __getitem__(self, k):
        return self.array()[k]


class ExactCoincTerm(object):
    """
    Exact coincident term for a precessing H1-L1 coinc, from the triggers'
    stored harmonic SNRs (see ExpFitFgBgNormMARGExactStatistic).

        exact = ln < e^{lnL_H} e^{lnL_L} p(SNR ratio, dphase | source, S)
                     p(dt | sky, S) >_{orientation, sky ring, polarisation}
                - lambda_H - lambda_L + ln(twin 2 pi)

    lnL_d is the single-detector log likelihood maximised over the complex
    amplitude, for the waveform the source puts in detector d (grid A, B,
    the template's raw harmonics and the detector's PSD, rebuilt as in
    pycbc_inspiral_tha). The amplitude/phase density is a Gaussian in the
    difference between the two detectors' best-fit complex amplitudes
    (normalised in SNR ratio and phase difference, with a fractional
    systematic floor); the time density a Gaussian in dt - tau(sky) (with a
    floor for the trigger time's sample rounding). Only the ring of sky
    positions whose H1-L1 delay matches dt is used; the single-detector
    marginals lambda_d are over the same points (for one detector the sky
    position only enters through the effective polarisation, which is
    uniform for every sky point, so this equals the all-sky value).
    """

    def __init__(self, ifos, grid_file, bank_file, psd_files, n_fine=262144,
                 max_ring=1000, n_pol=8, prune=10., amp_phase_floor=0.,
                 timing_floor=None, timing_uncertainty=0.001,
                 snr_reference=5., psi_gp=16, cache_size=64,
                 quiet_snr=12., quiet_ring=32, quiet_pol=4, quiet_ang=16,
                 check_frac=0., batch_elems=4e6, seed=0,
                 quiet_orient_step=4, method='grid', q_tau=6, q_az=8,
                 q_pol=4, l_tau=12, l_az=24, l_pol=6, r_tau=24, r_az=48,
                 r_pol=12, check_ref='reference', psd_mode='segment',
                 bound_margin=2., refine_above=4., gram_bins=8192,
                 phase_time=1, phase_time_sign=-1., phase_time_freq=0.):
        self.ifos = list(ifos)
        if len(self.ifos) != 2:
            raise ValueError('The exact coincident term is for two ifos')
        self.grid_file = grid_file
        self.bank_file = bank_file
        self.psd_files = psd_files
        self.n_fine = int(n_fine)
        self.max_ring = int(max_ring)
        self.n_pol = int(n_pol)
        self.prune = float(prune)
        self.eps2 = float(amp_phase_floor) ** 2
        self.timing_floor = timing_floor
        self.timing_uncertainty = float(timing_uncertainty)
        self.snr_reference = float(snr_reference)
        self.psi_gp = int(psi_gp)
        self.cache_size = int(cache_size)
        # quiet coincs (network SNR below quiet_snr, almost all background):
        # their likelihood varies slowly over sky and polarisation, so they
        # are done in batches with quiet_ring sky points x quiet_pol
        # polarisations; a fraction check_frac of them is also done at full
        # resolution, and the difference is kept in self.check_diffs
        self.quiet_snr = float(quiet_snr)
        self.quiet_ring = int(quiet_ring)
        self.quiet_pol = int(quiet_pol)
        self.quiet_ang = int(quiet_ang)
        # quiet coincs: the joint sum uses every quiet_orient_step-th
        # orientation (staggered in alpha from one theta row to the next),
        # each standing for quiet_orient_step of them
        self.quiet_orient_step = int(quiet_orient_step)
        self.check_frac = float(check_frac)
        self.batch_elems = float(batch_elems)
        self.check_rng = numpy.random.default_rng(int(seed))
        self.check_diffs = []
        self.check_full = []
        self.check_kind = []
        # 'grid': sky integral by quadrature over the H1-L1 baseline
        # coordinates (Gauss-Legendre in the delay, evenly spaced in the
        # azimuth around the baseline), with (q_tau, q_az, q_pol) nodes for
        # quiet coincs and (l_tau, l_az, l_pol) for loud ones;
        # 'reference': every coinc at (r_tau, r_az, r_pol) with every
        # orientation (slow; the yardstick for the others);
        # 'ring': the earlier random-ring version (_batch / _one)
        self.method = str(method)
        self.q_nodes = (int(q_tau), int(q_az), int(q_pol))
        self.l_nodes = (int(l_tau), int(l_az), int(l_pol))
        self.r_nodes = (int(r_tau), int(r_az), int(r_pol))
        # what the spot checks compare with: 'reference' or 'ring'
        self.check_ref = str(check_ref)
        # 'segment': responses for each trigger's own PSD segment;
        # 'reference': one PSD per ifo (the median over the job's segments)
        # for every trigger, so one response set per template and ifo
        self.psd_mode = str(psd_mode)
        self.bound_margin = float(bound_margin)
        # quiet coincs whose statistic comes out above refine_above are
        # recomputed with the loud settings (they are the rare ones that
        # matter: every decimation threshold is above it by default)
        self.refine_above = float(refine_above)
        self.gram_bins = int(gram_bins)
        # Time-phase coupling: noise moves each ifo's SNR peak by dt_d and
        # rotates its phase by 2 pi fbar dt_d. At a sky position with delay
        # tau the shifts' difference is dt - tau, so L1's phase is rotated
        # back by -sign 2 pi fbar (dt - tau) before the ifos are compared.
        # fbar: the template's mean frequency, weighted by |h|^2 / PSD (or
        # phase_time_freq if > 0). sign: measured on injections (-1).
        self.phase_time = bool(float(phase_time))
        self.phase_time_sign = float(phase_time_sign)
        self.phase_time_freq = float(phase_time_freq)
        self.n_refined = 0
        self.t_raws = 0.
        self.t_resp = 0.
        self._gl = {}
        self.n_quiet = 0
        self.n_loud = 0
        self.loaded = False
        self.t_setup = 0.
        self.t_coinc = 0.

    def load(self):
        import pycbc.detector
        from pycbc.waveform.bank import TemplateBank
        self.gf = h5py.File(self.grid_file, 'r')
        ga = self.gf.attrs
        self.df = float(ga['df'])
        self.kmin, self.kmax = int(ga['kmin']), int(ga['kmax'])
        self.sample_rate = float(ga['sample_rate'])
        self.f_lower = float(ga['low_frequency_cutoff'])
        self.f_final = self.sample_rate / 2.
        self.nfreq = int(self.f_final / self.df) + 1
        self.theta_grid = numpy.asarray(ga['theta_grid'])
        self.alpha_grid = numpy.asarray(ga['alpha_grid'])
        w = numpy.asarray(ga['weights_ta'], dtype=float)
        self.weights_ta = w
        self.ln_wta = numpy.log(w / w.sum())
        self.grid_row = {int(h): i for i, h in
                         enumerate(self.gf['template_hash'][:])}
        self.bank = TemplateBank(self.bank_file)
        if self.timing_floor is None:
            self.timing_floor = 1. / (self.sample_rate * numpy.sqrt(12.))
        self.timing_floor = float(self.timing_floor)

        self.psd_h = {}
        self.psd_times = {}
        for ifo in self.ifos:
            f = h5py.File(self.psd_files[ifo], 'r')
            self.psd_h[ifo] = f
            self.psd_times[ifo] = (f[ifo]['start_time'][:],
                                   f[ifo]['end_time'][:])

        # Sky positions in Earth-fixed coordinates (longitude, latitude) and
        # their H1-L1 delays, which do not depend on time in this frame
        self.dets = {ifo: pycbc.detector.Detector(ifo) for ifo in self.ifos}
        k = numpy.arange(self.n_fine)
        self.lon = (k * numpy.pi * (3. - numpy.sqrt(5.))) % (2. * numpy.pi)
        self.lat = numpy.arcsin(1. - (2. * k + 1.) / self.n_fine)
        t0 = 1e9
        ra0 = self.lon + self._gmst(t0)
        h, l = self.ifos
        self.tau = numpy.asarray(
            self.dets[h].time_delay_from_earth_center(ra0, self.lat, t0)
            - self.dets[l].time_delay_from_earth_center(ra0, self.lat, t0))
        # psi and psi + pi/2 differ by a sign in both detectors, absorbed by
        # the common phase: [0, pi/2) covers every case
        self.pols = (numpy.arange(self.n_pol) + 0.5) * (numpy.pi / 2) \
            / self.n_pol
        # antenna patterns are fixed in the Earth frame too: F(psi) =
        # F(0) cos 2psi + F(pi/4) sin 2psi for both F+ and Fx
        self.F = {}
        for ifo in self.ifos:
            p0, c0 = self.dets[ifo].antenna_pattern(ra0, self.lat, 0., t0)
            p4, c4 = self.dets[ifo].antenna_pattern(ra0, self.lat,
                                                    numpy.pi / 4, t0)
            self.F[ifo] = tuple(numpy.asarray(v, dtype=numpy.float32)
                                for v in (p0, p4, c0, c4))
        self.tau_order = numpy.argsort(self.tau)
        self.tau_sorted = self.tau[self.tau_order]
        if self.psd_mode == 'reference':
            self._make_reference_psds()
        if self.method in ('grid', 'reference'):
            n_e = numpy.stack([numpy.cos(self.lat) * numpy.cos(self.lon),
                               numpy.cos(self.lat) * numpy.sin(self.lon),
                               numpy.sin(self.lat)], axis=1)
            self._setup_geometry(n_e, {ifo: numpy.asarray(
                self.dets[ifo].response, dtype=numpy.float64)
                for ifo in self.ifos})
        # antenna-vector angle for the single-ifo marginals (F and -F are
        # the same up to the fitted phase, so [0, pi) covers it)
        self.ang = (numpy.arange(64) + 0.5) * numpy.pi / 64
        self.tmpl_cache = {}
        self.resp_cache = {}
        self.loaded = True
        logger.info('Exact coincident term: grid %s (%d orientations), '
                    'bank %s, PSDs %s', self.grid_file, len(self.theta_grid),
                    self.bank_file, self.psd_files)

    @staticmethod
    def _gmst(t):
        import lal
        return float(lal.GreenwichMeanSiderealTime(lal.LIGOTimeGPS(float(t))))

    def _psd_index(self, ifo, t):
        st, et = self.psd_times[ifo]
        inside = numpy.nonzero((st <= t) & (t < et))[0]
        if len(inside) == 0:
            return None
        if self.psd_mode == 'reference':
            return 0
        return int(inside[numpy.argmin(numpy.abs(
            0.5 * (st[inside] + et[inside]) - t))])

    def _make_reference_psds(self):
        """One PSD per ifo: 1 / (mean over the job's PSD segments of
        1/PSD), built while reading so that only one segment is held at a
        time"""
        self.ref_psd = {}
        for ifo in self.ifos:
            n = len(self.psd_times[ifo][0])
            acc, cnt = None, None
            for i in range(n):
                p = self._psd_read(ifo, i)
                w = numpy.where(p > 0, 1. / numpy.where(p > 0, p, 1.), 0.)
                acc = w if acc is None else acc + w
                cnt = (p > 0).astype(float) if cnt is None \
                    else cnt + (p > 0)
            mw = numpy.where(cnt > 0, acc / numpy.maximum(cnt, 1), 0.)
            self.ref_psd[ifo] = numpy.where(mw > 0, 1. / numpy.where(
                mw > 0, mw, 1.), 0.)
        logger.info('Exact coincident term: one reference PSD per ifo '
                    '(average of 1/PSD over %s segments)',
                    {i: len(self.psd_times[i][0]) for i in self.ifos})

    def _psd(self, ifo, i):
        if self.psd_mode == 'reference' or i == 'ref':
            if not hasattr(self, 'ref_psd'):
                self._make_reference_psds()
            return self.ref_psd[ifo]
        return self._psd_segment(ifo, i)

    def _psd_segment(self, ifo, i):
        # full PSDs are not kept (hundreds of segments x ~10^6 bins); the
        # responses keep each segment's band means instead (see _gram)
        return self._psd_read(ifo, i)

    def _psd_read(self, ifo, i):
        import pycbc.psd
        from pycbc.types import FrequencySeries
        d = self.psd_h[ifo][ifo]['psds'][str(i)]
        fs = FrequencySeries(d[:].astype(numpy.float64),
                             delta_f=float(d.attrs['delta_f']))
        fs = pycbc.psd.interpolate(fs, self.df)
        arr = numpy.zeros(self.nfreq)
        n = min(self.nfreq, len(fs))
        arr[:n] = fs.numpy()[:n]
        return arr

    def _template(self, tid):
        if tid not in self.tmpl_cache:
            import time as _time
            from pycbc.waveform.bank import PhenomXPTemplate
            from pycbc.waveform.tha_marg_grid import get_raw_harmonics
            t0 = _time.time()
            if len(self.tmpl_cache) >= self.cache_size:
                self.tmpl_cache.clear()
            row = self.grid_row.get(int(self.bank.table.template_hash[tid]))
            if row is None:
                self.tmpl_cache[tid] = None
            else:
                tmpl = PhenomXPTemplate(self.bank.table[tid], self.sample_rate,
                                        self.f_lower)
                b = float(tmpl.beta)
                aligned = bool((b < 1e-6) or (b > numpy.pi - 1e-6))
                A, B = self.gf['A'][row], self.gf['B'][row]
                w = self.ln_wta
                if aligned:
                    # alpha0 does nothing for aligned spins: keep one alpha0
                    # per theta, carrying the weight of all of them
                    th = numpy.round(self.theta_grid, 12)
                    _, first = numpy.unique(th, return_index=True)
                    first = numpy.sort(first)
                    w = numpy.array([numpy.logaddexp.reduce(
                        self.ln_wta[th == th[i]]) for i in first])
                    A, B = A[first], B[first]
                self.tmpl_cache[tid] = dict(
                    aligned=aligned, A=A, B=B, ln_w=w,
                    raws=get_raw_harmonics(tmpl, self.df, self.f_final,
                                           self.kmin, self.kmax, interp=True))
            self.t_setup += _time.time() - t0
            self.t_raws += _time.time() - t0
        return self.tmpl_cache[tid]

    def _responses(self, tid, ifo, ipsd):
        if self.psd_mode == 'reference':
            ipsd = 'ref'
        key = (tid, ifo, ipsd)
        if key not in self.resp_cache:
            import time as _time
            from pycbc.filter.tha_marginalize import derive_ep_ec_hh
            T = self._template(tid)
            t0 = _time.time()
            # keep responses across templates (coincs of several templates
            # may alternate), dropping the oldest once there are many
            if len(self.resp_cache) >= 4000:
                for k_ in list(self.resp_cache)[:1000]:
                    del self.resp_cache[k_]
            psd = _LazyPSD(self, ifo, ipsd)
            if T['aligned']:
                Ep, Ec, hpp, hcc, hpc = self._aligned_ep_ec(T, psd)
            else:
                Ep, Ec, hpp, hcc, hpc = self._precessing_ep_ec(T, psd)
            # Scale the responses to order unity: the likelihood, the
            # amplitude/phase density and the time density are all unchanged
            # by a COMMON scale (X -> sX, sigma^2 -> s^2 sigma^2), and the
            # joint sum is done in single precision, where the raw values
            # (strain units) would underflow. The scale must be the same for
            # every ifo and PSD segment of the template: a separate scale per
            # ifo would divide out the ifos' relative sensitivity and shift
            # every predicted amplitude ratio.
            hpp, hcc, hpc = numpy.real(hpp), numpy.real(hcc), numpy.real(hpc)
            if not hasattr(self, '_tscale'):
                self._tscale = {}
            if tid not in self._tscale:
                self._tscale[tid] = 1. / numpy.sqrt(numpy.median(hpp + hcc))
            scale = self._tscale[tid]
            fbar = numpy.nan
            wkey = (ifo, 'ref' if self.psd_mode == 'reference' else ipsd)
            if 'Pf' in T and wkey in getattr(self, '_wb', {}):
                wb = self._wb[wkey]
                fbar = float(numpy.dot(T['Pf'], wb)
                             / numpy.dot(numpy.real(T['Pb'][0, 0]), wb))
            self.resp_cache[key] = dict(
                Ep=numpy.conj(numpy.asarray(Ep)) * scale,
                Ec=numpy.conj(numpy.asarray(Ec)) * scale,
                hpp=hpp * scale ** 2, hcc=hcc * scale ** 2,
                hpc=hpc * scale ** 2, fbar=fbar)
            self.t_setup += _time.time() - t0
            self.t_resp += _time.time() - t0
        return self.resp_cache[key]

    def _gram(self, T, psd, nh=None):
        """
        Inner products of the template's harmonics, <raw_k|raw_l> =
        4 df sum conj(raw_k) raw_l / psd over [kmin, kmax).

        The products conj(raw_k) raw_l change slowly with frequency (the
        chirp phase cancels between harmonics) and so does 1/psd away from
        narrow lines, so each template's products are summed once into
        gram_bins frequency bands, and each PSD needs only its mean 1/psd
        per band: a few thousand operations per PSD segment instead of
        millions. Checked against the exact sum (and raw_gram_matrix) on
        the first template; if the banded sum is off by more than 1e-4 the
        exact sum is used throughout.
        """
        from pycbc.filter.tha_marginalize import raw_gram_matrix
        np = numpy
        k0, k1 = self.kmin, self.kmax
        nb = int(getattr(self, 'gram_bins', 8192))
        if not hasattr(self, '_wb'):
            self._wb = {}
        key = getattr(psd, 'key', None)

        def band_means(edges):
            if key is not None and key in self._wb:
                return self._wb[key]
            p_ = np.asarray(psd[k0:k1], dtype=np.float64)
            w_ = np.where(p_ > 0, 1. / np.where(p_ > 0, p_, 1.), 0.)
            wb_ = np.add.reduceat(w_, edges[:-1]) / np.diff(edges)
            if key is not None:
                self._wb[key] = wb_
            return wb_

        if 'Pb' in T and getattr(self, 'banded_gram', None):
            # band sums already made; the harmonics themselves are freed
            G = 4. * self.df * (T['Pb'] @ band_means(T['edges']))
            return G.T if self.gram_transpose else G
        p = np.asarray(psd[k0:k1], dtype=np.float64)
        w = np.where(p > 0, 1. / np.where(p > 0, p, 1.), 0.)
        if 'R' not in T:
            raws = T['raws'] if nh is None else T['raws'][:nh]
            T['R'] = np.array([np.asarray(r[k0:k1]) for r in raws],
                              dtype=np.complex128)
        R = T['R']
        nk = R.shape[1]
        use_bins = getattr(self, 'banded_gram', None) is not False and \
            nk > 4 * nb
        if use_bins:
            edges = np.linspace(0, nk, nb + 1).astype(int)
            T['edges'] = edges
            if 'Pb' not in T:
                nh = len(R)
                Pb = np.zeros((nh, nh, nb), dtype=np.complex128)
                Rc = np.conj(R)
                for a_ in range(nh):
                    for b_ in range(a_, nh):
                        Pb[a_, b_] = np.add.reduceat(Rc[a_] * R[b_],
                                                     edges[:-1])
                        if b_ != a_:
                            Pb[b_, a_] = np.conj(Pb[a_, b_])
                T['Pb'] = Pb
                fr = (k0 + np.arange(nk)) * self.df
                T['Pf'] = np.add.reduceat(fr * np.abs(R[0]) ** 2,
                                          edges[:-1])
            wb = np.add.reduceat(w, edges[:-1]) / np.diff(edges)
            if key is not None:
                self._wb[key] = wb
            G = 4. * self.df * (T['Pb'] @ wb)
        if getattr(self, 'banded_gram', None) is None and nh is None:
            Gx = 4. * self.df * (np.conj(R) * w) @ R.T
            G0 = np.asarray(raw_gram_matrix(T['raws'], psd.array()
                                            if hasattr(psd, 'array') else psd,
                                            self.df, self.kmin, self.kmax))
            sc = np.abs(np.diag(G0)).max()
            tr = np.abs(Gx.T - G0).max() < np.abs(Gx - G0).max()
            self.gram_transpose = bool(tr)
            ex_err = np.abs((Gx.T if tr else Gx) - G0).max() / sc
            if use_bins:
                b_err = np.abs((G.T if tr else G) - G0).max() / sc
                self.banded_gram = bool(b_err < 1e-4 and ex_err < 1e-8)
            else:
                b_err = np.nan
                self.banded_gram = False
            self.exact_gram_ok = bool(ex_err < 1e-8)
            logger.info('Exact term: harmonic inner products (%d frequency '
                        'bins): exact matrix product vs raw_gram_matrix %.1e,'
                        ' %d-band sum vs raw_gram_matrix %.1e -> using %s',
                        nk, ex_err, nb, b_err,
                        'the band sums' if self.banded_gram else
                        ('the exact matrix product' if self.exact_gram_ok
                         else 'raw_gram_matrix'))
            if not self.banded_gram:
                return Gx.T if tr and self.exact_gram_ok else (
                    Gx if self.exact_gram_ok else G0)
        if getattr(self, 'banded_gram', None) and use_bins:
            # free the harmonics: only the band sums are needed from now on
            T.pop('R', None)
            T['raws'] = None
            return G.T if self.gram_transpose else G
        if getattr(self, 'banded_gram', None) is None:
            # not yet checked (an aligned template came first): exact sum
            Gx = 4. * self.df * (np.conj(R) * w) @ R.T
            return Gx
        if not getattr(self, 'exact_gram_ok', True):
            return raw_gram_matrix(T['raws'], psd.array() if hasattr(
                psd, 'array') else psd, self.df, self.kmin, self.kmax)
        Gx = 4. * self.df * (np.conj(R) * w) @ R.T
        return Gx.T if self.gram_transpose else Gx

    def _precessing_ep_ec(self, T, psd):
        """
        Responses of every grid orientation, as in
        tha_marginalize.derive_ep_ec_hh (same algebra, with the harmonic
        inner products from _gram).
        """
        from pycbc.filter.tha_marginalize import derive_gram_schmidt_transform
        np = numpy
        A, B = np.asarray(T['A']), np.asarray(T['B'])
        Graw = self._gram(T, psd)
        Tm, sigma = derive_gram_schmidt_transform(Graw)
        Gn = Graw / np.outer(sigma, sigma)
        M = (Tm.conj() @ Gn) * sigma[None, :]
        Ep = A @ M.T
        Ec = B @ M.T
        hh_pp = np.einsum('ij,jk,ik->i', A.conj(), Graw, A).real
        hh_cc = np.einsum('ij,jk,ik->i', B.conj(), Graw, B).real
        hh_pc = np.einsum('ij,jk,ik->i', A.conj(), Graw, B)
        return Ep, Ec, hh_pp, hh_cc, hh_pc

    def _aligned_ep_ec(self, T, psd):
        """
        Aligned-spin templates have one harmonic, so the stored SNRs are in
        the one-vector basis e1 = h1 / |h1| and the grid point's h+, hx are
        A[:, 0] h1, B[:, 0] h1 (the grid was built in that 1-harmonic
        subspace): the responses are A[:, 0] |h1|, B[:, 0] |h1|.
        """
        nh = numpy.sqrt(numpy.real(self._gram(T, psd, nh=1)[0, 0]))
        ng = len(T['A'])
        Ep = numpy.zeros((ng, 5), dtype=complex)
        Ec = numpy.zeros((ng, 5), dtype=complex)
        Ep[:, 0] = T['A'][:, 0] * nh
        Ec[:, 0] = T['B'][:, 0] * nh
        hpp = numpy.abs(Ep[:, 0]) ** 2
        hcc = numpy.abs(Ec[:, 0]) ** 2
        hpc = numpy.conj(Ep[:, 0]) * Ec[:, 0]
        return Ep, Ec, hpp, hcc, hpc

    def _psd_indices(self, ifo, t):
        """_psd_index for an array of times (-1 where none covers it)"""
        st, et = self.psd_times[ifo]
        mid = 0.5 * (st + et)
        t = numpy.asarray(t, dtype=numpy.float64)
        inside = (st[None, :] <= t[:, None]) & (t[:, None] < et[None, :])
        dist = numpy.where(inside, numpy.abs(mid[None, :] - t[:, None]),
                           numpy.inf)
        idx = numpy.argmin(dist, axis=1)
        if self.psd_mode == 'reference':
            idx[:] = 0
        idx[~inside.any(axis=1)] = -1
        return idx

    def __call__(self, stats, dt, tid, ln_twin2pi, force_loud=False):
        """
        Exact term for the coincs given by single-ifo records stats[ifo]
        (from the statistic's single()), H1-L1 time differences dt (with
        the time slide applied) and template id tid. Returns NaN where it
        cannot be computed (the caller then keeps the flow value).
        """
        if not self.loaded:
            self.load()
        n = len(dt)
        out = numpy.full(n, numpy.nan)
        if n == 0 or self._template(tid) is None:
            return out
        dt = numpy.asarray(dt, dtype=numpy.float64)
        h, l = self.ifos
        ip = {ifo: self._psd_indices(ifo, stats[ifo]['end_time'])
              for ifo in self.ifos}
        snr2 = sum(2. * numpy.maximum(numpy.array(stats[ifo]['marg_lnl'],
                                                  dtype=numpy.float64), 1e-9)
                   for ifo in self.ifos)
        ok = (ip[h] >= 0) & (ip[l] >= 0)
        if self.method == 'reference' or force_loud:
            quiet = numpy.zeros(n, dtype=bool)
        else:
            quiet = ok & (snr2 < self.quiet_snr ** 2)
        loud = ok & ~quiet
        sub = lambda m: {ifo: stats[ifo][m] for ifo in self.ifos}
        keys = numpy.stack([ip[h], ip[l]], axis=1)
        # quiet coincs: batches sharing the PSD segments
        for key in (numpy.unique(keys[quiet], axis=0) if quiet.any() else []):
            m = numpy.nonzero(quiet & (keys[:, 0] == key[0])
                              & (keys[:, 1] == key[1]))[0]
            ipsd = (int(key[0]), int(key[1]))
            try:
                if self.method == 'ring':
                    out[m] = self._batch(sub(m), dt[m], tid, ipsd)
                else:
                    out[m] = self._grid(sub(m), dt[m], tid, ipsd,
                                        self.q_nodes,
                                        orient_step=self.quiet_orient_step,
                                        prune=None, n_ang=self.quiet_ang)
                out[m] += ln_twin2pi
            except Exception as e:  # pylint:disable=broad-except
                logger.warning('Exact coincident term (batch) failed for '
                               'template %d: %s', tid, e)
        self.n_quiet += int(quiet.sum())
        # loud coincs: one at a time, keeping the orientations that matter
        for j in numpy.nonzero(loud)[0]:
            try:
                if self.method == 'ring':
                    out[j] = self._one(sub(j), float(dt[j]), tid)
                elif self.method == 'reference':
                    out[j] = self._grid(sub([j]), dt[[j]], tid,
                                        (int(ip[h][j]), int(ip[l][j])),
                                        self.r_nodes, orient_step=1,
                                        prune=None, n_ang=64)[0]
                else:
                    out[j] = self._grid(sub([j]), dt[[j]], tid,
                                        (int(ip[h][j]), int(ip[l][j])),
                                        self.l_nodes, orient_step=1,
                                        prune=self.prune, n_ang=64)[0]
                out[j] += ln_twin2pi
            except Exception as e:      # pylint:disable=broad-except
                logger.warning('Exact coincident term failed for template '
                               '%d: %s', tid, e)
        self.n_loud += int(loud.sum())
        # spot checks against the yardstick
        if self.check_frac > 0 and self.method != 'reference' and \
                not force_loud:
            chk = numpy.nonzero(ok)[0]
            chk = chk[self.check_rng.random(len(chk)) < self.check_frac]
            for j in chk:
                try:
                    full = self.reference_value(sub([j]), dt[[j]], tid,
                                                (int(ip[h][j]),
                                                 int(ip[l][j]))) + ln_twin2pi
                    self.check_diffs.append(out[j] - full)
                    self.check_full.append(full)
                    self.check_kind.append('quiet' if quiet[j] else 'loud')
                except Exception:       # pylint:disable=broad-except
                    pass
        return out

    def reference_value(self, rec, dt, tid, ipsd):
        """The yardstick value for one coinc (without ln_twin2pi)"""
        import time as _time
        t0, s0 = _time.time(), self.t_coinc
        if self.check_ref == 'ring':
            v = self._one({k: v[0] for k, v in rec.items()}, float(dt[0]),
                          tid)
        else:
            if not hasattr(self, 'bhat'):
                n_e = numpy.stack([numpy.cos(self.lat) * numpy.cos(self.lon),
                                   numpy.cos(self.lat) * numpy.sin(self.lon),
                                   numpy.sin(self.lat)], axis=1)
                self._setup_geometry(n_e, {ifo: numpy.asarray(
                    self.dets[ifo].response, dtype=numpy.float64)
                    for ifo in self.ifos})
            v = self._grid(rec, dt, tid, ipsd, self.r_nodes, orient_step=1,
                           prune=None, n_ang=64)[0]
        # checks are not part of the coinc time
        self.t_coinc = s0
        return v

    # ---- baseline-coordinate sky quadrature ---------------------------
    def _setup_geometry(self, n_e, responses):
        """
        H1-L1 baseline and detector tensors in the Earth frame, from the
        Fibonacci sky points' delays and pycbc's antenna patterns (fitted, so
        the sign conventions are pycbc's own): tau(n) = b . n.
        """
        b, res, _, _ = numpy.linalg.lstsq(n_e, self.tau, rcond=None)
        resid = numpy.abs(n_e @ b - self.tau).max()
        self.D = float(numpy.linalg.norm(b))
        self.bhat = b / self.D
        tmp = numpy.array([1., 0., 0.]) if abs(self.bhat[0]) < 0.9 \
            else numpy.array([0., 1., 0.])
        self.e1 = numpy.cross(self.bhat, tmp)
        self.e1 /= numpy.linalg.norm(self.e1)
        self.e2 = numpy.cross(self.bhat, self.e1)
        self.Dt = {ifo: numpy.asarray(r, dtype=numpy.float64)
                   for ifo, r in responses.items()}
        self.hand = 1.
        # check against the Fibonacci points' antenna patterns: |F|^2 is the
        # same in any polarisation basis; the 2D cross product of F_H and
        # F_L fixes the handedness
        sel = numpy.linspace(0, len(n_e) - 1, 4000).astype(int)
        mine = {ifo: self._ant(n_e[sel], ifo) for ifo in self.ifos}
        h, l = self.ifos
        err = 0.
        for ifo in self.ifos:
            p0, _, c0, _ = (v[sel].astype(numpy.float64)
                            for v in self.F[ifo])
            err = max(err, numpy.abs(mine[ifo][0] ** 2 + mine[ifo][1] ** 2
                                     - p0 ** 2 - c0 ** 2).max())
        ph0, _, ch0, _ = (v[sel].astype(numpy.float64) for v in self.F[h])
        pl0, _, cl0, _ = (v[sel].astype(numpy.float64) for v in self.F[l])
        cross_p = ph0 * cl0 - ch0 * pl0
        cross_m = mine[h][0] * mine[l][1] - mine[h][1] * mine[l][0]
        if numpy.sum(cross_p * cross_m) < 0:
            self.hand = -1.
        mine = {ifo: self._ant(n_e[sel], ifo) for ifo in self.ifos}
        cross_m = mine[h][0] * mine[l][1] - mine[h][1] * mine[l][0]
        cerr = numpy.abs(cross_m - cross_p).max()
        logger.info('Exact coincident term, sky quadrature: baseline %.3f '
                    'ms (delay fit error %.1e s), antenna |F|^2 error %.1e, '
                    'handedness %+d (error %.1e)', self.D * 1e3, resid, err,
                    int(self.hand), cerr)
        if resid > 1e-6 or err > 1e-3 or cerr > 1e-3:
            raise RuntimeError('Exact coincident term: the baseline '
                               'geometry does not match pycbc\'s antenna '
                               'patterns (use exact_method:ring)')

    def _ant(self, n, ifo):
        """F+, Fx at psi = 0 in a basis of our own (any basis will do, as
        the polarisation is integrated over a full period)"""
        z = numpy.array([0., 0., 1.])
        X = numpy.cross(z, n)
        nx = numpy.linalg.norm(X, axis=-1, keepdims=True)
        alt = numpy.cross(numpy.array([1., 0., 0.]), n)
        X = numpy.where(nx > 1e-6, X, alt)
        X /= numpy.linalg.norm(X, axis=-1, keepdims=True)
        Y = numpy.cross(n, X)
        D = self.Dt[ifo]
        DX = X @ D
        DY = Y @ D
        fp = numpy.sum(X * DX, axis=-1) - numpy.sum(Y * DY, axis=-1)
        fc = 2. * numpy.sum(X * DY, axis=-1)
        return fp, self.hand * fc

    def _legendre(self, k):
        if k not in self._gl:
            self._gl[k] = numpy.polynomial.legendre.leggauss(k)
        return self._gl[k]

    def _sky_nodes(self, dt, sigf, n_tau, n_az):
        """
        Quadrature nodes over the sky for each coinc: mu = cos(angle to the
        baseline), on [mu_c - 6 sigma/D, mu_c + 6 sigma/D] cut to [-1, 1]
        (Gauss-Legendre), and the azimuth around the baseline (evenly
        spaced). Returns the antenna patterns at psi = 0 per ifo, shape
        (n, n_tau * n_az), and ln(sky weight x time density) for each node.
        """
        x, w = self._legendre(n_tau)
        mu_c = numpy.clip(dt / self.D, -1., 1.)
        half = 6. * sigf / self.D
        a = numpy.maximum(-1., mu_c - half)
        b = numpy.minimum(1., mu_c + half)
        mu = 0.5 * (a + b)[:, None] + 0.5 * (b - a)[:, None] * x[None, :]
        # uniform sky: d mu / 2 (times d phi / 2 pi, below)
        lnwmu = numpy.log(0.25 * (b - a))[:, None] + numpy.log(w)[None, :]
        tau = self.D * mu
        lnT = (-0.5 * ((dt[:, None] - tau) / sigf[:, None]) ** 2
               - numpy.log(numpy.sqrt(2 * numpy.pi) * sigf)[:, None])
        phi = 2 * numpy.pi * (numpy.arange(n_az) + 0.5) / n_az
        st = numpy.sqrt(numpy.maximum(1. - mu ** 2, 0.))
        nv = (mu[:, :, None, None] * self.bhat
              + st[:, :, None, None] * (numpy.cos(phi)[None, None, :, None]
                                        * self.e1
                                        + numpy.sin(phi)[None, None, :, None]
                                        * self.e2))
        nv = nv.reshape(len(dt), n_tau * n_az, 3)
        F0 = {ifo: self._ant(nv, ifo) for ifo in self.ifos}
        lnsky = numpy.repeat(lnwmu + lnT, n_az, axis=1) - numpy.log(n_az)
        self._last_tau = numpy.repeat(tau, n_az, axis=1)
        return F0, lnsky

    def _singles(self, Q, n_ang, keep=None, lnw=None):
        """Single-ifo marginal lambda over the antenna-vector angle for each
        coinc (orientations outside keep left out)"""
        from scipy.special import logsumexp
        ang = (numpy.arange(n_ang) + 0.5) * numpy.pi / n_ang
        ca = numpy.cos(ang)[None, :, None]
        sa = numpy.sin(ang)[None, :, None]
        q11, q22, q12, m11, m22, m12 = Q
        num = (ca ** 2 * q11[:, None, :] + sa ** 2 * q22[:, None, :]
               + 2 * ca * sa * q12[:, None, :])
        den = ca ** 2 * m11 + sa ** 2 * m22 + 2 * ca * sa * m12 + 1e-300
        t = lnw[None, None, :] + num / (2 * den)
        if keep is not None:
            t = numpy.where(keep[:, None, :], t, -numpy.inf)
        return logsumexp(t, axis=(1, 2)) - numpy.log(n_ang)

    def _proj(self, stats, R):
        """Harmonic SNRs projected on the responses, the quadratic forms and
        the best single-ifo ln L per orientation"""
        Z = numpy.stack([numpy.asarray(stats['snr_comp_%d' % q],
                                       dtype=numpy.float64)
                         * numpy.exp(1j * numpy.asarray(
                             stats['coa_phase_comp_%d' % q],
                             dtype=numpy.float64))
                         for q in range(1, 6)], axis=1)
        P = (Z @ R['Ep'].T, Z @ R['Ec'].T)
        Q = self._quad(P, R)
        q11, q22, q12, m11, m22, m12 = Q
        a = m11 * m22 - m12 ** 2
        b = m11 * q22 + m22 * q11 - 2 * m12 * q12
        c = q11 * q22 - q12 ** 2
        a = numpy.where(a > 0, a, 1e-300)
        lt = 0.25 * (b + numpy.sqrt(numpy.maximum(b * b - 4 * a * c, 0.))) / a
        return P, Q, lt

    def _grid(self, stats, dt, tid, ipsd, nodes, orient_step=1, prune=None,
              n_ang=16):
        """
        The exact term (without ln_twin2pi) for coincs of one template and
        one pair of PSD segments, by quadrature: nodes = (n_tau, n_az, n_pol)
        over the delay, the azimuth around the baseline and the
        polarisation; every orient_step-th orientation (staggered, each
        standing for orient_step of them); with prune, only orientations
        where either ifo is within prune of its own best are kept.
        """
        import time as _time
        t0 = _time.time()
        s0 = self.t_setup
        n_tau, n_az, n_pol = nodes
        T = self._template(tid)
        h, l = self.ifos
        R = {h: self._responses(tid, h, ipsd[0]),
             l: self._responses(tid, l, ipsd[1])}
        lnw = T['ln_w']
        n = len(dt)
        ng = len(lnw)
        P, Q, lt, snr = {}, {}, {}, {}
        for ifo in self.ifos:
            P[ifo], Q[ifo], lt[ifo] = self._proj(stats[ifo], R[ifo])
            snr[ifo] = numpy.sqrt(2. * numpy.maximum(numpy.asarray(
                stats[ifo]['marg_lnl'], dtype=numpy.float64), 1e-9))
        keep_i = None
        if prune is not None:
            keep_i = {ifo: lt[ifo] >= lt[ifo].max(axis=1)[:, None] - prune
                      for ifo in self.ifos}
        lam = sum(self._singles(Q[ifo], n_ang,
                                None if keep_i is None else keep_i[ifo], lnw)
                  for ifo in self.ifos)

        sig0 = numpy.hypot(self.timing_uncertainty * self.snr_reference
                           / snr[h],
                           self.timing_uncertainty * self.snr_reference
                           / snr[l])
        sigf = numpy.sqrt(sig0 ** 2 + 2 * self.timing_floor ** 2)
        F0, lnsky = self._sky_nodes(dt, sigf, n_tau, n_az)
        rot = None
        if self.phase_time:
            fb = self.phase_time_freq
            if fb <= 0:
                fb = numpy.nanmean([R[h].get('fbar', numpy.nan),
                                    R[l].get('fbar', numpy.nan)])
            if numpy.isfinite(fb) and fb > 0:
                self.fbar_used = getattr(self, 'fbar_used', []) + [fb]
                # (n, S): rotation of L1's phase at each sky node
                rot = (-self.phase_time_sign * 2 * numpy.pi * fb
                       * (dt[:, None] - self._last_tau))
        pols = (numpy.arange(n_pol) + 0.5) * (numpy.pi / 2) / n_pol
        c2 = numpy.cos(2 * pols).astype(numpy.float32)
        s2 = numpy.sin(2 * pols).astype(numpy.float32)
        lnpol = -numpy.log(n_pol)
        ln_jac = numpy.log(snr[h] * snr[l])

        if orient_step > 1 and ng > 100:
            row = numpy.arange(ng) // 24
            sub = numpy.nonzero((numpy.arange(ng) % orient_step)
                                == (row % orient_step))[0]
            lnw_s = lnw[sub] + numpy.log(ng / len(sub))
        else:
            sub = numpy.arange(ng)
            lnw_s = lnw
        if prune is not None:
            # per coinc (called with one coinc for loud ones): union
            kk = numpy.zeros(len(sub), dtype=bool)
            for ifo in self.ifos:
                kk |= keep_i[ifo][:, sub].any(axis=0)
            sub, lnw_s = sub[kk], lnw_s[kk]
        lnw_s = lnw_s.astype(numpy.float32)
        S = n_tau * n_az
        per = S * n_pol * len(sub)
        chunk = max(1, int(self.batch_elems // max(per, 1)))
        resp = {ifo: [R[ifo][k][sub].astype(numpy.float32)
                      for k in ('hpp', 'hcc', 'hpc')] for ifo in self.ifos}
        joint = numpy.empty(n)
        f32 = numpy.float32
        for s_ in range(0, n, chunk):
            sl = slice(s_, s_ + chunk)
            F, Pri = {}, {}
            for ifo in self.ifos:
                f0, g0 = (v[sl].astype(f32)[:, :, None] for v in F0[ifo])
                F[ifo] = ((f0 * c2 + g0 * s2)[..., None],     # (m, S, P, 1)
                          (g0 * c2 - f0 * s2)[..., None])
                pp = P[ifo][0][sl][:, sub]
                pc = P[ifo][1][sl][:, sub]
                Pri[ifo] = tuple(x.astype(f32)[:, None, None, :] for x in
                                 (pp.real, pc.real, pp.imag, pc.imag))
            rr = None
            if rot is not None:
                rs = rot[sl]
                rr = (numpy.cos(rs).astype(f32)[:, :, None, None],
                      numpy.sin(rs).astype(f32)[:, :, None, None])
            tot = self._kernel(F, Pri, resp, lnw_s,
                               lnsky[sl].astype(f32)[:, :, None, None], rr)
            m = tot.reshape(len(tot), -1).max(axis=1)
            tot -= m[:, None, None, None]
            numpy.exp(tot, out=tot)
            ssum = tot.reshape(len(tot), -1).sum(axis=1, dtype=numpy.float64)
            joint[sl] = m + numpy.log(ssum) + lnpol + ln_jac[sl]
            del tot
        self.t_coinc += _time.time() - t0 - (self.t_setup - s0)
        return joint - lam

    def _kernel(self, F, Pri, resp, lnw, lnsky, rot=None):
        """
        ln of the joint integrand at every (coinc, sky node, polarisation,
        orientation), in single-precision real arithmetic (no complex
        temporaries):
            L_J + d2 eps2|a_J|^2 / (2 V0 V) - ln(2 pi Y_L V) + ln w + ln sky
        with X_d = F+ P+ + Fx Px, Y_d = F.M.F, a_J = X_J / Y_J, d2 =
        |X_L/Y_L - X_H/Y_H|^2, V0 = 1/Y_H + 1/Y_L, V = V0 + eps2 |a_J|^2
        (the same as L_H + L_L - d2 / 2V + ln density, written so that
        nothing cancels).
        """
        f32 = numpy.float32
        h, l = self.ifos
        out = {}
        for ifo in (h, l):
            fp, fc = F[ifo]
            hpp, hcc, hpc = resp[ifo]
            Y = (fp * fp) * hpp
            Y += (fc * fc) * hcc
            Y += (2 * fp * fc) * hpc
            Y += f32(1e-30)
            ppr, pcr, ppi, pci = Pri[ifo]
            xr = fp * ppr
            xr += fc * pcr
            xi = fp * ppi
            xi += fc * pci
            out[ifo] = (Y, xr, xi)
        Yh, xrh, xih = out[h]
        Yl, xrl, xil = out[l]
        if rot is not None:
            # L1's phase rotated back by the time-phase coupling at each sky
            # node (cos, sin of shape (m, S, 1, 1))
            cr, sr = rot
            xrl, xil = xrl * cr - xil * sr, xrl * sr + xil * cr
        YJ = Yh + Yl
        # d2, using the per-ifo best-fit amplitudes
        ar = xrl / Yl
        ar -= xrh / Yh
        ai = xil / Yl
        ai -= xih / Yh
        d2 = ar * ar
        ai *= ai
        d2 += ai
        del ar, ai
        xrh += xrl
        xih += xil
        del xrl, xil
        nJ = xrh * xrh
        xih *= xih
        nJ += xih
        del xrh, xih
        V0 = YJ / (Yh * Yl)
        e = nJ / (YJ * YJ)
        e *= f32(self.eps2)
        V = V0 + e
        tot = nJ / (2 * YJ)                       # L_J
        e *= d2
        e /= (2 * V0 * V)
        tot += e
        del e, d2, V0, nJ
        V *= Yl
        V *= f32(2 * numpy.pi)
        numpy.log(V, out=V)
        tot -= V
        tot += lnw
        tot += lnsky
        return tot

    def trigger_terms(self, ifo, stats, tid):
        """
        For single triggers of one ifo and template: c = (best ln L over the
        orientations and polarisation mix) - lambda, and the SNR used in the
        exact term. c bounds how much the other ifo's trigger can raise the
        joint sum (see ExpFitFgBgNormMARGExactStatistic.coinc_upper_bound).
        NaN where it cannot be computed.
        """
        if not self.loaded:
            self.load()
        n = len(stats)
        c = numpy.full(n, numpy.nan)
        rho = numpy.sqrt(2. * numpy.maximum(numpy.asarray(
            stats['marg_lnl'], dtype=numpy.float64), 1e-9))
        if n == 0 or self._template(tid) is None:
            return c, rho
        import time as _time
        t0 = _time.time()
        s0 = self.t_setup
        ip = self._psd_indices(ifo, stats['end_time'])
        lnw = self._template(tid)['ln_w']
        # One response set per template and ifo (the median PSD): c only has
        # to bound the coinc term, and the bound's margin covers the small
        # PSD dependence. Computing it per PSD segment for every trigger was
        # much slower than the coincs themselves.
        m = numpy.nonzero(ip >= 0)[0]
        try:
            R = self._responses(tid, ifo, 'ref')
            for s_ in range(0, len(m), 4000):
                mm = m[s_:s_ + 4000]
                _, Q, lt = self._proj(stats[mm], R)
                lam = self._singles(Q, 8, None, lnw)
                c[mm] = lt.max(axis=1) - lam
        except Exception as e:  # pylint:disable=broad-except
            logger.warning('Exact term bound failed for template %d: %s',
                           tid, e)
        self.t_bterms = getattr(self, 't_bterms', 0.) + _time.time() - t0 \
            - (self.t_setup - s0)
        self.n_bterms = getattr(self, 'n_bterms', 0) + n
        self._bt_calls = getattr(self, '_bt_calls', 0) + 1
        if self._bt_calls <= 4 or self._bt_calls % 20 == 0:
            logger.info('Exact term bound terms: template %d %s, %d triggers, '
                        '%.1f s so far (%d calls)', tid, ifo, n,
                        self.t_bterms, self._bt_calls)
        return c, rho

    def _batch(self, stats, dt, tid, ipsd):
        """
        The exact term for many quiet coincs of one template at once (same
        PSD segments), with quiet_ring sky points along each coinc's ring,
        quiet_pol polarisations and quiet_ang antenna angles for the
        single-ifo marginals. Same quantity as _one.
        """
        import time as _time
        from scipy.special import logsumexp
        t0 = _time.time()
        s0 = self.t_setup
        T = self._template(tid)
        h, l = self.ifos
        R = {h: self._responses(tid, h, ipsd[0]),
             l: self._responses(tid, l, ipsd[1])}
        lnw = T['ln_w']
        n = len(dt)
        ng = len(lnw)
        P, Q, lt, snr = {}, {}, {}, {}
        for ifo in self.ifos:
            st = stats[ifo]
            Z = numpy.stack([st['snr_comp_%d' % q]
                             * numpy.exp(1j * st['coa_phase_comp_%d' % q])
                             for q in range(1, 6)], axis=1).astype(complex)
            P[ifo] = (Z @ R[ifo]['Ep'].T, Z @ R[ifo]['Ec'].T)     # (n, ng)
            Q[ifo] = self._quad(P[ifo], R[ifo])
            q11, q22, q12, m11, m22, m12 = Q[ifo]
            a = m11 * m22 - m12 ** 2
            b = m11 * q22 + m22 * q11 - 2 * m12 * q12
            c = q11 * q22 - q12 ** 2
            a = numpy.where(a > 0, a, 1e-300)
            lt[ifo] = 0.25 * (b + numpy.sqrt(numpy.maximum(b * b - 4 * a * c,
                                                           0.))) / a
            snr[ifo] = numpy.sqrt(2. * numpy.maximum(
                numpy.array(st['marg_lnl'], dtype=numpy.float64), 1e-9))

        # single-ifo marginals over the antenna angle
        na = self.quiet_ang
        ang = (numpy.arange(na) + 0.5) * numpy.pi / na
        ca, sa = numpy.cos(ang)[None, :, None], numpy.sin(ang)[None, :, None]
        lam = numpy.zeros(n)
        for ifo in self.ifos:
            q11, q22, q12, m11, m22, m12 = Q[ifo]
            num = (ca ** 2 * q11[:, None, :] + sa ** 2 * q22[:, None, :]
                   + 2 * ca * sa * q12[:, None, :])
            den = (ca ** 2 * m11 + sa ** 2 * m22 + 2 * ca * sa * m12
                   + 1e-300)
            lam += logsumexp(lnw[None, None, :] + num / (2 * den),
                             axis=(1, 2)) - numpy.log(na)

        # ring of each coinc: quiet_ring points spread evenly along it
        sig0 = numpy.hypot(self.timing_uncertainty * self.snr_reference
                           / snr[h],
                           self.timing_uncertainty * self.snr_reference
                           / snr[l])
        sigf = numpy.sqrt(sig0 ** 2 + 2 * self.timing_floor ** 2)
        lo = numpy.searchsorted(self.tau_sorted, dt - 6 * sigf)
        hi = numpy.searchsorted(self.tau_sorted, dt + 6 * sigf)
        nr = self.quiet_ring
        # as in _one: if the ring is (nearly) empty, e.g. dt beyond the light
        # travel time, use the 50 closest delays
        short = (hi - lo) < 50
        c0 = numpy.searchsorted(self.tau_sorted, dt)
        lo = numpy.where(short, numpy.clip(c0 - 25, 0, self.n_fine - 50), lo)
        hi = numpy.where(short, lo + 50, hi)
        span = (hi - lo).astype(numpy.float64)
        k = (numpy.arange(nr) + 0.5) / nr
        pos = lo[:, None] + numpy.floor(span[:, None] * k[None, :]).astype(int)
        near = self.tau_order[numpy.clip(pos, 0, self.n_fine - 1)]  # (n, nr)
        npol = self.quiet_pol
        pols = (numpy.arange(npol) + 0.5) * (numpy.pi / 2) / npol
        c2 = numpy.cos(2 * pols).astype(numpy.float32)
        s2 = numpy.sin(2 * pols).astype(numpy.float32)
        tau = self.tau[near]                                      # (n, nr)
        lnT = (-0.5 * ((dt[:, None] - tau) / sigf[:, None]) ** 2
               - numpy.log(numpy.sqrt(2 * numpy.pi) * sigf)[:, None])
        lnsky = numpy.log(span / (nr * self.n_fine))              # (n,)
        lnpol = -numpy.log(npol)
        ln_jac = numpy.log(snr[h] * snr[l])

        k = self.quiet_orient_step
        if k > 1 and ng > 100:
            row = numpy.arange(ng) // 24
            sub = numpy.nonzero((numpy.arange(ng) % k) == (row % k))[0]
            lnw_j = lnw[sub] + numpy.log(ng / len(sub))
        else:
            sub = numpy.arange(ng)
            lnw_j = lnw
        lnw_j = lnw_j.astype(numpy.float32)
        ngj = len(sub)
        per = nr * npol * ngj
        chunk = max(1, int(self.batch_elems // per))
        joint = numpy.empty(n)
        hpp = {ifo: R[ifo]['hpp'][sub].astype(numpy.float32)
               for ifo in self.ifos}
        hcc = {ifo: R[ifo]['hcc'][sub].astype(numpy.float32)
               for ifo in self.ifos}
        hpc = {ifo: R[ifo]['hpc'][sub].astype(numpy.float32)
               for ifo in self.ifos}
        lnT32 = lnT.astype(numpy.float32)
        for s_ in range(0, n, chunk):
            sl = slice(s_, s_ + chunk)
            X, Y = {}, {}
            for ifo in self.ifos:
                p0, p4, q0, q4 = (v[near[sl]][:, :, None]
                                  for v in self.F[ifo])
                fp = (p0 * c2 + p4 * s2)[..., None]       # (m, nr, npol, 1)
                fc = (q0 * c2 + q4 * s2)[..., None]
                pp = P[ifo][0][sl][:, sub].astype(numpy.complex64)[:, None, None, :]
                pc = P[ifo][1][sl][:, sub].astype(numpy.complex64)[:, None, None, :]
                X[ifo] = fp * pp + fc * pc
                Y[ifo] = (fp ** 2 * hpp[ifo] + fc ** 2 * hcc[ifo]
                          + 2. * fp * fc * hpc[ifo] + numpy.float32(1e-30))
            YJ = Y[h] + Y[l]
            XJ = X[h] + X[l]
            LJ = numpy.abs(XJ) ** 2 / (2 * YJ)
            d2 = numpy.abs(X[l] / Y[l] - X[h] / Y[h]) ** 2
            V0 = 1. / Y[h] + 1. / Y[l]
            V = V0 + self.eps2 * numpy.abs(XJ / YJ) ** 2
            core = LJ + d2 / (2 * V0) - d2 / (2 * V) \
                - numpy.log(2 * numpy.pi * Y[l] * V)
            tot = core + lnw_j + lnT32[sl][:, :, None, None]
            m = tot.reshape(len(tot), -1).max(axis=1)
            ssum = numpy.exp(tot - m[:, None, None, None]).reshape(
                len(tot), -1).sum(axis=1, dtype=numpy.float64)
            joint[sl] = m + numpy.log(ssum) + lnsky[sl] + lnpol + ln_jac[sl]
            del X, Y, YJ, XJ, LJ, d2, V0, V, core, tot
        self.t_coinc += _time.time() - t0 - (self.t_setup - s0)
        return joint - lam

    @staticmethod
    def _quad(P, R):
        """
        Single-ifo quadratic forms over the polarisation mix: for antenna
        vector F = (F+, Fx), |X|^2 = F.Q.F and sigma^2 = F.M.F, so the
        single-ifo log likelihood is (F.Q.F) / (2 F.M.F).
        """
        pp, pc = P
        q11 = numpy.abs(pp) ** 2
        q22 = numpy.abs(pc) ** 2
        q12 = numpy.real(numpy.conj(pp) * pc)
        return q11, q22, q12, R['hpp'], R['hcc'], R['hpc']

    def _one(self, rec, dt, tid):
        import time as _time
        from scipy.special import logsumexp
        t0 = _time.time()
        s0 = self.t_setup
        T = self._template(tid)
        R, P, Q, lt, snrs = {}, {}, {}, {}, []
        for ifo in self.ifos:
            r = rec[ifo]
            ipsd = self._psd_index(ifo, float(r['end_time']))
            if ipsd is None:
                return numpy.nan
            R[ifo] = self._responses(tid, ifo, ipsd)
            z = numpy.array([r['snr_comp_%d' % q]
                             * numpy.exp(1j * r['coa_phase_comp_%d' % q])
                             for q in range(1, 6)], dtype=complex)
            P[ifo] = (R[ifo]['Ep'] @ z, R[ifo]['Ec'] @ z)
            Q[ifo] = self._quad(P[ifo], R[ifo])
            q11, q22, q12, m11, m22, m12 = Q[ifo]
            # best single-ifo ln L at each orientation over the polarisation
            # mix: largest generalised eigenvalue of (Q, M), over 2
            a = m11 * m22 - m12 ** 2
            b = m11 * q22 + m22 * q11 - 2 * m12 * q12
            c = q11 * q22 - q12 ** 2
            a = numpy.where(a > 0, a, 1e-300)
            lt[ifo] = 0.25 * (b + numpy.sqrt(numpy.maximum(b * b - 4 * a * c,
                                                           0.))) / a
            snrs.append(numpy.sqrt(2. * max(float(r['marg_lnl']), 1e-9)))
        lnw_all = T['ln_w']

        # single-ifo marginals: for one ifo the sky position and polarisation
        # only enter through the antenna vector's angle (its length is
        # absorbed by the amplitude fit), which is uniform: average over the
        # angle directly, at the orientations within prune of the best
        ang = self.ang
        ca, sa = numpy.cos(ang)[:, None], numpy.sin(ang)[:, None]
        lam = 0.
        for ifo in self.ifos:
            k = lt[ifo] >= lt[ifo].max() - self.prune
            q11, q22, q12, m11, m22, m12 = [v[k] for v in Q[ifo]]
            num = ca ** 2 * q11 + sa ** 2 * q22 + 2 * ca * sa * q12
            den = ca ** 2 * m11 + sa ** 2 * m22 + 2 * ca * sa * m12 + 1e-300
            lam += logsumexp(lnw_all[k][None, :] + num / (2 * den)) \
                - numpy.log(len(ang))

        # joint: orientations where either ifo is within prune of its own
        # best (not only where the two together are: when the amplitude/
        # phase penalty is large, the orientation that dominates the joint
        # sum need not be one where the separate fits are best), and the
        # ring of sky positions whose delay matches dt
        keep = numpy.zeros(len(lnw_all), dtype=bool)
        for ifo in self.ifos:
            keep |= lt[ifo] >= lt[ifo].max() - self.prune
        lnw = lnw_all[keep]
        sig0 = numpy.hypot(*[self.timing_uncertainty * self.snr_reference / s
                             for s in snrs])
        sigf = numpy.sqrt(sig0 ** 2 + 2 * self.timing_floor ** 2)
        lo = numpy.searchsorted(self.tau_sorted, dt - 6 * sigf)
        hi = numpy.searchsorted(self.tau_sorted, dt + 6 * sigf)
        if hi - lo < 50:
            c0 = numpy.searchsorted(self.tau_sorted, dt)
            lo, hi = max(0, c0 - 25), min(self.n_fine, c0 + 25)
        step = max(1, int(numpy.ceil((hi - lo) / self.max_ring)))
        near = self.tau_order[lo:hi:step]
        h, l = self.ifos
        nr, npol = len(near), self.n_pol
        c2 = numpy.cos(2 * self.pols).astype(numpy.float32)
        s2 = numpy.sin(2 * self.pols).astype(numpy.float32)
        X, Y = {}, {}
        for ifo in self.ifos:
            p0, p4, q0, q4 = (v[near][:, None] for v in self.F[ifo])
            fp = (p0 * c2 + p4 * s2)[:, :, None]
            fc = (q0 * c2 + q4 * s2)[:, :, None]
            Rr = R[ifo]
            pp = P[ifo][0][keep].astype(numpy.complex64)
            pc = P[ifo][1][keep].astype(numpy.complex64)
            X[ifo] = fp * pp + fc * pc
            Y[ifo] = (fp ** 2 * Rr['hpp'][keep].astype(numpy.float32)
                      + fc ** 2 * Rr['hcc'][keep].astype(numpy.float32)
                      + 2. * fp * fc * Rr['hpc'][keep].astype(numpy.float32)
                      + numpy.float32(1e-30))
        YJ = Y[h] + Y[l]
        LJ = numpy.abs(X[h] + X[l]) ** 2 / (2 * YJ)
        # amplitude/phase density with the systematic floor
        aH, aL = X[h] / Y[h], X[l] / Y[l]
        V0 = 1. / Y[h] + 1. / Y[l]
        V = V0 + self.eps2 * numpy.abs((X[h] + X[l]) / YJ) ** 2
        d2 = numpy.abs(aL - aH) ** 2
        # LH + LL - pen, with pen = d2 / 2V  (= LJ + d2 / 2V0 - d2 / 2V)
        core = LJ + d2 / (2 * V0) - d2 / (2 * V)
        ln_dens = numpy.log(snrs[0] * snrs[1]) - numpy.log(2 * numpy.pi * Y[l] * V)
        tau = self.tau[near]
        lnT = (-0.5 * ((dt - tau) / sigf) ** 2
               - numpy.log(numpy.sqrt(2 * numpy.pi) * sigf))[:, None, None]
        lnpol = -numpy.log(npol)
        w_abs = lnw[None, None, :] + numpy.log(step / self.n_fine) + lnpol
        joint = logsumexp((w_abs + core + ln_dens + lnT).astype(numpy.float64))
        # coinc time only: leave out any template setup done in this call
        self.t_coinc += _time.time() - t0 - (self.t_setup - s0)
        return joint - lam


class ExpFitFgBgNormMARGExactStatistic(ExpFitFgBgNormMARGStatistic):
    """
    phasetd_exp_fit_fgbg_norm_marg with the coincident term computed exactly
    (ExactCoincTerm) for EVERY coinc, precessing and aligned-spin, instead of
    by the flows.

    The coincident term of phasetd_exp_fit_fgbg_norm_marg is
        (signal flow + Jacobian - noise flow) - log(noise volume)
    and is replaced here by ExactCoincTerm's value. The single-detector
    noise rates, network sensitivity and SNR^-4 terms are unchanged, so the
    two statistics differ only in the coincident term. The flow value is
    still computed and kept in self.last_flow (one array per call), so the
    two can be compared coinc by coinc (pycbc_coinc_findtrigs writes it as
    'stat_flow'). Where the exact term cannot be computed (no PSD covering
    the trigger time, template missing from the grid) the flow value is
    used and counted in self.n_fallback.

    Statistic keywords (--statistic-keywords KEY:VALUE):
      exact_grid           marginalisation grid file (as for the inspiral)
      exact_bank           template bank (BANK2HDF; pycbc_coinc_findtrigs
                           passes its --template-bank automatically)
      exact_psd_<IFO>      merged PSD file of each ifo (MERGE_PSDS)
      exact_max_ring, exact_n_pol, exact_prune, exact_amp_phase_floor,
      exact_timing_floor, exact_timing_uncertainty, exact_snr_reference,
      exact_n_fine        see ExactCoincTerm
    """

    def __init__(self, sngl_ranking, files=None, ifos=None, **kwargs):
        ex = {k[6:]: kwargs.pop(k) for k in list(kwargs)
              if k.startswith('exact_')}
        ExpFitFgBgNormMARGStatistic.__init__(self, sngl_ranking, files=files,
                                             ifos=ifos, **kwargs)
        self.single_dtype.append(('template_id', numpy.int64))
        psds = {k[4:]: ex.pop(k) for k in list(ex) if k.startswith('psd_')}
        grid = ex.pop('grid', None)
        bank = ex.pop('bank', None)
        missing = [n for n, v in [('exact_grid', grid), ('exact_bank', bank)]
                   if v is None]
        missing += ['exact_psd_%s' % i for i in self.ifos if i not in psds]
        if missing:
            raise ValueError('The exact statistic needs the statistic '
                             'keywords %s' % ', '.join(missing))
        if not self.has_hist:
            self.get_hist()
        def conv(v):
            try:
                return float(v)
            except (TypeError, ValueError):
                return v
        # exact_flow:0 skips the flow statistic (kept only where the exact
        # term fails); exact_check_bound:1 compares every computed coinc with
        # its upper bound (pycbc_coinc_findtrigs then filters nothing)
        self.compute_flow = bool(conv(ex.pop('flow', 0)))
        self.check_bound = bool(conv(ex.pop('check_bound', 0)))
        self.single_dtype.append(('exact_c', numpy.float32))
        self.single_dtype.append(('exact_rho', numpy.float32))
        self.exact = ExactCoincTerm(self.model_ifos, grid, bank, psds,
                                    **{k: conv(v) for k, v in ex.items()})
        self.n_exact = 0
        self.n_fallback = 0
        self.last_flow = None
        self.rho_range = {}
        self.bound_viol = 0
        self.bound_checked = 0
        self.bound_excess = []
        self.bound_tested = 0
        self.bound_passed = 0

    def single(self, trigs):
        singles = ExpFitFgBgNormMARGStatistic.single(self, trigs)
        try:
            singles['template_id'] = trigs.template_num
        except AttributeError:
            singles['template_id'] = trigs['template_id'][:]
        # per-trigger terms of the upper bound used to skip coincs that
        # cannot reach a decimation threshold
        singles['exact_c'] = numpy.nan
        singles['exact_rho'] = numpy.sqrt(2. * numpy.maximum(
            singles['marg_lnl'].astype(numpy.float64), 1e-9))
        ifo = getattr(trigs, 'ifo', None)
        if ifo in self.model_ifos and len(singles):
            try:
                tid = int(numpy.atleast_1d(singles['template_id'])[0])
                if numpy.all(singles['template_id'] == tid):
                    c, _ = self.exact.trigger_terms(ifo, singles, tid)
                    singles['exact_c'] = c
            except Exception as e:  # pylint:disable=broad-except
                logger.warning('Exact term bound terms failed: %s', e)
            r = singles['exact_rho']
            self.rho_range[ifo] = (float(r.min()), float(r.max()))
        return singles

    def _exact_bound_terms(self, rho_ref, rho_oth, c_ref, c_oth,
                           time_addition, dt=None):
        """
        Upper bound on (exact term + SNR terms): the amplitude/phase density
        is at most rho_H rho_L / 2 pi, and the joint sum at most e^(best
        ln L) of one ifo times the other's marginal times the sky-averaged
        time density, so the orientation/sky consistency adds at most
        min(c_H, c_L).
        """
        ex = self.exact
        if not ex.loaded:
            ex.load()
        noise_twindow = coinc_rate.multiifo_noise_coincident_area(
            self.hist_ifos, time_addition)
        sig0 = numpy.hypot(ex.timing_uncertainty * ex.snr_reference / rho_ref,
                           ex.timing_uncertainty * ex.snr_reference / rho_oth)
        sigf = numpy.sqrt(sig0 ** 2 + 2 * ex.timing_floor ** 2)
        c = numpy.minimum(c_ref, c_oth)
        c = numpy.where(numpy.isfinite(c), c, numpy.inf)
        # For one ifo the average of e^{ln L} over the polarisation is the
        # same at every sky position (only the antenna vector's angle
        # matters, and it is uniform), so after bounding the other ifo's
        # factor by its maximum, the time density only enters through its
        # all-sky average: (Phi((D - dt)/s) - Phi((-D - dt)/s)) / 2D, at
        # most 1/2D when dt is not known
        from scipy.special import ndtr
        D = getattr(ex, 'D', 0.010)
        if dt is None:
            lnT = -numpy.log(2 * D) + 0. * sigf
        else:
            p = ndtr((D - dt) / sigf) - ndtr((-D - dt) / sigf)
            lnT = numpy.log(numpy.maximum(p, 1e-300)) - numpy.log(2 * D)
        return (lnT + numpy.log(noise_twindow * 2 * numpy.pi)
                + numpy.log(rho_oth) - 4. * numpy.log(rho_ref / self.ref_snr)
                - numpy.log(2 * numpy.pi)
                + c + ex.bound_margin)

    def _base(self, s, time_addition):
        sngl_rates = {sngl[0]: sngl[1]['snglstat'] for sngl in s}
        ln_noise_rate = coinc_rate.combination_noise_lograte(
                                  sngl_rates, time_addition)
        ln_noise_rate -= self.benchmark_lograte
        network_sigmasq = numpy.amin([sngl[1]['sigmasq'] for sngl in s],
                                     axis=0)
        network_logvol = 1.5 * numpy.log(network_sigmasq)
        network_logvol -= s[0][1]['benchmark_logvol']
        return network_logvol - ln_noise_rate

    def coinc_upper_bound(self, s, slide, step, to_shift, **kwargs):
        """
        Upper bound on rank_stat_coinc for each coinc, from single-trigger
        quantities only (cheap): coincs whose bound is below a decimation
        threshold need not be computed.
        """
        stat = {ifo: st for ifo, st in s}
        ref, other = self.model_ifos[0], self.model_ifos[1]
        shift = {ifo: sh for ifo, sh in zip(self.ifos, to_shift)}
        sh = numpy.array(slide * step, dtype=numpy.float64, ndmin=1)
        dt = (stat[ref]['end_time'] + sh * shift[ref]) \
            - (stat[other]['end_time'] + sh * shift[other])
        ub = self._base(s, kwargs['time_addition']) + self._exact_bound_terms(
            stat[ref]['exact_rho'].astype(numpy.float64),
            stat[other]['exact_rho'].astype(numpy.float64),
            stat[ref]['exact_c'].astype(numpy.float64),
            stat[other]['exact_c'].astype(numpy.float64),
            kwargs['time_addition'], dt=dt)
        return numpy.where(numpy.isfinite(ub), ub, numpy.inf)

    def rank_stat_coinc(self, s, slide, step, to_shift,
                        **kwargs): # pylint:disable=unused-argument
        """
        The coincident statistic with the exact coincident term. The flow
        statistic for the same coincs is left in self.last_flow.
        """
        sngl_rates = {sngl[0]: sngl[1]['snglstat'] for sngl in s}
        ln_noise_rate = coinc_rate.combination_noise_lograte(
                                  sngl_rates, kwargs['time_addition'])
        ln_noise_rate -= self.benchmark_lograte
        network_sigmasq = numpy.amin([sngl[1]['sigmasq'] for sngl in s],
                                     axis=0)
        network_logvol = 1.5 * numpy.log(network_sigmasq)
        benchmark_logvol = s[0][1]['benchmark_logvol']
        network_logvol -= benchmark_logvol
        base = network_logvol - ln_noise_rate

        stat = {ifo: st for ifo, st in s}
        noise_twindow = coinc_rate.multiifo_noise_coincident_area(
                            self.hist_ifos, kwargs['time_addition'])

        n_ifos = len(self.hist_ifos)
        hist_vol = noise_twindow * (2 * numpy.pi * 2 * numpy.pi) ** (n_ifos - 1)
        hist_vol_aligned = noise_twindow * (2 * numpy.pi) ** (n_ifos - 1)

        def flow_stat(m):
            # the flow statistic for the coincs in mask m
            st = {ifo: stat[ifo][m] for ifo in stat}
            shf = numpy.array(slide * step, dtype=numpy.float64, ndmin=1)
            shf = shf[m] if len(shf) == len(m) else shf
            logr_s = self.logsignalrate(st, shf, to_shift)
            aligned = self.is_aligned(st[self.model_ifos[0]]['beta'])
            logr_n = numpy.where(aligned, - numpy.log(hist_vol_aligned),
                                 - numpy.log(hist_vol))
            f = base[m] + logr_s - logr_n
            f[f < -30.] = -30.
            return f

        n = len(base)
        flow = numpy.full(n, numpy.nan)
        if self.compute_flow and n:
            flow = flow_stat(numpy.ones(n, dtype=bool))
        self.last_flow = flow.copy()
        if n == 0:
            return flow
        ref, other = self.model_ifos[0], self.model_ifos[1]
        shift = {ifo: sh for ifo, sh in zip(self.ifos, to_shift)}
        sh = numpy.array(slide * step, dtype=numpy.float64, ndmin=1)
        if len(sh) == 1:
            sh = numpy.repeat(sh, n)
        dt = (stat[ref]['end_time'] + sh * shift[ref]) \
            - (stat[other]['end_time'] + sh * shift[other])
        # SNR^-4 population and reference-SNR Jacobian, as in logsignalrate
        snr_ref = numpy.sqrt(2. * numpy.maximum(
            numpy.array(stat[ref]['marg_lnl'], dtype=numpy.float64,
                        ndmin=1), 1e-30))
        snr_terms = -numpy.log(snr_ref) - 4. * numpy.log(snr_ref / self.ref_snr)
        tids = numpy.array(stat[ref]['template_id'], ndmin=1)
        ex = numpy.full(n, numpy.nan)
        ln_twin2pi = numpy.log(noise_twindow * 2 * numpy.pi)
        for tid in numpy.unique(tids):
            m = tids == tid
            ex[m] = self.exact({ifo: stat[ifo][m] for ifo in self.model_ifos},
                               dt[m], int(tid), ln_twin2pi)
        loglr = base + snr_terms + ex
        # recompute the louder quiet coincs with the loud settings
        snr2 = sum(2. * numpy.maximum(numpy.asarray(stat[i]['marg_lnl'],
                                                    dtype=numpy.float64), 0.)
                   for i in self.model_ifos)
        redo = numpy.isfinite(loglr) & (loglr > self.exact.refine_above) & \
            (snr2 < self.exact.quiet_snr ** 2) & \
            (self.exact.method == 'grid')
        for tid in numpy.unique(tids[redo]):
            m = redo & (tids == tid)
            ex[m] = self.exact({ifo: stat[ifo][m] for ifo in self.model_ifos},
                               dt[m], int(tid), ln_twin2pi, force_loud=True)
            self.exact.n_refined += int(m.sum())
        loglr = base + snr_terms + ex
        bad = ~numpy.isfinite(loglr)
        if bad.any():
            if not self.compute_flow:
                flow[bad] = flow_stat(bad)
                self.last_flow = flow.copy()
            loglr[bad] = flow[bad]
        self.n_exact += int((~bad).sum())
        self.n_fallback += int(bad.sum())
        if self.check_bound:
            ub = self.coinc_upper_bound(s, slide, step, to_shift, **kwargs)
            g = ~bad
            self.bound_checked += int(g.sum())
            over = g & (loglr > ub + 1e-6)
            self.bound_viol += int(over.sum())
            self.bound_excess += list((loglr - ub)[g])
        loglr[loglr < -30.] = -30.
        return loglr

    def coinc_lim_for_thresh(self, s, thresh, limifo,
                             **kwargs): # pylint:disable=unused-argument
        """
        For each trigger of the fixed ifo, the largest limifo single
        statistic (log noise rate) with which a coinc could still reach
        thresh: coinc_upper_bound at its most favourable over the limifo
        triggers of this template (their SNR range, recorded in single()).
        """
        n = len(s[0][1]['snglstat'])
        if thresh <= -30 or self.check_bound or limifo not in self.rho_range:
            return numpy.ones(n) * numpy.inf
        fixed = s[0][0]
        fs = s[0][1]
        rmin, rmax = self.rho_range[limifo]
        sngl_rates = {fixed: fs['snglstat']}
        sngl_rates[limifo] = numpy.zeros(n)
        ln_noise_rate = coinc_rate.combination_noise_lograte(
                                  sngl_rates, kwargs['time_addition'])
        ln_noise_rate -= self.benchmark_lograte
        network_logvol = 1.5 * numpy.log(numpy.ones(n) * kwargs['max_sigmasq'])
        network_logvol -= fs['benchmark_logvol']
        rf = fs['exact_rho'].astype(numpy.float64)
        cf = fs['exact_c'].astype(numpy.float64)
        ref = self.model_ifos[0]
        if limifo == ref:
            # bound terms grow as the ref SNR falls (rho^-4) and as either
            # SNR rises (time density): take each at its best
            terms = self._exact_bound_terms(numpy.ones(n) * rmax, rf,
                                            numpy.inf, cf,
                                            kwargs['time_addition'])
            terms += 4. * numpy.log(rmax / self.ref_snr) \
                - 4. * numpy.log(rmin / self.ref_snr)
        else:
            terms = self._exact_bound_terms(rf, numpy.ones(n) * rmax,
                                            cf, numpy.inf,
                                            kwargs['time_addition'])
        lim = - thresh + network_logvol - ln_noise_rate + terms
        return numpy.where(numpy.isfinite(lim), lim, numpy.inf)


statistic_dict = {
    "quadsum": QuadratureSumStatistic,
    "single_ranking_only": QuadratureSumStatistic,
    "phasetd": PhaseTDStatistic,
    "exp_fit_csnr": ExpFitCombinedSNR,
    "exp_fit": ExpFitStatistic,
    "phasetd_exp_fit_fgbg_norm_tha": ExpFitFgBgNormTHAStatistic,
    "phasetd_exp_fit_fgbg_norm_marg": ExpFitFgBgNormMARGStatistic,
    "phasetd_exp_fit_fgbg_norm_marg_exact": ExpFitFgBgNormMARGExactStatistic,
}


def get_statistic(stat):
    """
    Error-handling sugar around dict lookup for coincident statistics

    Parameters
    ----------
    stat : string
        Name of the coincident statistic

    Returns
    -------
    class
        Subclass of Stat base class

    Raises
    ------
    RuntimeError
        If the string is not recognized as corresponding to a Stat subclass
    """
    try:
        return statistic_dict[stat]
    except KeyError:
        raise RuntimeError("%s is not an available detection statistic" % stat)


def insert_statistic_option_group(parser, default_ranking_statistic=None):
    """
    Add ranking statistic options to the optparser object.

    Adds the options used to initialize a PyCBC Stat class.

    Parameters
    -----------
    parser : object
        OptionParser instance.
    default_ranking_statisic : str
        Allows setting a default statistic for the '--ranking-statistic'
        option. The option is no longer required if a default is provided.

    Returns
    --------
    strain_opt_group : optparser.argument_group
        The argument group that is added to the parser.
    """
    statistic_opt_group = parser.add_argument_group(
        "Options needed to initialize a PyCBC Stat class for computing the "
        "ranking of events from a PyCBC search."
    )

    statistic_opt_group.add_argument(
        "--ranking-statistic",
        default=default_ranking_statistic,
        choices=statistic_dict.keys(),
        required=True if default_ranking_statistic is None else False,
        help="The coinc ranking statistic to calculate",
    )

    statistic_opt_group.add_argument(
        "--sngl-ranking",
        choices=ranking.sngls_ranking_function_dict.keys(),
        required=True,
        help="The single-detector trigger ranking to use.",
    )

    statistic_opt_group.add_argument(
        "--statistic-files",
        nargs="*",
        action="append",
        default=[],
        help="Files containing ranking statistic info",
    )

    statistic_opt_group.add_argument(
        "--statistic-keywords",
        nargs="*",
        default=[],
        help="Provide additional key-word arguments to be sent to "
        "the statistic class when it is initialized. Should "
        "be given in format --statistic-keywords "
        "KWARG1:VALUE1 KWARG2:VALUE2 KWARG3:VALUE3 ...",
    )

    statistic_opt_group.add_argument(
        "--statistic-features",
        nargs="*",
        default=[],
        choices=_allowed_statistic_features,
        help="Provide additional arguments to include particular "
        "features in the ranking statistic.",
    )

    return statistic_opt_group


def parse_statistic_feature_options(stat_features, stat_kwarg_list):
    """
    Parse the list of statistic keywords into an appropriate dictionary.

    Take input from the input argument ["KWARG1:VALUE1", "KWARG2:VALUE2",
    "KWARG3:VALUE3"] and convert into a dictionary.

    Parameters
    ----------
    stat_kwarg_list : list
        Statistic keywords in list format

    Returns
    -------
    stat_kwarg_dict : dict
        Statistic keywords in dict format
    """
    stat_kwarg_dict = {}

    # Check that the statistic keywords are allowed
    for feature in stat_features:
        if feature not in _allowed_statistic_features:
            err_msg = f"--statistic-feature {feature} not recognised"
            raise NotImplementedError(err_msg)

    # Set values for each feature key to a boolean of whether we want them
    for feature in _allowed_statistic_features:
        stat_kwarg_dict[feature] = feature in stat_features

    for inputstr in stat_kwarg_list:
        try:
            key, value = inputstr.split(":")
            stat_kwarg_dict[key] = value
        except ValueError:
            err_txt = (
                "--statistic-keywords must take input in the "
                "form KWARG1:VALUE1 KWARG2:VALUE2 KWARG3:VALUE3 ... "
                "Received {}".format(" ".join(stat_kwarg_list))
            )
            raise ValueError(err_txt)

    return stat_kwarg_dict


def get_statistic_from_opts(opts, ifos):
    """
    Return a Stat class from an optparser object.

    This will assume that the options in the statistic_opt_group are present
    and will use these options to call stat.get_statistic and initialize the
    appropriate Stat subclass with appropriate kwargs.

    Parameters
    ----------
    opts : optparse.OptParser instance
        The command line options
    ifos : list
        The list of detector names

    Returns
    -------
    class
        Subclass of Stat base class
    """
    # Allow None inputs
    if opts.statistic_files is None:
        opts.statistic_files = []
    if opts.statistic_keywords is None:
        opts.statistic_keywords = []

    # flatten the list of lists of filenames to a single list (may be empty)
    # if needed (e.g. not calling get_statistic_from_opts in a loop)
    if len(opts.statistic_files) > 0 and \
            isinstance(opts.statistic_files[0], list):
        opts.statistic_files = sum(opts.statistic_files, [])

    extra_kwargs = parse_statistic_feature_options(
        opts.statistic_features,
        opts.statistic_keywords,
    )

    stat_class = get_statistic(opts.ranking_statistic)(
        opts.sngl_ranking, opts.statistic_files, ifos=ifos, **extra_kwargs
    )

    return stat_class