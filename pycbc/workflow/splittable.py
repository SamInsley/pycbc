# Copyright (C) 2013  Ian Harry
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
This module is responsible for setting up the splitting output files stage of
workflows. For details about this module and its capabilities see here:
https://ldas-jobs.ligo.caltech.edu/~cbc/docs/pycbc/NOTYETCREATED.html
"""

import os
import logging

from pycbc.workflow.core import Executable, File, FileList, Node
from pycbc.workflow.core import resolve_url_to_file
from pycbc.workflow.jobsetup import (PycbcSplitBankExecutable,
        PycbcSplitBankXmlExecutable, PycbcSplitInspinjExecutable,
        PycbcHDFSplitInjExecutable)

logger = logging.getLogger('pycbc.workflow.splittable')

def select_splitfilejob_instance(curr_exe):
    """
    This function returns an instance of the class that is appropriate for
    splitting an output file up within workflow (for e.g. splitbank).

    Parameters
    ----------
    curr_exe : string
        The name of the Executable that is being used.
    curr_section : string
        The name of the section storing options for this executble

    Returns
    --------
    exe class : sub-class of pycbc.workflow.core.Executable
        The class that holds the utility functions appropriate
        for the given Executable. This class **must** contain
        * exe_class.create_job()
        and the job returned by this **must** contain
        * job.create_node()
    """
    if curr_exe == 'pycbc_hdf5_splitbank':
        exe_class = PycbcSplitBankExecutable
    elif curr_exe == 'pycbc_splitbank':
        exe_class = PycbcSplitBankXmlExecutable
    elif curr_exe == 'pycbc_split_inspinj':
        exe_class = PycbcSplitInspinjExecutable
    elif curr_exe == 'pycbc_hdf_splitinj':
        exe_class = PycbcHDFSplitInjExecutable
    else:
        # Should we try some sort of default class??
        err_string = "No class exists for Executable %s" %(curr_exe,)
        raise NotImplementedError(err_string)

    return exe_class

def setup_splittable_workflow(workflow, input_tables, out_dir=None, tags=None):
    '''
    This function aims to be the gateway for code that is responsible for taking
    some input file containing some table, and splitting into multiple files
    containing different parts of that table. For now the only supported operation
    is using lalapps_splitbank to split a template bank xml file into multiple
    template bank xml files.

    Parameters
    -----------
    workflow : pycbc.workflow.core.Workflow
        The Workflow instance that the jobs will be added to.
    input_tables : pycbc.workflow.core.FileList
        The input files to be split up.
    out_dir : path
        The directory in which output will be written.

    Returns
    --------
    split_table_outs : pycbc.workflow.core.FileList
        The list of split up files as output from this job.
    '''
    if tags is None:
        tags = []
    logger.info("Entering split output files module.")
    # Parse for options in .ini file
    splitMethod = workflow.cp.get_opt_tags("workflow-splittable",
                                           "splittable-method", tags)

    if splitMethod == "IN_WORKFLOW":
        # Scope here for choosing different options
        logger.info("Adding split output file jobs to workflow.")
        split_table_outs = setup_splittable_dax_generated(workflow,
                input_tables, out_dir, tags)
    elif splitMethod == "NOOP":
        # Probably better not to call the module at all, but this option will
        # return the input file list.
        split_table_outs = input_tables
    else:
        errMsg = "Splittable method not recognized. Must be one of "
        errMsg += "IN_WORKFLOW or NOOP."
        raise ValueError(errMsg)

    logger.info("Leaving split output files module.")
    return split_table_outs

def setup_splittable_dax_generated(workflow, input_tables, out_dir, tags):
    '''
    Function for setting up the splitting jobs as part of the workflow.

    Parameters
    -----------
    workflow : pycbc.workflow.core.Workflow
        The Workflow instance that the jobs will be added to.
    input_tables : pycbc.workflow.core.FileList
        The input files to be split up.
    out_dir : path
        The directory in which output will be written.

    Returns
    --------
    split_table_outs : pycbc.workflow.core.FileList
        The list of split up files as output from this job.
    '''
    cp = workflow.cp

    # Get values from ini file
    try:
        num_splits = cp.get_opt_tags("workflow-splittable",
                                     "splittable-num-banks", tags)
    except BaseException:
        inj_interval = int(cp.get_opt_tags("workflow-splittable",
                                           "splitinjtable-interval", tags))
        if cp.has_option_tags("em_bright_filter", "max-keep", tags) and \
                cp.has_option("workflow-injections", "em-bright-only"):
            num_injs = int(cp.get_opt_tags("em_bright_filter", "max-keep",
                                           tags))
        else:
            # This needed to be changed from num-injs to ninjections in order
            # to work properly with pycbc_create_injections
            num_injs = int(cp.get_opt_tags("workflow-injections",
                                           "ninjections", tags))
        inj_tspace = float(abs(workflow.analysis_time)) / num_injs
        num_splits = int(inj_interval // inj_tspace) + 1

    split_exe_tag = cp.get_opt_tags("workflow-splittable",
                                    "splittable-exe-tag", tags)
    split_exe = os.path.basename(cp.get("executables", split_exe_tag))
    # Select the appropriate class
    exe_class = select_splitfilejob_instance(split_exe)

    # Set up output structure
    out_file_groups = FileList([])

    # Set up the condorJob class for the current executable
    curr_exe_job = exe_class(workflow.cp, split_exe_tag, num_splits,
                             out_dir=out_dir)

    for input in input_tables:
        node = curr_exe_job.create_node(input, tags=tags)
        workflow.add_node(node)
        out_file_groups += node.output_files
        setup_split_marg_grid(workflow, node.output_files, out_dir, tags)
    return out_file_groups


def setup_split_marg_grid(workflow, split_banks, out_dir, tags):
    """Split the THA marginalization grid to match the split banks.

    If [workflow-splittable] (or a tagged subsection) has a
    marginalized-grid-file option, add one job that reads that grid and
    writes one small grid file per split bank, containing only that bank's
    templates. Each split bank File is then given a ``marg_grid_file``
    attribute pointing at its grid file, which the inspiral jobs pick up
    (see PyCBCInspiralExecutable.create_node), so each inspiral job only
    receives its own bank's part of the grid.

    Does nothing if the option is not set.
    """
    cp = workflow.cp
    if not cp.has_option_tags('workflow-splittable',
                              'marginalized-grid-file', tags):
        return
    grid_path = cp.get_opt_tags('workflow-splittable',
                                'marginalized-grid-file', tags)
    logger.info("Splitting marginalization grid %s into %d per-bank files",
                grid_path, len(split_banks))
    grid_file = resolve_url_to_file(grid_path)

    exe = Executable(cp, 'split_marg_grid', ifos=workflow.ifos,
                     out_dir=out_dir, tags=tags)
    node = Node(exe)
    node.add_input_opt('--grid-file', grid_file)
    node.add_input_list_opt('--bank-files', split_banks)
    grid_outs = FileList([])
    for bank in split_banks:
        grid_outs.append(File(bank.ifo_list,
                              bank.description + '_MARG_GRID',
                              bank.segment, extension='.hdf',
                              directory=out_dir, tags=bank.tags,
                              store_file=exe.retain_files))
    node.add_output_list_opt('--output-files', grid_outs)
    workflow.add_node(node)

    for bank, grid in zip(split_banks, grid_outs):
        bank.marg_grid_file = grid

