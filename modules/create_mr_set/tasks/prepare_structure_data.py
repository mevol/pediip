#!/usr/bin/env python3

import argparse
import Bio.Seq
import Bio.SeqIO
import Bio.SeqRecord
import datetime
import gemmi
import glob
import gzip
import modules.create_mr_set.models as models
import os
import modules.create_mr_set.pdbtools as pdbtools
import random
import modules.create_mr_set.rcsb as rcsb
import sys
import modules.create_mr_set.tasks as tasks
import urllib.request
import modules.create_mr_set.utils as utils
import uuid
import xml.etree.ElementTree as ET


## PREPARE STRUCTURE DATA

def unzip_input_files(key, structure):
  utils.run("gunzip", ["-c", gzipped_coords(structure.id)], stdout=structure.path("deposited.pdb"))
  utils.run("gunzip", ["-c", gzipped_sfs(structure.id)], stdout=structure.path("deposited.cif"))
  structure.add_metadata("semet", utils.is_semet(structure.path("deposited.pdb")))
  return key, structure

def convert_to_mtz(key, structure):
  hklin = structure.path("deposited.cif")
  prefix = structure.path("cif2mtz")
  result = tasks.cif2mtz(hklin, prefix)
  structure.jobs["cif2mtz"] = result
  return key, structure

def convert_amplitudes(key, structure):
  hklin = structure.jobs["cif2mtz"]["hklout"]
  seqin = structure.path("deposited.fasta")
  prefix = structure.path("ctruncate")
  result = tasks.convert_amplitudes(hklin, seqin, prefix)
  structure.jobs["ctruncate"] = result
  return key, structure

def add_freer_flag(key, structure):
  hklin = structure.jobs["ctruncate"]["hklout"]
  prefix = structure.path("freerflag")
  result = tasks.add_freer_flag(hklin, prefix)
  structure.jobs["freerflag"] = result
  return key, structure

def rename_columns(key, structure):
  hklin = structure.jobs["freerflag"]["hklout"]
  colin = ["FreeR_flag"] + structure.jobs["ctruncate"]["colout"]
  colout = ["FREE", "FP", "SIGFP"]
  prefix = structure.path("cad")
  result = tasks.select_and_rename_columns(hklin, colin, colout, prefix)
  structure.jobs["cad"] = result
  return key, structure

def remove_unl_residues(key, structure):
  xyzin = structure.path("deposited.pdb")
  prefix = structure.path("no_unl")
  result = tasks.remove_unl_residues(xyzin, prefix)
  structure.jobs["no_unl"] = result
  return key, structure

def refine_deposited_structure(key, structure):
  hklin = structure.jobs["cad"]["hklout"]
  xyzin = structure.jobs["no_unl"]["xyzout"]
  prefix = structure.path("refmac")
  result = tasks.refine(hklin, xyzin, prefix)
  structure.jobs["refmac"] = result
  if "Refmac:  End of Refmac" in result:
#  if "error" not in result:
    mtz = gemmi.read_mtz_file(result["hklout"])
    structure.add_metadata("spacegroup", mtz.spacegroup.hm)
    structure.add_metadata("resolution", round(mtz.resolution_high(), 2))
    structure.add_metadata("asu_volume", round(mtz.cell.volume / mtz.nsymop))
    structure.add_metadata("data_completeness", result["data_completeness"])
    structure.add_metadata("refined_rwork", result["final_rwork"])
    structure.add_metadata("refined_rfree", result["final_rfree"])
#  else:
#    print("No results found for structure")
  return key, structure

def prepare_structure_data(structures):
  utils.print_section_title("Preparing Structure Data")
  steps = [
    ("Unzipping input files", unzip_input_files),
    ("Converting CIF files to MTZ", convert_to_mtz),
    ("Converting amplitudes", convert_amplitudes),
    ("Adding free-R flags", add_freer_flag),
    ("Renaming columns", rename_columns),
    ("Removing UNL residues", remove_unl_residues),
    ("Refining structures", refine_deposited_structure),
    #added new fucntion
    #("Removing structures with errors", remove_errors),
  ]
  for title, func in steps:
    utils.parallel(title, func, structures, args.jobs)
    utils.remove_errors(structures)
  print("")

def remove_structures(structures):
  reason_count = {}
  def remove(structure, reason):
    if reason not in reason_count: reason_count[reason] = 0
    reason_count[reason] += 1
    structure.add_metadata("error", reason)
    del structures[structure.id]
  
  
  for structure in list(structures.values()):
    print(structure)
    print(metadata)
    if structure.metadata.get("refined_rwork") == False:
      print("structure was not refined")
      
    else:
      rwork_diff = structure.metadata["refined_rwork"] - structure.metadata["reported_rwork"]
      if rwork_diff > args.tolerance_rwork:
        remove(structure, "Refined R-work is >%d higher than reported R-work" % args.tolerance_rwork)




#    if ["refined_rwork"] not in structure.metadata:
#      #remove(structure, "refined_rwork not in metadata")
#      structure.metadata.get("refined_rwork", 0)
#      pass
#    else:
#      rwork_diff = structure.metadata["refined_rwork"] - structure.metadata["reported_rwork"]
#      if rwork_diff > args.tolerance_rwork:
#        remove(structure, "Refined R-work is >%d higher than reported R-work" % args.tolerance_rwork)
  for structure in list(structures.values()):
    if structure.metadata["data_completeness"] < args.tolerance_completeness:
      remove(structure, "Data completeness is below %d" % args.tolerance_completeness)
  if len(reason_count) > 0:
    print("Removed some structures:")
    for reason in reason_count:
      print("%s (%d removed)" % (reason, reason_count[reason]))
    print("")
  if len(structures) < 1:
    sys.exit("No structures left!")
