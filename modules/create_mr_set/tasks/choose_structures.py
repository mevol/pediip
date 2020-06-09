#!/usr/bin/env python3

import argparse
import Bio.Seq
import Bio.SeqIO
import Bio.SeqRecord
import datetime
import gemmi
import glob
import gzip
import modules.create_mr_set.utils.models as models
import os
import modules.create_mr_set.utils.pdbtools as pdbtools
import random
import modules.create_mr_set.utils.rcsb as rcsb
import sys
import modules.create_mr_set.tasks.tasks as tasks
import urllib.request
import modules.create_mr_set.utils.utils as utils
import uuid
import xml.etree.ElementTree as ET


## CHOOSE STRUCTURES

class ResolutionBin:
  def __init__(self, i):
    self.min_res = args.res_min + i * args.res_step
    self.max_res = args.res_min + (i + 1) * args.res_step
    self.structures = []
    self.chosen = []

def assign_resolution_bins(structures):
  bins = [ResolutionBin(i) for i in range(args.res_bins)]
  for structure in structures.values():
    if (structure.resolution < args.res_min or
        structure.resolution >= args.res_max):
      continue
    i = int((structure.resolution - args.res_min) / args.res_step)
    bins[i].structures.append(structure)
  return bins

def gzipped_coords(pdb_id):
  pdb = pdb_id.lower()
  return os.path.join(args.pdb_coords, pdb[1:3], "pdb%s.ent.gz" % pdb)

def gzipped_sfs(pdb_id):
  pdb = pdb_id.lower()
  return os.path.join(args.pdb_sfs, pdb[1:3], "r%ssf.ent.gz" % pdb)

def gzipped_report(pdb_id):
  pdb = pdb_id.lower()
  return os.path.join(args.pdb_reports, pdb[1:3], pdb, "%s_validation.xml.gz" % pdb)

def input_files_exist(structure):
  return all(os.path.exists(path) for path in {
    gzipped_coords(structure.id),
    gzipped_sfs(structure.id),
    gzipped_report(structure.id),
  })

def validation_report_okay(structure):
  attrib_key_dict = {
    "relative-percentile-DCC_Rfree": "validation_rfree",
    "relative-percentile-clashscore": "validation_clash",
    "relative-percentile-percent-RSRZ-outliers": "validation_rsrz",
    "relative-percentile-percent-rama-outliers": "validation_rama",
    "relative-percentile-percent-rota-outliers": "validation_rota",
  }
  path = gzipped_report(structure.id)
  if not os.path.exists(path): return False
  with gzip.open(path) as f:
    content = f.read()
  attribs = ET.fromstring(content).find("Entry").attrib
  for attrib in attrib_key_dict:
    key = attrib_key_dict[attrib]
    if attrib not in attribs: return False
    percentile = float(attribs[attrib])
    threshold = getattr(args, key)
    if percentile < threshold: return False
    setattr(structure, key, percentile)
  return True

def choose_structures(structures):
  utils.print_section_title("Choosing Structures")
  res_bins = assign_resolution_bins(structures)
  chosen_clusters = set()
  cluster_attr = "cluster%d" % args.structure_structure_max_seqid
  res_bins.sort(key=lambda res_bin: len(res_bin.structures))
  for res_bin in res_bins:
    title = "Choosing %.2f-%.2fA structures (%d to choose from)" % (
      res_bin.min_res, res_bin.max_res, len(res_bin.structures))
    progress_bar = utils.ProgressBar(title, args.num_structures)
    random.shuffle(res_bin.structures)
    num_checked = 0
    num_missing_files = 0
    num_too_similar = 0
    num_failed_validation = 0
    for structure in res_bin.structures:
      passed = True
      num_checked += 1
      if not input_files_exist(structure):
        num_missing_files += 1
        passed = False
      clusters = {getattr(c, cluster_attr) for c in structure.chains.values()}
      if any(c in chosen_clusters for c in clusters):
        num_too_similar += 1
        passed = False
      if not validation_report_okay(structure):
        num_failed_validation += 1
        passed = False
      if passed:
        res_bin.chosen.append(models.Structure(structure))
        chosen_clusters.update(clusters)
        progress_bar.increment()
        if len(res_bin.chosen) == args.num_structures:
          break
    progress_bar.finish()
    print("Total number checked:          %6d" % num_checked)
    print("Missing input files:           %6d" % num_missing_files)
    print("Too similar to already chosen: %6d" % num_too_similar)
    print("Failed validation checks:      %6d" % num_failed_validation)
    print("")
  return {s.id: s for r in res_bins for s in r.chosen}
