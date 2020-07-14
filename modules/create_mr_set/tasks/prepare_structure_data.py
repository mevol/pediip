#!/usr/bin/env python3

import argparse
import datetime
import glob
import os
import sys
import modules.create_mr_set.utils.utils as utils

## PREPARE STRUCTURE DATA

def path_coords(pdb_id, args):
  pdb = pdb_id.lower()
  return os.path.join(args.pdb_coords, pdb[1:3], "%s_final.pdb" % pdb)

def path_sfs(pdb_id, args):
  pdb = pdb_id.lower()
  return os.path.join(args.pdb_sfs, pdb[1:3], "%s_final.mtz" % pdb)

def prepare_structure_data(structures, args):
  utils.print_section_title("Preparing Structure Data")
  title = "Creating symlinks for files"
  progress_bar = utils.ProgressBar(title, len(structures))
  for structure in structures:
    progress_bar.increment()
    coords = args.pdb_coords
    sfs = args.pdb_sfs
    coords_source = path_coords(structure, args)
    sfs_source = path_sfs(structure, args)
    target_dir = os.path.join("structures", structure)
#    coords_target = os.path.join(target_dir, "%s_final.pdb" % structure)
#    sfs_target = os.path.join(target_dir, "%s_final.mtz" % structure)
    coords_target = os.path.join(target_dir, "refmac.pdb")
    sfs_target = os.path.join(target_dir, "refmac.mtz")
    os.symlink(coords_source, coords_target)
    os.symlink(sfs_source, sfs_target)
  print("")
  print("")
