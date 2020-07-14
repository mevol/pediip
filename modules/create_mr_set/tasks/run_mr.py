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

## MR

def path_coords(pdb_id, args):
  pdb = pdb_id.lower()
  return os.path.join(args.pdb_coords, pdb[1:3], "%s_final.pdb" % pdb)

def superpose_homologue(key, homologue, args):
  xyzin1 = homologue.chain.structure.path("refmac.pdb")
  chain1 = homologue.chain.id
  xyzin2 = path_coords(homologue.hit_pdb, args)
  chain2 = homologue.hit_chain
  prefix = homologue.path("gesamt")
  result = tasks.superpose(xyzin1, chain1, xyzin2, chain2, prefix)
  homologue.jobs["gesamt"] = result
  if "ERROR" not in result and "qscore" in result:
    homologue.add_metadata("gesamt_qscore", result["qscore"])
    homologue.add_metadata("gesamt_rmsd", result["rmsd"])
    homologue.add_metadata("gesamt_length", result["length"])
    homologue.add_metadata("gesamt_seqid", result["seqid"])
  return key, homologue

def prepare_sculptor_alignment(key, homologue, args):
  if os.path.exists(homologue.path("gesamt.seq")):
    seqin = homologue.path("gesamt.seq")
    seqout = homologue.path("sculptor.aln")
    records = list(Bio.SeqIO.parse(seqin, "fasta"))
    for record in records:
      record.seq = Bio.Seq.Seq(str(record.seq).upper())
    Bio.SeqIO.write(records, seqout, "clustal")
  return key, homologue

def trim_model(key, homologue, args):
  model = path_coords(homologue.hit_pdb, args)
  chain = homologue.hit_chain
  alignment = homologue.path("sculptor.aln")
  prefix = homologue.path("sculptor")
  result = tasks.trim_model(model, chain, alignment, prefix)
  homologue.jobs["sculptor"] = result
  return key, homologue

def mr(key, homologue, args):
  hklin = homologue.chain.structure.path("refmac.mtz")
  xyzin = glob.glob(homologue.path("sculptor*.pdb"))[0]
  identity = homologue.metadata["gesamt_seqid"]
  prefix = homologue.path("phaser")
  copies = homologue.chain.metadata["copies"]
  atom_counts = pdbtools.count_elements(homologue.chain.structure.path("refmac.pdb"))
  result = tasks.mr(hklin, xyzin, identity, prefix, copies, atom_counts)
  homologue.jobs["phaser"] = result
  if "error" not in result:
    homologue.add_metadata("phaser_llg", result["llg"])
    homologue.add_metadata("phaser_rmsd", result["rmsd"])
  return key, homologue

def refine_placed_model(key, homologue, args):
  hklin = homologue.chain.structure.path("refmac.mtz")
  xyzin = homologue.path("phaser.1.pdb")
  prefix = homologue.path("refmac")
  result = tasks.refine(hklin, xyzin, prefix)
  homologue.jobs["refmac"] = result
  if "error" not in result:
    homologue.add_metadata("final_rfree", result["final_rfree"])
    homologue.add_metadata("final_rwork", result["final_rwork"])
    homologue.add_metadata("initial_rfree", result["initial_rfree"])
    homologue.add_metadata("initial_rwork", result["initial_rwork"])
  return key, homologue

def prepare_and_do_mr(homologues, args):
  print(args)
  print(args.jobs)
  utils.print_section_title("Preparing Models")
  utils.parallel("Superposing homologues for sequence alignments", superpose_homologue, homologues, args, args.jobs)
  utils.parallel("Preparing alignments for sculptor", prepare_sculptor_alignment, homologues, args, args.jobs)
  utils.parallel("Trimming input models with sculptor", trim_model, homologues, args, args.jobs)
  if not args.stop_before_mr:
    utils.print_section_title("Performing Molecular Replacement")
    utils.parallel("Performing molecular replacement with phaser", mr, homologues, args, int(args.jobs / 4))
    utils.parallel("Refining placed models", refine_placed_model, homologues, args, args.jobs)
  utils.remove_errors(homologues)
  print("")
