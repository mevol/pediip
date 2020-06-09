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

## MR

def superpose_homologue(key, homologue):
  xyzin1 = homologue.chain.structure.path("refmac.pdb")
  chain1 = homologue.chain.id
  xyzin2 = gzipped_coords(homologue.hit_pdb)
  chain2 = homologue.hit_chain
  prefix = homologue.path("gesamt")
  result = tasks.superpose(xyzin1, chain1, xyzin2, chain2, prefix)
  homologue.jobs["gesamt"] = result
  if "error" not in result:
    homologue.add_metadata("gesamt_qscore", result["qscore"])
    homologue.add_metadata("gesamt_rmsd", result["rmsd"])
    homologue.add_metadata("gesamt_length", result["length"])
    homologue.add_metadata("gesamt_seqid", result["seqid"])
  return key, homologue

def prepare_sculptor_alignment(key, homologue):
  seqin = homologue.path("gesamt.seq")
  seqout = homologue.path("sculptor.aln")
  records = list(Bio.SeqIO.parse(seqin, "fasta"))
  for record in records:
    record.seq = Bio.Seq.Seq(str(record.seq).upper())
  Bio.SeqIO.write(records, seqout, "clustal")
  return key, homologue

def trim_model(key, homologue):
  model = gzipped_coords(homologue.hit_pdb)
  chain = homologue.hit_chain
  alignment = homologue.path("sculptor.aln")
  prefix = homologue.path("sculptor")
  result = tasks.trim_model(model, chain, alignment, prefix)
  homologue.jobs["sculptor"] = result
  return key, homologue

def mr(key, homologue):
  hklin = homologue.chain.structure.path("cad.mtz")
  xyzin = glob.glob(homologue.path("sculptor*.pdb"))[0]
  identity = homologue.metadata["gesamt_seqid"]
  prefix = homologue.path("phaser")
  copies = homologue.chain.metadata["copies"]
  atom_counts = pdbtools.count_elements(homologue.chain.structure.path("deposited.pdb"))
  result = tasks.mr(hklin, xyzin, identity, prefix, copies, atom_counts)
  homologue.jobs["phaser"] = result
  if "error" not in result:
    homologue.add_metadata("phaser_llg", result["llg"])
    homologue.add_metadata("phaser_rmsd", result["rmsd"])
  return key, homologue

def refine_placed_model(key, homologue):
  hklin = homologue.chain.structure.path("cad.mtz")
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

def write_combined_mtz(key, homologue):
  prefix = homologue.path("cmtzjoin")
  result = tasks.combine_mtz(prefix, [
    (homologue.chain.structure.path("cad.mtz"), "FP,SIGFP", "FP,SIGFP"),
    (homologue.chain.structure.path("cad.mtz"), "FREE", "FREE"),
    (homologue.chain.structure.path("refmac.mtz"), "HLACOMB,HLBCOMB,HLCCOMB,HLDCOMB", "reference.HLA,reference.HLB,reference.HLC,reference.HLD"),
    (homologue.path("refmac.mtz"), "HLACOMB,HLBCOMB,HLCCOMB,HLDCOMB", "model.HLA,model.HLB,model.HLC,model.HLD"),
  ])
  homologue.jobs["cmtzjoin"] = result
  return key, homologue
