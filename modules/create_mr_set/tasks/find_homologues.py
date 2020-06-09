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


## FIND HOMOLOGUES

def search_for_homologues(chains):
  todo = [c for c in chains.values() if not os.path.exists(c.path("gesamt.txt"))]
  if len(todo) == 0: return
  progress_bar = utils.ProgressBar("Performing gesamt searches", len(todo))
  for chain in todo:
    xyzin = chain.structure.path("refmac.pdb")
    prefix = chain.path("gesamt")
    result = tasks.structural_homologues(xyzin, chain.id, prefix, args.gesamt_archive, args.jobs)
    chain.jobs["gesamt"] = result
    progress_bar.increment()
  progress_bar.finish()

class Hit:
  def __init__(self, line):
    split = line.split()
    self.pdb = split[1]
    self.chain = split[2]
    self.qscore = float(split[3])
    self.rmsd = float(split[4])
    self.seqid = float(split[5])
    self.mmcluster = rcsb.cluster_number(self.pdb, self.chain, args.model_model_max_seqid)
    self.mtcluster = rcsb.cluster_number(self.pdb, self.chain, args.model_target_max_seqid)

def filtered_gesamt_hits(chain):
  target_cluster = rcsb.cluster_number(chain.structure.id, chain.id, args.model_target_max_seqid)
  with open(chain.path("gesamt.txt")) as f:
    for line in f:
      if line[0] == "#": continue
      hit = Hit(line)
      if hit.mtcluster is None or hit.mtcluster == target_cluster: continue
      if (hit.qscore > args.model_target_min_qscore and
          hit.rmsd < args.model_target_max_rmsd and
          hit.seqid < (args.model_target_max_seqid / 100.0)):
        yield hit

def superpose_result(hit1, hit2):
  tmp_prefix = "tmp%s" % uuid.uuid4()
  path1 = gzipped_coords(hit1.pdb)
  path2 = gzipped_coords(hit2.pdb)
  result = tasks.superpose(path1, hit1.chain, path2, hit2.chain, tmp_prefix)
  utils.remove_files_starting_with(tmp_prefix)
  return result

def should_choose(hit, chosen_hits):
  if hit.mmcluster is None: return False
  for chosen_hit in reversed(chosen_hits):
    if hit.mmcluster == chosen_hit.mmcluster: return False
  for chosen_hit in reversed(chosen_hits):
    result = superpose_result(hit, chosen_hit)
    if "error" in result: return False
    if result["rmsd"] < args.model_model_min_rmsd: return False
    if result["seqid"] > (args.model_model_max_seqid / 100.0):return False
  return True

def choose_hits(key, chain):
  if os.path.exists(chain.path("homologues")):
    for hid in os.listdir(chain.path("homologues")):
      split = hid.split("_")
      models.Homologue(split[0], split[1], chain)
  else:
    chosen_hits = []
    for hit in filtered_gesamt_hits(chain):
      if should_choose(hit, chosen_hits):
        chosen_hits.append(hit)
        if len(chosen_hits) == args.num_models:
          break
    for hit in chosen_hits:
      models.Homologue(hit.pdb, hit.chain, chain)
  return key, chain

def find_homologues(chains):
  utils.print_section_title("Finding Homologues to Make MR Models From")
  search_for_homologues(chains)
  utils.parallel("Choosing from gesamt results", choose_hits, chains, args.jobs)
