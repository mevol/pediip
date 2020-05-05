#!/usr/bin/env python3

import collections
import json
import numpy as np
import os
import random
import shutil

def get_structures():
  structures = {}
  structures_dir = "structures"
  if os.path.exists(structures_dir):
    for sid in os.listdir(structures_dir):
      structure_dir = os.path.join(structures_dir, sid)
      metadata_path = os.path.join(structure_dir, "metadata.json")
      if os.path.exists(metadata_path):
        with open(metadata_path) as f:
          structures[sid] = json.load(f)

        structures[sid]["chains"] = {}
        chains_dir = os.path.join(structure_dir, "chains")
        if os.path.exists(chains_dir):
          for cid in os.listdir(chains_dir):
            chain_dir = os.path.join(chains_dir, cid)
            metadata_path = os.path.join(chain_dir, "metadata.json")
            if os.path.exists(metadata_path):
              with open(metadata_path) as f:
                structures[sid]["chains"][cid] = json.load(f)

              structures[sid]["chains"][cid]["homologues"] = {}
              homologues_dir = os.path.join(chain_dir, "homologues")
              if os.path.exists(homologues_dir):
                for hid in os.listdir(homologues_dir):
                  homologue_dir = os.path.join(homologues_dir, hid)
                  metadata_path = os.path.join(homologue_dir, "metadata.json")
                  if os.path.exists(metadata_path):
                    with open(metadata_path) as f:
                      structures[sid]["chains"][cid]["homologues"][hid] = json.load(f)
  return structures

def pick_reduced(structures, min_fmap, max_fmap, step, worst_res):
  reduced = []
  class Bin:
    def __init__(self, min_fmap, max_fmap):
      self.min = min_fmap
      self.max = max_fmap
      self.count = 0
  bins = [Bin(x, x + step) for x in np.linspace(min_fmap, max_fmap - step, (max_fmap - min_fmap) / step)]
  for sid, structure in structures.items():
    if worst_res is not None:
      if "resolution" not in structure: continue
      if structure["resolution"] > worst_res: continue
    models = []
    for cid, chain in structure["chains"].items():
      for hid, homologue in chain["homologues"].items():
        if "f_map_correlation" in homologue and all(homologue[k] is not None for k in homologue):
          models.append((cid, hid))
    bins.sort(key=lambda b: b.count)
    random.shuffle(models)
    for b in bins:
      found = False
      for cid, hid in models:
        f_map_correlation = structure["chains"][cid]["homologues"][hid]["f_map_correlation"]
        if f_map_correlation > b.min and f_map_correlation <= b.max:
          found = True
          reduced.append((sid, cid, hid))
          b.count += 1
          break
      if found:
        break
  bins.sort(key=lambda b: b.min)
  for b in bins:
    print ("Bin %.2f %.2f: %4d structures" % (b.min, b.max, b.count))
  return reduced

def write_files(dirname, sid, cid, hid):
  sdir = os.path.join("structures", sid)
  cdir = os.path.join(sdir, "chains", cid)
  hdir = os.path.join(cdir, "homologues", hid)
  new_dir = os.path.join(dirname, sid)
  os.mkdir(new_dir)

  with open(os.path.join(sdir, "metadata.json")) as f:
    structure = json.load(f)
  with open(os.path.join(cdir, "metadata.json")) as f:
    chain = json.load(f)
  with open(os.path.join(hdir, "metadata.json")) as f:
    metadata = json.load(f)

  metadata.update({
    "asu_volume": structure["asu_volume"],
    "data_completeness": structure["data_completeness"],
    "data_resolution": structure["resolution"],
    "model_chain": hid.split("_")[1],
    "model_copies": chain["copies"],
    "model_pdb": hid.split("_")[0],
    "reference_rfree": structure["refined_rfree"],
    "reference_rwork": structure["refined_rwork"],
    "reported_resolution": structure["reported_resolution"],
    "reported_rfree": structure["reported_rfree"],
    "reported_rwork": structure["reported_rwork"],
    "semet": structure["semet"],
    "spacegroup": structure["spacegroup"],
    "target_chain": cid,
    "target_pdb": sid,
  })

  with open(os.path.join(new_dir, "metadata.json"), "w") as f:
    json.dump(metadata, f, indent=2, sort_keys=True)

  src = os.path.join(sdir, "refmac.pdb")
  dst = os.path.join(new_dir, "reference.pdb")
  shutil.copy(src, dst)
  src = os.path.join(sdir, "unique.fasta")
  dst = os.path.join(new_dir, "sequence.fasta")
  shutil.copy(src, dst)
  src = os.path.join(hdir, "refmac.pdb")
  dst = os.path.join(new_dir, "model.pdb")
  shutil.copy(src, dst)
  src = os.path.join(hdir, "cmtzjoin.mtz")
  dst = os.path.join(new_dir, "reflections.mtz")
  shutil.copy(src, dst)

def reduce_test_set(structures, dirname, min_fmap, max_fmap, step, worst_res=None):
  if os.path.exists(dirname): return
  os.mkdir(dirname)
  reduced = pick_reduced(structures, min_fmap, max_fmap, step, worst_res)
  for sid, cid, hid in reduced:
    write_files(dirname, sid, cid, hid)

if __name__ == "__main__":
  structures = get_structures()
  reduce_test_set(structures, "reduced_full", 0.2, 0.9, 0.1)
  reduce_test_set(structures, "reduced_easy", 0.7, 0.95, 0.05, worst_res=2.5)
