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




## GET SEQUENCES

def download_sequences(structures):
  print("Downloading sequences ...")
  ids = [s.id for s in structures.values()]
  url = "https://www.rcsb.org/pdb/download/downloadFastaFiles.do"
  url += "?structureIdList=%s" % ",".join(ids)
  url += "&compressionType=uncompressed"
  urllib.request.urlretrieve(url, "sequences.fasta")

def extract_sequences(structures):
  print("Extracting sequences ...")
  for record in Bio.SeqIO.parse("sequences.fasta", "fasta"):
    structure_id = record.id[:4]
#    print(structure_id)
    chain_id = record.id[5:6]
#    print(chain_id)
    if structure_id in structures:
      structure = structures[structure_id]
      if chain_id in structure.chains:
        chain = structure.chains[chain_id]
        chain.add_metadata("seq", str(record.seq))
        chain.add_metadata("length", len(chain.metadata["seq"]))

def write_sequence(structure, path):
  records = []
  for chain in structure.chains.values():
    record = Bio.SeqRecord.SeqRecord(Bio.Seq.Seq(chain.metadata["seq"]),
      id="%s:%s" % (structure.id, chain.id), description="")
    records.append(record)
  Bio.SeqIO.write(records, path, "fasta")
  return structure

def write_deposited_sequence(key, structure, args):
  structure = write_sequence(structure, structure.path("deposited.fasta"))
  return key, structure

def remove_duplicate_chains(key, structure, args):
  seq_copies_dict = {}
  for chain_id, chain in sorted(structure.chains.items()):
    if chain.metadata["seq"] not in seq_copies_dict:
      seq_copies_dict[chain.metadata["seq"]] = 1
    else:
      seq_copies_dict[chain.metadata["seq"]] += 1
      del structure.chains[chain_id]
      chain.remove_directory()
  for chain in structure.chains.values():
    chain.add_metadata("copies", seq_copies_dict[chain.metadata["seq"]])
  return key, structure

def write_unique_sequence(key, structure, args):
  structure = write_sequence(structure, structure.path("unique.fasta"))
  return key, structure

def get_sequences(structures, args):
  utils.print_section_title("Getting Full Sequences")
  download_sequences(structures)
  extract_sequences(structures)
  utils.parallel("Writing deposited sequences", write_deposited_sequence, structures, args, args.jobs)
  utils.parallel("Removing duplicate chains", remove_duplicate_chains, structures, args, args.jobs)
  utils.parallel("Writing unique sequences", write_unique_sequence, structures, args, args.jobs)
  print("")
