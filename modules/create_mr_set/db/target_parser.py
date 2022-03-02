#!/bin/env python3

import os
import json
import argparse
import gzip
import pandas as pd
from itertools import islice

class TargetParser(object):
  '''
  A class to parse a pdb file
  '''
  def __init__(self, handle):
    '''
    Initialise the class with the handle
    '''
    self.handle = handle

  def add_entry(self, structure, results_dir, local_pdb):
    '''
    Add the pdb entry to the database
    '''
    cur = self.handle.cursor()
    
    cur.executescript( '''
      INSERT OR IGNORE INTO pdb_id
      (pdb_id) VALUES ("%s");
      '''% (structure))

    cur.executescript( '''
      INSERT OR IGNORE INTO target_stats (pdb_id_id)
      SELECT id FROM pdb_id
      WHERE pdb_id.pdb_id="%s";
      ''' % (structure))

    structure_dir = os.path.join(results_dir, "structures")
    structure_location = os.path.join(structure_dir, structure)
    chain_location = os.path.join(structure_location, "chains")
    print("CHAIN LOCATION", chain_location)
    chain_name = os.listdir(chain_location)
    chain_details = os.path.join(chain_location, chain_name[0])
    chain_meta_json = os.path.join(chain_details, "metadata.json")
    with open(chain_meta_json, "r") as chain_json:
      chain_reader = json.load(chain_json)    

    number_of_copies = chain_reader["copies"]
    sequence_length = chain_reader["length"]
    sequence = chain_reader["seq"]

    homologue_dir = os.path.join(chain_details, "homologues")
    
    number_of_homologues = 0
    
    if os.path.exists(homologue_dir):
      homologue_lst = os.listdir(homologue_dir)
      number_of_homologues = len(homologue_lst)


    # getting some additional stats from the mmcif header for a given current target PDB ID
    lower = structure.lower()
    if lower == '3wcx':
      lower = '7cne'
    centre = lower[1:-1]
    pdb_devided_folder = os.path.join(local_pdb, centre)
    os.chdir(pdb_devided_folder)
    try:
      os.path.isfile(str(lower)+'.cif.gz')
    except FileNotFoundError:
      print("Could not find mmcif file for PDB ID")
    else:
      f=gzip.open(str(lower)+'.cif.gz','rt')
      for line in f.readlines():
        if '_reflns.d_resolution_high' in line:
            split_1 = line.split(' ')
            reso = split_1[-2]
            if reso == '?':
                reso = 0
        if '_reflns.pdbx_netI_over_sigmaI' in line:
            split_2 = line.split(' ')
            isigma = split_2[-2]
            if isigma == '?':
                isigma = 0 
        if '_reflns.pdbx_redundancy' in line:
            split_3 = line.split(' ')
            multi = split_3[-2]
            if multi == '?':
                multi = 0
        if '_reflns.percent_possible_obs' in line:
            split_4 = line.split(' ')
            compl = split_4[-2]
            if compl == '?':
                compl = 0

    target_dict = {
          "number_of_copies"     : int(number_of_copies),
          "sequence_length"      : int(sequence_length),
          "sequence"             : str(sequence),
          "number_of_homologues" : int(number_of_homologues),
          "resolution"           : reso,
          "multiplicity"         : multi,
          "Isigma"               : isigma,
          "completeness"         : compl
                  }

    cur.execute('''
      SELECT id FROM pdb_id
      WHERE pdb_id="%s"
      ''' % (structure))
    pdb_id = cur.fetchone()[0]

    for entry in target_dict:
      cur.execute('''
        UPDATE target_stats
        SET "%s" = "%s"
        WHERE pdb_id_id = "%s";
        ''' % (entry, target_dict[entry], pdb_id))
        
    self.handle.commit()  


