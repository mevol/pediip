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
    # connecting to the database
    cur = self.handle.cursor()
    
    # adding a PDB ID for a given target structure into the database
    cur.executescript( '''
      INSERT OR IGNORE INTO pdb_id
      (pdb_id) VALUES ("%s");
      '''% (structure))

    # connect the target PDB ID to the target_stats table which will contain details from the target mmcif header
    cur.executescript( '''
      INSERT OR IGNORE INTO target_stats (pdb_id_id)
      SELECT id FROM pdb_id
      WHERE pdb_id.pdb_id="%s";
      ''' % (structure))

    # expanding path to the target structure collection
    structure_dir = os.path.join(results_dir, "structures")
    # expand the path for the current target structure being worked on
    structure_location = os.path.join(structure_dir, structure)
    # expand the path for the chain of the current target structure that is being worked on
    chain_location = os.path.join(structure_location, "chains")
    chain_name = os.listdir(chain_location)
    chain_details = os.path.join(chain_location, chain_name[0])
    # find the metadata.json file for the chain of the current target structure that contains some stats
    chain_meta_json = os.path.join(chain_details, "metadata.json")
    with open(chain_meta_json, "r") as chain_json:
      chain_reader = json.load(chain_json)    

    # extract the stats from metadata.json
    number_of_copies = chain_reader["copies"]
    sequence_length = chain_reader["length"]
    sequence = chain_reader["seq"]

    # get the number of homologues for current target structure
    homologue_dir = os.path.join(chain_details, "homologues")
    
    number_of_homologues = 0
    
    if os.path.exists(homologue_dir):
      homologue_lst = os.listdir(homologue_dir)
      number_of_homologues = len(homologue_lst)


    # getting some additional stats from the mmcif header for a given current target PDB ID
    reso = 0
    isigma = 0
    multi = 0
    compl = 0
    lower = structure.lower()
    if lower == '3wcx':
      lower = '7cne'
    centre = lower[1:-1]
    pdb_devided_folder = os.path.join(local_pdb, centre)
    try:
      os.path.isfile(os.path.join(pdb_devided_folder, str(lower)+'.cif.gz'))
    except FileNotFoundError:
      print("Could not find mmcif file for PDB ID")
    else:
      f=gzip.open(os.path.join(pdb_devided_folder, str(lower)+'.cif.gz'),'rt')
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

    # aggregating all the stats in a target_dict
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

    # writing the stats for the current target structure into the database
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


