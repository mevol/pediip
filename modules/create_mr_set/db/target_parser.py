#!/bin/env python3

import os
import json
import argparse
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
        
    target_dict = {
          "number_of_copies"     : int(number_of_copies),
          "sequence_length"      : int(sequence_length),
          "sequence"             : str(sequence),
          "number_of_homologues" : int(number_of_homologues)
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


