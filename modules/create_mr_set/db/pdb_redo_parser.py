#!/bin/env python3
from modules.create_mr_set.utils.utils import ProgressBar
import pandas as pd

class PDBRedo(object):
  '''
  A class to parse a pdb file
  '''
  def __init__(self, handle):
    '''
    Initialise the class with the handle
    '''
    self.handle = handle

  def add_entry(self, filename):
    '''
    Add the pdb entry to the database
    '''
    cur = self.handle.cursor()
    
    with open(filename, "r") as data_file:
      data = data_file.read().split("\n")
      #progress_bar = ProgressBar("Adding PDB_redo stats", len(data))
      print(len(data))
      #progress_bar.increment()
      for sample in data:
        structure_id = sample.split()[0].upper()
        rwork_deposited = sample.split()[2]
        rfree_deposited = sample.split()[3]
        rwork_tls = sample.split()[9]
        rfree_tls = sample.split()[10]
        rwork_final = sample.split()[14]
        rfree_final = sample.split()[15]
        completeness = sample.split()[-21]
    
        cur.executescript( '''
          INSERT OR IGNORE INTO pdb_id
          (pdb_id) VALUES ("%s");
          '''% (structure_id))
    
        cur.executescript( '''
          INSERT OR IGNORE INTO pdb_redo_stats (pdb_id_id)
          SELECT id FROM pdb_id
          WHERE pdb_id.pdb_id="%s";
          ''' % (structure_id))
    
        pdb_redo_dict = {
            "rWork_depo"         : rwork_deposited,
            "rFree_depo"         : rfree_deposited,
            "rWork_tls"          : rwork_tls,
            "rFree_tls"          : rfree_tls,
            "rWork_final"        : rwork_final,
            "rFree_final"        : rfree_final,
            "completeness"       : completeness
                    }
    
        cur.execute('''
          SELECT id FROM pdb_id
          WHERE pdb_id="%s"
          ''' % (structure_id))
        pdb_id = cur.fetchone()[0]
        
        for data in pdb_redo_dict:
          cur.execute('''
            UPDATE pdb_redo_stats
            SET "%s" = "%s"
            WHERE pdb_id_id = "%s";
            ''' % (data, pdb_redo_dict[data], pdb_id))
        self.handle.commit()
      #progress_bar.finish()

