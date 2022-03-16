#!/bin/env python3

import os
import re
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

  def add_entry(self, structure, local_pdb_redo):
    '''
    Add the pdb entry to the database
    '''
    # connect to the database
    cur = self.handle.cursor()

    # check whether the given PDB-redo summary stats file exists
    try:
      os.path.exists(os.path.join(local_pdb_redo, "others/alldata.txt"))
      filename = os.path.join(local_pdb_redo, "others/alldata.txt")
    except:
      print("No data file found for PDB-redo")
    pass

    # set default value for the different stats
    rwork_deposited = 0
    rfree_deposited = 0
    rwork_tls = 0
    rfree_tls = 0
    rwork_final = 0
    rfree_final = 0
#    completeness = 0

    # open the data file if it exists
    with open(filename, "r") as data_file:
      # find the line staring with lower case structure PDB ID
      line = next((l for l in data_file if structure.lower() in l), None)
      # split the line and pick the corresponding values based on column names declared
      # in the file header which is being ignored here
      split = line.split()
      rwork_deposited = split[2]
      rfree_deposited = split[3]
      rwork_tls = split[9]
      rfree_tls = split[10]
      rwork_final = split[14]
      rfree_final = split[15]
#      completeness = split[-21]#-21

      # find the relevant structure entry in the database
      cur.executescript( '''
          INSERT OR IGNORE INTO pdb_id
          (pdb_id) VALUES ("%s");
          '''% (structure))

      # link the structure entry to the pdb-redo table
      cur.executescript( '''
          INSERT OR IGNORE INTO pdb_redo_stats (pdb_id_id)
          SELECT id FROM pdb_id
          WHERE pdb_id.pdb_id="%s";
          ''' % (structure))

      # dictionary with PDV-redo stats
      pdb_redo_dict = {
            "rWork_depo"         : rwork_deposited,
            "rFree_depo"         : rfree_deposited,
            "rWork_tls"          : rwork_tls,
            "rFree_tls"          : rfree_tls,
            "rWork_final"        : rwork_final,
            "rFree_final"        : rfree_final,
#            "completeness"       : completeness
                    }

      # entering the PDB-redo stats into the corresponding table in the database
      cur.execute('''
          SELECT id FROM pdb_id
          WHERE pdb_id="%s"
          ''' % (structure))
      pdb_id = cur.fetchone()[0]
        
      for data in pdb_redo_dict:
        cur.execute('''
            UPDATE pdb_redo_stats
            SET "%s" = "%s"
            WHERE pdb_id_id = "%s";
            ''' % (data, pdb_redo_dict[data], pdb_id))
      self.handle.commit()

