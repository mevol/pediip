#!/bin/env python3
import sqlite3
import os

from modules.create_mr_set.db.initialiser import Initialiser
from modules.create_mr_set.db.pdb_redo_parser import PDBRedo
from modules.create_mr_set.db.pdb_mr_ssm_parser import MRSSMParser
from modules.create_mr_set.db.target_parser import TargetParser

class DB(object):
  '''
  A high level class to perform operations on the results database
  '''

  def __init__(self, overwrite = False):
    '''
    Initialise the database
    '''
    initialiser = Initialiser(overwrite=overwrite)
    self.handle = initialiser.handle


  def add_pdb_redo(self, filename):
    '''
    Add a pdb redo stats to the database
    '''
    parser = PDBRedo(self.handle)
    parser.add_entry(filename)


  def add_pdb_targets(self, structure, results_dir, local_pdb):
    '''
    Add PDB targets to the database
    '''    
    print("Adding target structure", structure)
    parser = TargetParser(self.handle)
    parser.add_entry(structure, results_dir, local_pdb)


  def add_mr_ssm_stats(self, homologue):
    '''
    Add homologue details to the database
    '''    
    print("Homologue handed to the parser", homologue)
    parser = MRSSMParser(self.handle)
    parser.add_entry(homologue)

