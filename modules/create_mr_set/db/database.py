#!/bin/env python3

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


  def add_pdb_redo(self, structure, local_pdb_redo):
    '''
    Add a pdb redo stats to the database
    '''
    print("Adding PDB-redo stats for target structure: ", structure)
    parser = PDBRedo(self.handle)
    parser.add_entry(structure, local_pdb_redo)


  def add_pdb_targets(self, structure, results_dir, local_pdb):
    '''
    Add PDB targets to the database
    '''    
    print("Adding PDB stats for target structure: ", structure)
    parser = TargetParser(self.handle)
    parser.add_entry(structure, results_dir, local_pdb)


  def add_mr_ssm_stats(self, homologue, local_pdb_redo):
    '''
    Add homologue details to the database
    '''    
    print("Adding MR and SSM stats for homologue: ", homologue)
    parser = MRSSMParser(self.handle)
    parser.add_entry(homologue, local_pdb_redo)

