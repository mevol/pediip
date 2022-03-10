#!/bin/env python3

import os
import json
import pandas as pd
from itertools import islice
from linecache import getline

class MRSSMParser(object):
  '''
  A class to parse a pdb file
  '''
  def __init__(self, handle):
    '''
    Initialise the class with the handle
    '''
    self.handle = handle

  def add_entry(self, homologue):
    '''
    Add the pdb entry to the database
    '''
    # connect to the database
    cur = self.handle.cursor()

    # get the target PDB ID from the file path
    target_pdb = homologue.split("/")[8]

    # find PDB ID in database and retrieve key
    cur.execute('''
      SELECT id FROM pdb_id
      WHERE pdb_id="%s"
      ''' % (target_pdb))
    pdb_id = cur.fetchone()[0]

    # get PDB ID for homologue from file path
    homologue_name = homologue.split("/")[-1]

    # add homologue PDB name for given target structure to database
    cur.execute('''
      INSERT OR IGNORE INTO homologue_name (homologue_name)
      VALUES ("%s")
      ''' % (homologue_name))

    cur.execute('''
      SELECT id FROM homologue_name
      WHERE homologue_name.homologue_name="%s"
      ''' % (homologue_name))
    homologue_pk = cur.fetchall()[-1][0]

    # link the homologues to a given PDB target
    cur.execute('''
      UPDATE homologue_name
      SET pdb_id_id = "%s"
      WHERE id = "%s"
      ''' % (pdb_id, homologue_pk))    

    # add the homologue to the homologue_stats table
    cur.executescript( '''
      INSERT OR IGNORE INTO homologue_stats (homologue_name_id)
      VALUES (%s);
      ''' % (homologue_pk))

    # find the metadata.json file containing analysis stats from MR, SSM and refinement for a given homologue
    h_meta_json = os.path.join(homologue, "metadata.json")
    try:
      os.path.exists(h_meta_json)
    except:
      print("Failed to find metadata.json")
    pass

    # find the phaser.log file containing analysis stats from MR, SSM and refinement for a given homologue
    phaser_log = os.path.join(homologue, "phaser.log")
    try:
      os.path.exists(phaser_log)
    except:
      print("Failed to find phaser.log")
    pass  

    # find the prosmart_align_logfile.txt file containing analysis stats from MR, SSM and refinement for a given homologue
    prosmart_dir = os.path.join(homologue, "prosmart")
    prosmart_log = os.path.join(prosmart_dir, "prosmart_align_logfile.txt")
    try:
      os.path.exists(prosmart_log)
    except:
      print("Failed to find prosmart_align_logfile.txt")
    pass


##########################################################################################
    # transfer details from metadata.json into a dict for entering into database
    if os.path.exists(h_meta_json):
      with open(h_meta_json, "r") as h_json:
        h_reader = json.load(h_json)          
        try:
          gesamt_length = h_reader["gesamt_length"]
        except:
          gesamt_length = 0
        try:
          gesamt_qscore = h_reader["gesamt_qscore"]
        except:
          gesamt_qscore = 0
        try:
          gesamt_seqid = h_reader["gesamt_seqid"]
        except:
          gesamt_seqid = 0
        try:
          gesamt_rmsd = h_reader["gesamt_rmsd"]
        except:
          gesamt_rmsd = 0

##########################################################################################
######### Prosmart related variables
        try:
          prosmart_length_number = h_reader["prosmart_length_number"]
        except:
          prosmart_length_number = 0
        try:
          prosmart_rmsd = h_reader["prosmart_rmsd"]
        except:
          prosmart_rmsd = 0
        try:
          prosmart_seqid = h_reader["prosmart_seqid"]
        except:
          prosmart_seqid = 0
        try:
          procrustes = h_reader["procrustes"]
        except:
          procrustes = 0
        try:
          flexible = h_reader["flexible"]
        except:
          flexible = 0
        try:
          cluster0 = h_reader["cluster0"]
        except:
          cluster0 = 0
        try:
          fragments0 = h_reader["fragments0"]
        except:
          fragments0 = 0
        try:
          mean_cos_theta0 = h_reader["mean_cos_theta0"]
        except:
          mean_cos_theta0 = 0
        try:
          sd_cos_theta0 = h_reader["sd_cos_theta0"]
        except:
          sd_cos_theta0 = 0
        try:
          cluster1 = h_reader["cluster1"]
        except:
          cluster1 = 0
        try:
          fragments1 = h_reader["fragments1"]
        except:
          fragments1 = 0
        try:
          mean_cos_theta1 = h_reader["mean_cos_theta1"]
        except:
          mean_cos_theta1 = 0
        try:
          sd_cos_theta1 = h_reader["sd_cos_theta1"]
        except:
          sd_cos_theta1 = 0
######### Refmac0-Prosmart
        try:
          initial_rfree_afterSSM0 = h_reader["initial_rfree_afterSSM0"]
        except:
          initial_rfree_afterSSM0 = 0
        try:
          final_rfree_afterSSM0 = h_reader["final_rfree_afterSSM0"]
        except:
          final_rfree_afterSSM0 = 0
        try:
          initial_rwork_afterSSM0 = h_reader["initial_rwork_afterSSM0"]
        except:
          initial_rwork_afterSSM0 = 0
        try:
          final_rwork_afterSSM0 = h_reader["final_rwork_afterSSM0"]
        except:
          final_rwork_afterSSM0 = 0
######### PhaseError-Refmac0-Prosmart
        try:
          mean_phase_error_afterSSM0 = h_reader["mean_phase_error_afterSSM0"]
        except:
          mean_phase_error_afterSSM0 = 0
        try:
          f_map_correlation_afterSSM0 = h_reader["f_map_correlation_afterSSM0"]
        except:
          f_map_correlation_afterSSM0 = 0
######### Buccaneer-Refmac0-Prosmart
        try:
          num_fragments_afterSSM0 = h_reader["num_fragments_afterSSM0"]
        except:
          num_fragments_afterSSM0 = 0
        try:
          num_res_built_afterSSM0 = h_reader["num_res_built_afterSSM0"]
        except:
          num_res_built_afterSSM0 = 0
        try:
          num_res_sequenced_afterSSM0 = h_reader["num_res_sequenced_afterSSM0"]
        except:
          num_res_sequenced_afterSSM0 = 0
        try:
          num_res_unique_afterSSM0 = h_reader["num_res_unique_afterSSM0"]
        except:
          num_res_unique_afterSSM0 = 0
        try:
          longest_fragments_afterSSM0 = h_reader["longest_fragments_afterSSM0"]
        except:
          longest_fragments_afterSSM0 = 0
        try:
          percent_chain_complete_afterSSM0 = h_reader["percent_chain_complete_afterSSM0"]
        except:
          percent_chain_complete_afterSSM0 = 0
        try:
          percent_res_complete_afterSSM0 = h_reader["percent_res_complete_afterSSM0"]
        except:
          percent_res_complete_afterSSM0 = 0
######### Refmac0-Buccaneer-Refmac0-Prosmart
        try:
          initial_rfree_refmac0_afterSSM0_Buccaneer = h_reader["initial_rfree_refmac0_afterSSM0_Buccaneer"]
        except:
          initial_rfree_refmac0_afterSSM0_Buccaneer = 0
        try:
          final_rfree_refmac0_afterSSM0_Buccaneer = h_reader["final_rfree_refmac0_afterSSM0_Buccaneer"]
        except:
          final_rfree_refmac0_afterSSM0_Buccaneer = 0
        try:
          initial_rwork_refmac0_afterSSM0_Buccaneer = h_reader["initial_rwork_refmac0_afterSSM0_Buccaneer"]
        except:
          initial_rwork_refmac0_afterSSM0_Buccaneer = 0
        try:
          final_rwork_refmac0_afterSSM0_Buccaneer = h_reader["final_rwork_refmac0_afterSSM0_Buccaneer"]
        except:
          final_rwork_refmac0_afterSSM0_Buccaneer = 0
######### PhaseError-Refmac0-Buccaneer-Refmac0-Prosmart
        try:
          mean_phase_error_afterSSM0_Buccaneer_refmac0 = h_reader["mean_phase_error_afterSSM0_Buccaneer_refmac0"]
        except:
          mean_phase_error_afterSSM0_Buccaneer_refmac0 = 0
        try:
          f_map_correlation_afterSSM0_Buccaneer_refmac0 = h_reader["f_map_correlation_afterSSM0_Buccaneer_refmac0"]
        except:
          f_map_correlation_afterSSM0_Buccaneer_refmac0 = 0
######### RefmacJellyBody-Buccaneer-Refmac0-Prosmart
        try:
          initial_rfree_refmac_afterSSM0_Buccaneer = h_reader["initial_rfree_refmac_afterSSM0_Buccaneer"]
        except:
          initial_rfree_refmac_afterSSM0_Buccaneer = 0
        try:
          final_rfree_refmac_afterSSM0_Buccaneer = h_reader["final_rfree_refmac_afterSSM0_Buccaneer"]
        except:
          final_rfree_refmac_afterSSM0_Buccaneer = 0
        try:
          initial_rwork_refmac_afterSSM0_Buccaneer = h_reader["initial_rwork_refmac_afterSSM0_Buccaneer"]
        except:
          initial_rwork_refmac_afterSSM0_Buccaneer = 0
        try:
          final_rwork_refmac_afterSSM0_Buccaneer = h_reader["final_rwork_refmac_afterSSM0_Buccaneer"]
        except:
          final_rwork_refmac_afterSSM0_Buccaneer = 0
######### PhaseError-RefmacJellyBody-Buccaneer-Refmac0-Prosmart
        try:
          mean_phase_error_afterSSM0_Buccaneer_refmac = h_reader["mean_phase_error_afterSSM0_Buccaneer_refmac"]
        except:
          mean_phase_error_afterSSM0_Buccaneer_refmac = 0
        try:
          f_map_correlation_afterSSM0_Buccaneer_refmac = h_reader["f_map_correlation_afterSSM0_Buccaneer_refmac"]
        except:
          f_map_correlation_afterSSM0_Buccaneer_refmac = 0
######### RefmacJellyBody-Prosmart
        try:
          initial_rfree_afterSSM = h_reader["initial_rfree_afterSSM"]
        except:
          initial_rfree_afterSSM = 0
        try:
          final_rfree_afterSSM = h_reader["final_rfree_afterSSM"]
        except:
          final_rfree_afterSSM = 0
        try:
          initial_rwork_afterSSM = h_reader["initial_rwork_afterSSM"]
        except:
          initial_rwork_afterSSM = 0
        try:
          final_rwork_afterSSM = h_reader["final_rwork_afterSSM"]
        except:
          final_rwork_afterSSM = 0
######### PhaseError-RefmacJellyBody-Prosmart
        try:
          mean_phase_error_afterSSM = h_reader["mean_phase_error_afterSSM"]
        except:
          mean_phase_error_afterSSM = 0
        try:
          f_map_correlation_afterSSM = h_reader["f_map_correlation_afterSSM"]
        except:
          f_map_correlation_afterSSM = 0
######### Buccaneer-RefmacJellyBody-Prosmart
        try:
          num_fragments_afterSSM = h_reader["num_fragments_afterSSM"]
        except:
          num_fragments_afterSSM = 0
        try:
          num_res_built_afterSSM = h_reader["num_res_built_afterSSM"]
        except:
          num_res_built_afterSSM = 0
        try:
          num_res_sequenced_afterSSM = h_reader["num_res_sequenced_afterSSM"]
        except:
          num_res_sequenced_afterSSM = 0
        try:
          num_res_unique_afterSSM = h_reader["num_res_unique_afterSSM"]
        except:
          num_res_unique_afterSSM = 0
        try:
          longest_fragments_afterSSM = h_reader["longest_fragments_afterSSM"]
        except:
          longest_fragments_afterSSM = 0
        try:
          percent_chain_complete_afterSSM = h_reader["percent_chain_complete_afterSSM"]
        except:
          percent_chain_complete_afterSSM = 0
        try:
          percent_res_complete_afterSSM = h_reader["percent_res_complete_afterSSM"]
        except:
          percent_res_complete_afterSSM = 0
######### Refmac0-Buccaneer-RefmacJellyBody-Prosmart
        try:
          initial_rfree_refmac0_afterSSM_Buccaneer = h_reader["initial_rfree_refmac0_afterSSM_Buccaneer"]
        except:
          initial_rfree_refmac0_afterSSM_Buccaneer = 0
        try:
          final_rfree_refmac0_afterSSM_Buccaneer = h_reader["final_rfree_refmac0_afterSSM_Buccaneer"]
        except:
          final_rfree_refmac0_afterSSM_Buccaneer = 0
        try:
          initial_rwork_refmac0_afterSSM_Buccaneer = h_reader["initial_rwork_refmac0_afterSSM_Buccaneer"]
        except:
          initial_rwork_refmac0_afterSSM_Buccaneer = 0
        try:
          final_rwork_refmac0_afterSSM_Buccaneer = h_reader["final_rwork_refmac0_afterSSM_Buccaneer"]
        except:
          final_rwork_refmac0_afterSSM_Buccaneer = 0
######### PhaseError-Refmac0-Buccaneer-RefmacJellyBody-Prosmart
        try:
          mean_phase_error_afterSSM_Buccaneer_refmac0 = h_reader["mean_phase_error_afterSSM_Buccaneer_refmac0"]
        except:
          mean_phase_error_afterSSM_Buccaneer_refmac0 = 0
        try:
          f_map_correlation_afterSSM_Buccaneer_refmac0 = h_reader["f_map_correlation_afterSSM_Buccaneer_refmac0"]
        except:
          f_map_correlation_afterSSM_Buccaneer_refmac0 = 0
######### RefmacJellyBody-Buccaneer-RefmacJellyBody-Prosmart
        try:
          initial_rfree_refmac_afterSSM_Buccaneer = h_reader["initial_rfree_refmac_afterSSM_Buccaneer"]
        except:
          initial_rfree_refmac_afterSSM_Buccaneer = 0
        try:
          final_rfree_refmac_afterSSM_Buccaneer = h_reader["final_rfree_refmac_afterSSM_Buccaneer"]
        except:
          final_rfree_refmac_afterSSM_Buccaneer = 0
        try:
          initial_rwork_refmac_afterSSM_Buccaneer = h_reader["initial_rwork_refmac_afterSSM_Buccaneer"]
        except:
          initial_rwork_refmac_afterSSM_Buccaneer = 0
        try:
          final_rwork_refmac_afterSSM_Buccaneer = h_reader["final_rwork_refmac_afterSSM_Buccaneer"]
        except:
          final_rwork_refmac_afterSSM_Buccaneer = 0
######### PhaseError-RefmacJellyBody-Buccaneer-RefmacJellyBody-Prosmart
        try:
          mean_phase_error_afterSSM_Buccaneer_refmac = h_reader["mean_phase_error_afterSSM_Buccaneer_refmac"]
        except:
          mean_phase_error_afterSSM_Buccaneer_refmac = 0
        try:
          f_map_correlation_afterSSM_Buccaneer_refmac = h_reader["f_map_correlation_afterSSM_Buccaneer_refmac"]
        except:
          f_map_correlation_afterSSM_Buccaneer_refmac = 0


##########################################################################################
######### Molrep related variables
        try:
          molrep_TF_sig = h_reader["molrep_TF_sig"]
        except:
          molrep_TF_sig = 0
        try:
          molrep_contrast = h_reader["molrep_contrast"]
        except:
          molrep_contrast = 0
        try:
          molrep_corrD = h_reader["molrep_corrD"]
        except:
          molrep_corrD = 0
        try:
          molrep_corrF = h_reader["molrep_corrF"]
        except:
          molrep_corrF = 0
        try:
          molrep_final_cc = h_reader["molrep_final_cc"]
        except:
          molrep_final_cc = 0
        try:
          molrep_packing_coeff = h_reader["molrep_packing_coeff"]
        except:
          molrep_packing_coeff = 0
######### Refmac0-Molrep
        try:
          initial_rfree_afterMolrep0 = h_reader["initial_rfree_afterMolrep0"]
        except:
          initial_rfree_afterMolrep0 = 0
        try:
          final_rfree_afterMolrep0 = h_reader["final_rfree_afterMolrep0"]
        except:
          final_rfree_afterMolrep0 = 0
        try:
          initial_rwork_afterMolrep0 = h_reader["initial_rwork_afterMolrep0"]
        except:
          initial_rwork_afterMolrep0 = 0
        try:
          final_rwork_afterMolrep0 = h_reader["final_rwork_afterMolrep0"]
        except:
          final_rwork_afterMolrep0 = 0
######### PhaseError-Refmac0-Molrep
        try:
          mean_phase_error_afterMolrep0 = h_reader["mean_phase_error_afterMolrep0"]
        except:
          mean_phase_error_afterMolrep0 = 0
        try:
          f_map_correlation_afterMolrep0 = h_reader["f_map_correlation_afterMolrep0"]
        except:
          f_map_correlation_afterMolrep0 = 0
######### Buccaneer-Refmac0-Molrep
        try:
          num_fragments_afterMolrep0 = h_reader["num_fragments_afterMolrep0"]
        except:
          num_fragments_afterMolrep0 = 0
        try:
          num_res_built_afterMolrep0 = h_reader["num_res_built_afterMolrep0"]
        except:
          num_res_built_afterMolrep0 = 0
        try:
          num_res_sequenced_afterMolrep0 = h_reader["num_res_sequenced_afterMolrep0"]
        except:
          num_res_sequenced_afterMolrep0 = 0
        try:
          num_res_unique_afterMolrep0 = h_reader["num_res_unique_afterMolrep0"]
        except:
          num_res_unique_afterMolrep0 = 0
        try:
          longest_fragments_afterMolrep0 = h_reader["longest_fragments_afterMolrep0"]
        except:
          longest_fragments_afterMolrep0 = 0
        try:
          percent_chain_complete_afterMolrep0 = h_reader["percent_chain_complete_afterMolrep0"]
        except:
          percent_chain_complete_afterMolrep0 = 0
        try:
          percent_res_complete_afterMolrep0 = h_reader["percent_res_complete_afterMolrep0"]
        except:
          percent_res_complete_afterMolrep0 = 0
######### Refmac0-Buccaneer-Refmac0-Molrep
        try:
          initial_rfree_refmac0_afterMolrep0_Buccaneer = h_reader["initial_rfree_refmac0_afterMolrep0_Buccaneer"]
        except:
          initial_rfree_refmac0_afterMolrep0_Buccaneer = 0
        try:
          final_rfree_refmac0_afterMolrep0_Buccaneer = h_reader["final_rfree_refmac0_afterMolrep0_Buccaneer"]
        except:
          final_rfree_refmac0_afterMolrep0_Buccaneer = 0
        try:
          initial_rwork_refmac0_afterMolrep0_Buccaneer = h_reader["initial_rwork_refmac0_afterMolrep0_Buccaneer"]
        except:
          initial_rwork_refmac0_afterMolrep0_Buccaneer = 0
        try:
          final_rwork_refmac0_afterMolrep0_Buccaneer = h_reader["final_rwork_refmac0_afterMolrep0_Buccaneer"]
        except:
          final_rwork_refmac0_afterMolrep0_Buccaneer = 0
######### PhaseError-Refmac0-Buccaneer-Refmac0-Molrep
        try:
          mean_phase_error_afterMolrep0_Buccaneer_refmac0 = h_reader["mean_phase_error_afterMolrep0_Buccaneer_refmac0"]
        except:
          mean_phase_error_afterMolrep0_Buccaneer_refmac0 = 0
        try:
          f_map_correlation_afterMolrep0_Buccaneer_refmac0 = h_reader["f_map_correlation_afterMolrep0_Buccaneer_refmac0"]
        except:
          f_map_correlation_afterMolrep0_Buccaneer_refmac0 = 0
######### RefmacJellyBody-Buccaneer-Refmac0-Molrep
        try:
          initial_rfree_refmac_afterMolrep0_Buccaneer = h_reader["initial_rfree_refmac_afterMolrep0_Buccaneer"]
        except:
          initial_rfree_refmac_afterMolrep0_Buccaneer = 0
        try:
          final_rfree_refmac_afterMolrep0_Buccaneer = h_reader["final_rfree_refmac_afterMolrep0_Buccaneer"]
        except:
          final_rfree_refmac_afterMolrep0_Buccaneer = 0
        try:
          initial_rwork_refmac_afterMolrep0_Buccaneer = h_reader["initial_rwork_refmac_afterMolrep0_Buccaneer"]
        except:
          initial_rwork_refmac_afterMolrep0_Buccaneer = 0
        try:
          final_rwork_refmac_afterMolrep0_Buccaneer = h_reader["final_rwork_refmac_afterMolrep0_Buccaneer"]
        except:
          final_rwork_refmac_afterMolrep0_Buccaneer = 0
######### PhaseError-RefmacJellyBody-Buccaneer-Refmac0-Molrep
        try:
          mean_phase_error_afterMolrep0_Buccaneer_refmac = h_reader["mean_phase_error_afterMolrep0_Buccaneer_refmac"]
        except:
          mean_phase_error_afterMolrep0_Buccaneer_refmac = 0
        try:
          f_map_correlation_afterMolrep0_Buccaneer_refmac = h_reader["f_map_correlation_afterMolrep0_Buccaneer_refmac"]
        except:
          f_map_correlation_afterMolrep0_Buccaneer_refmac = 0
######### RefmacJellyBody-Molrep
        try:
          initial_rfree_afterMolrep = h_reader["initial_rfree_afterMolrep"]
        except:
          initial_rfree_afterMolrep = 0
        try:
          final_rfree_afterMolrep = h_reader["final_rfree_afterMolrep"]
        except:
          final_rfree_afterMolrep = 0
        try:
          initial_rwork_afterMolrep = h_reader["initial_rwork_afterMolrep"]
        except:
          initial_rwork_afterMolrep = 0
        try:
          final_rwork_afterMolrep = h_reader["final_rwork_afterMolrep"]
        except:
          final_rwork_afterMolrep = 0
######### PhaseError-RefmacJellyBody-Molrep
        try:
          mean_phase_error_afterMolrep = h_reader["mean_phase_error_afterMolrep"]
        except:
          mean_phase_error_afterMolrep = 0
        try:
          f_map_correlation_afterMolrep = h_reader["f_map_correlation_afterMolrep"]
        except:
          f_map_correlation_afterMolrep = 0
########## Buccaneer-RefmacJellyBody-Molrep
        try:
          num_fragments_afterMolrep = h_reader["num_fragments_afterMolrep"]
        except:
          num_fragments_afterMolrep = 0
        try:
          num_res_built_afterMolrep = h_reader["num_res_built_afterMolrep"]
        except:
          num_res_built_afterMolrep = 0
        try:
          num_res_sequenced_afterMolrep = h_reader["num_res_sequenced_afterMolrep"]
        except:
          num_res_sequenced_afterMolrep = 0
        try:
          num_res_unique_afterMolrep = h_reader["num_res_unique_afterMolrep"]
        except:
          num_res_unique_afterMolrep = 0
        try:
          longest_fragments_afterMolrep = h_reader["longest_fragments_afterMolrep"]
        except:
          longest_fragments_afterMolrep = 0
        try:
          percent_chain_complete_afterMolrep = h_reader["percent_chain_complete_afterMolrep"]
        except:
          percent_chain_complete_afterMolrep = 0
        try:
          percent_res_complete_afterMolrep = h_reader["percent_res_complete_afterMolrep"]
        except:
          percent_res_complete_afterMolrep = 0
######### Refmac0-Buccaneer-RefmacJellyBody-Molrep
        try:
          initial_rfree_refmac0_afterMolrep_Buccaneer = h_reader["initial_rfree_refmac0_afterMolrep_Buccaneer"]
        except:
          initial_rfree_refmac0_afterMolrep_Buccaneer = 0
        try:
          final_rfree_refmac0_afterMolrep_Buccaneer = h_reader["final_rfree_refmac0_afterMolrep_Buccaneer"]
        except:
          final_rfree_refmac0_afterMolrep_Buccaneer = 0
        try:
          initial_rwork_refmac0_afterMolrep_Buccaneer = h_reader["initial_rwork_refmac0_afterMolrep_Buccaneer"]
        except:
          initial_rwork_refmac0_afterMolrep_Buccaneer = 0
        try:
          final_rwork_refmac0_afterMolrep_Buccaneer = h_reader["final_rwork_refmac0_afterMolrep_Buccaneer"]
        except:
          final_rwork_refmac0_afterMolrep_Buccaneer = 0
######### PhaseError-Refmac0-Buccaneer-RefmacJellyBody-Molrep
        try:
          mean_phase_error_afterMolrep_Buccaneer_refmac0 = h_reader["mean_phase_error_afterMolrep_Buccaneer_refmac0"]
        except:
          mean_phase_error_afterMolrep_Buccaneer_refmac0 = 0
        try:
          f_map_correlation_afterMolrep_Buccaneer_refmac0 = h_reader["f_map_correlation_afterMolrep_Buccaneer_refmac0"]
        except:
          f_map_correlation_afterMolrep_Buccaneer_refmac0 = 0
######### RefmacJellyBody-Buccaneer-RefmacJellyBody-Molrep
        try:
          initial_rfree_refmac_afterMolrep_Buccaneer = h_reader["initial_rfree_refmac_afterMolrep_Buccaneer"]
        except:
          initial_rfree_refmac_afterMolrep_Buccaneer = 0
        try:
          final_rfree_refmac_afterMolrep_Buccaneer = h_reader["final_rfree_refmac_afterMolrep_Buccaneer"]
        except:
          final_rfree_refmac_afterMolrep_Buccaneer = 0
        try:
          initial_rwork_refmac_afterMolrep_Buccaneer = h_reader["initial_rwork_refmac_afterMolrep_Buccaneer"]
        except:
          initial_rwork_refmac_afterMolrep_Buccaneer = 0
        try:
          final_rwork_refmac_afterMolrep_Buccaneer = h_reader["final_rwork_refmac_afterMolrep_Buccaneer"]
        except:
          final_rwork_refmac_afterMolrep_Buccaneer = 0
######### PhaseError-RefmacJellyBody-Buccaneer-RefmacJellyBody-Molrep
        try:
          mean_phase_error_afterMolrep_Buccaneer_refmac = h_reader["mean_phase_error_afterMolrep_Buccaneer_refmac"]
        except:
          mean_phase_error_afterMolrep_Buccaneer_refmac = 0
        try:
          f_map_correlation_afterMolrep_Buccaneer_refmac = h_reader["f_map_correlation_afterMolrep_Buccaneer_refmac"]
        except:
          f_map_correlation_afterMolrep_Buccaneer_refmac = 0

##########################################################################################
########## Phaser related variables
        try:
          phaser_llg = h_reader["phaser_llg"]
        except:
          phaser_llg = 0.0
        try:
          phaser_rmsd = h_reader["phaser_rmsd"]
        except:
          phaser_rmsd = 0
######### Refmac0-MR
        try:
          initial_rfree_afterMR0 = h_reader["initial_rfree_afterMR0"]
        except:
          initial_rfree_afterMR0 = 0
        try:
          final_rfree_afterMR0 = h_reader["final_rfree_afterMR0"]
        except:
          final_rfree_afterMR0 = 0
        try:
          initial_rwork_afterMR0 = h_reader["initial_rwork_afterMR0"]
        except:
          initial_rwork_afterMR0 = 0
        try:
          final_rwork_afterMR0 = h_reader["final_rwork_afterMR0"]
        except:
          final_rwork_afterMR0 = 0
######### PhaseError-Refmac0-MR
        try:
          mean_phase_error_afterMR0 = h_reader["mean_phase_error_afterMR0"]
        except:
          mean_phase_error_afterMR0 = 0
        try:
          f_map_correlation_afterMR0 = h_reader["f_map_correlation_afterMR0"]
        except:
          f_map_correlation_afterMR0 = 0
######### Buccaneer-Refmac0-MR
        try:
          num_fragments_afterMR0 = h_reader["num_fragments_afterMR0"]
        except:
          num_fragments_afterMR0 = 0
        try:
          num_res_built_afterMR0 = h_reader["num_res_built_afterMR0"]
        except:
          num_res_built_afterMR0 = 0
        try:
          num_res_sequenced_afterMR0 = h_reader["num_res_sequenced_afterMR0"]
        except:
          num_res_sequenced_afterMR0 = 0
        try:
          num_res_unique_afterMR0 = h_reader["num_res_unique_afterMR0"]
        except:
          num_res_unique_afterMR0 = 0
        try:
          longest_fragments_afterMR0 = h_reader["longest_fragments_afterMR0"]
        except:
          longest_fragments_afterMR0 = 0
        try:
          percent_chain_complete_afterMR0 = h_reader["percent_chain_complete_afterMR0"]
        except:
          percent_chain_complete_afterMR0 = 0
        try:
          percent_res_complete_afterMR0 = h_reader["percent_res_complete_afterMR0"]
        except:
          percent_res_complete_afterMR0 = 0
######### Refmac0-Buccaneer-Refmac0-MR
        try:
          initial_rfree_refmac0_afterMR0_Buccaneer = h_reader["initial_rfree_refmac0_afterMR0_Buccaneer"]
        except:
          initial_rfree_refmac0_afterMR0_Buccaneer = 0
        try:
          final_rfree_refmac0_afterMR0_Buccaneer = h_reader["final_rfree_refmac0_afterMR0_Buccaneer"]
        except:
          final_rfree_refmac0_afterMR0_Buccaneer = 0
        try:
          initial_rwork_refmac0_afterMR0_Buccaneer = h_reader["initial_rwork_refmac0_afterMR0_Buccaneer"]
        except:
          initial_rwork_refmac0_afterMR0_Buccaneer = 0
        try:
          final_rwork_refmac0_afterMR0_Buccaneer = h_reader["final_rwork_refmac0_afterMR0_Buccaneer"]
        except:
          final_rwork_refmac0_afterMR0_Buccaneer = 0
######### PhaseError-Refmac0-Buccaneer-Refmac0-MR
        try:
          mean_phase_error_afterMR0_Buccaneer_refmac0 = h_reader["mean_phase_error_afterMR0_Buccaneer_refmac0"]
        except:
          mean_phase_error_afterMR0_Buccaneer_refmac0 = 0
        try:
          f_map_correlation_afterMR0_Buccaneer_refmac0 = h_reader["f_map_correlation_afterMR0_Buccaneer_refmac0"]
        except:
          f_map_correlation_afterMR0_Buccaneer_refmac0 = 0
######### RefmacJellyBody-Buccaneer-Refmac0-MR
        try:
          initial_rfree_refmac_afterMR0_Buccaneer = h_reader["initial_rfree_refmac_afterMR0_Buccaneer"]
        except:
          initial_rfree_refmac_afterMR0_Buccaneer = 0
        try:
          final_rfree_refmac_afterMR0_Buccaneer = h_reader["final_rfree_refmac_afterMR0_Buccaneer"]
        except:
          final_rfree_refmac_afterMR0_Buccaneer = 0
        try:
          initial_rwork_refmac_afterMR0_Buccaneer = h_reader["initial_rwork_refmac_afterMR0_Buccaneer"]
        except:
          initial_rwork_refmac_afterMR0_Buccaneer = 0
        try:
          final_rwork_refmac_afterMR0_Buccaneer = h_reader["final_rwork_refmac_afterMR0_Buccaneer"]
        except:
          final_rwork_refmac_afterMR0_Buccaneer = 0
######### PhaseError-RefmacJellyBody-Buccaneer-Refmac0-MR
        try:
          mean_phase_error_afterMR0_Buccaneer_refmac = h_reader["mean_phase_error_afterMR0_Buccaneer_refmac"]
        except:
          mean_phase_error_afterMR0_Buccaneer_refmac = 0
        try:
          f_map_correlation_afterMR0_Buccaneer_refmac = h_reader["f_map_correlation_afterMR0_Buccaneer_refmac"]
        except:
          f_map_correlation_afterMR0_Buccaneer_refmac = 0
######### RefmacJellyBody-MR
        try:
          initial_rfree_afterMR = h_reader["initial_rfree_afterMR"]
        except:
          initial_rfree_afterMR = 0
        try:
          final_rfree_afterMR = h_reader["final_rfree_afterMR"]
        except:
          final_rfree_afterMR = 0
        try:
          initial_rwork_afterMR = h_reader["initial_rwork_afterMR"]
        except:
          initial_rwork_afterMR = 0
        try:
          final_rwork_afterMR = h_reader["final_rwork_afterMR"]
        except:
          final_rwork_afterMR = 0
######### PhaseError-RefmacJellyBody-MR
        try:
          mean_phase_error_afterMR = h_reader["mean_phase_error_afterMR"]
        except:
          mean_phase_error_afterMR = 0
        try:
          f_map_correlation_afterMR = h_reader["f_map_correlation_afterMR"]
        except:
          f_map_correlation_afterMR = 0
######### Buccaneer-RefmacJellyBody-MR
        try:
          num_fragments_afterMR = h_reader["num_fragments_afterMR"]
        except:
          num_fragments_afterMR = 0
        try:
          num_res_built_afterMR = h_reader["num_res_built_afterMR"]
        except:
          num_res_built_afterMR = 0
        try:
          num_res_sequenced_afterMR = h_reader["num_res_sequenced_afterMR"]
        except:
          num_res_sequenced_afterMR = 0
        try:
          num_res_unique_afterMR = h_reader["num_res_unique_afterMR"]
        except:
          num_res_unique_afterMR = 0
        try:
          longest_fragments_afterMR = h_reader["longest_fragments_afterMR"]
        except:
          longest_fragments_afterMR = 0
        try:
          percent_chain_complete_afterMR = h_reader["percent_chain_complete_afterMR"]
        except:
          percent_chain_complete_afterMR = 0
        try:
          percent_res_complete_afterMR = h_reader["percent_res_complete_afterMR"]
        except:
          percent_res_complete_afterMR = 0
######### Refmac0-Buccaneer-RefmacJellyBody-MR
        try:
          initial_rfree_refmac0_afterMR_Buccaneer = h_reader["initial_rfree_refmac0_afterMR_Buccaneer"]
        except:
          initial_rfree_refmac0_afterMR_Buccaneer = 0
        try:
          final_rfree_refmac0_afterMR_Buccaneer = h_reader["final_rfree_refmac0_afterMR_Buccaneer"]
        except:
          final_rfree_refmac0_afterMR_Buccaneer = 0
        try:
          initial_rwork_refmac0_afterMR_Buccaneer = h_reader["initial_rwork_refmac0_afterMR_Buccaneer"]
        except:
          initial_rwork_refmac0_afterMR_Buccaneer = 0
        try:
          final_rwork_refmac0_afterMR_Buccaneer = h_reader["final_rwork_refmac0_afterMR_Buccaneer"]
        except:
          final_rwork_refmac0_afterMR_Buccaneer = 0
######### PhaseError-Refmac0-Buccaneer-RefmacJellyBody-MR
        try:
          mean_phase_error_afterMR_Buccaneer_refmac0 = h_reader["mean_phase_error_afterMR_Buccaneer_refmac0"]
        except:
          mean_phase_error_afterMR_Buccaneer_refmac0 = 0
        try:
          f_map_correlation_afterMR_Buccaneer_refmac0 = h_reader["f_map_correlation_afterMR_Buccaneer_refmac0"]
        except:
          f_map_correlation_afterMR_Buccaneer_refmac0 = 0
######### RefmacJellyBody-Buccaneer-RefmacJellyBody-MR
        try:
          initial_rfree_refmac_afterMR_Buccaneer = h_reader["initial_rfree_refmac_afterMR_Buccaneer"]
        except:
          initial_rfree_refmac_afterMR_Buccaneer = 0
        try:
          final_rfree_refmac_afterMR_Buccaneer = h_reader["final_rfree_refmac_afterMR_Buccaneer"]
        except:
          final_rfree_refmac_afterMR_Buccaneer = 0
        try:
          initial_rwork_refmac_afterMR_Buccaneer = h_reader["initial_rwork_refmac_afterMR_Buccaneer"]
        except:
          initial_rwork_refmac_afterMR_Buccaneer = 0
        try:
          final_rwork_refmac_afterMR_Buccaneer = h_reader["final_rwork_refmac_afterMR_Buccaneer"]
        except:
          final_rwork_refmac_afterMR_Buccaneer = 0
######### PhaseError-RefmacJellyBody-Buccaneer-RefmacJellyBody-MR
        try:
          mean_phase_error_afterMR_Buccaneer_refmac = h_reader["mean_phase_error_afterMR_Buccaneer_refmac"]
        except:
          mean_phase_error_afterMR_Buccaneer_refmac = 0
        try:
          f_map_correlation_afterMR_Buccaneer_refmac = h_reader["f_map_correlation_afterMR_Buccaneer_refmac"]
        except:
          f_map_correlation_afterMR_Buccaneer_refmac = 0
          
          
        # assigning labels regarding the MR out come using an Rfree cut-off of 0.5
        # this is for now left as TO DO; I will assign the labels when working with the data downstream;
        # I may have to break it down into smaller sub-dataframes



# this is the original way for assigning the labels based on what Garib had in mind; this will be ignored for now;
########################################################################################################################
#        
#        if prosmart_rmsd > 0 or molrep_contrast > 0:
#          mr_success_lable = 2
#          if final_rfree_afterSSM <= 0.5 or final_rfree_afterMolrep <= 0.5:
#            refinement_success_lable = "2a"
#          if final_rfree_afterSSM > 0.5 or final_rfree_afterSSM == 0 or final_rfree_afterMolrep > 0.5 or final_rfree_afterMolrep == 0:
#            refinement_success_lable = "2b"
#        #if final_rfree <= initial_rfree and phaser_llg >= 60.0:
#        if phaser_llg >= 60.0:  
#          #mr_success_lable = 1
#          mr_success_lable = 1
#          if final_rfree_afterMR <= 0.5:
#            refinement_success_lable = "1a"
#          if final_rfree_afterMR > 0.5 or final_rfree_afterMR == 0:
#            refinement_success_lable = "1b"          
#        #mr_success_lable = 0
#        mr_success_lable = "no"
#        if phaser_llg > 0.0:  
#          mr_success_lable = "very_weak"          
#        #if final_rfree <= initial_rfree and phaser_llg >= 60.0:
#        if phaser_llg >= 60.0:  
#          #mr_success_lable = 1
#          mr_success_lable = "weak"
#        #if final_rfree <= initial_rfree and phaser_llg >= 120.0:
#        if phaser_llg >= 120.0:  
#          #mr_success_lable = 2
#          mr_success_lable = "yes"
#########################################################################################################################          

    # transfer details from phaser.log into a dict for entering into database
    if os.path.exists(phaser_log):
      with open(phaser_log, "r") as p_log:
        for line in p_log:
          if line.rstrip() == "   eLLG: eLLG of chain alone":
            phaser_ellg = list(islice(p_log, 2))[1].split()[0]
    else:
      phaser_ellg = 0

    # transfer details from prosmart_align_logfile.txt into a dict for entering into database
    if os.path.exists(prosmart_log):
      with open(prosmart_log, "r") as pro_log:
        for ind, line in enumerate(pro_log, 1):
          if line.strip().startswith("Average residue scores:"):
            result1 = list(islice(pro_log, 2))
            procrustes = result1[0].split()[-1]
            flexible = result1[1].split()[-1]
          if line.strip().startswith("Final clustering results:"):
            dummy0 = list(islice(pro_log, 2))
            split0 = dummy0[1].split()
            cluster0 = split0[0]
            fragments0 = split0[1]
            mean_cos_theta0 = split0[2]
            sd_cos_theta0 = split0[3]
            dummy1 = getline(pro_log.name, ind + 5).split('\t')#add 1 to 5 to go to 6th line
            try:
              split1 = list(filter(None, dummy1))
              cluster1 = split1[0]
              fragments1 = split1[1]
              mean_cos_theta1 = split1[2]
              sd_cos_theta1 = split1[3].strip("\n")
            except:
              cluster1 = 0
              fragments1 = 0
              mean_cos_theta1 = 0
              sd_cos_theta1 = 0

    homologue_dict = {
        # GESAMT related variables
        "gesamt_length"                    : gesamt_length,
        "gesamt_qscore"                    : gesamt_qscore,
        "gesamt_seqid"                     : gesamt_seqid,
        "gesamt_rmsd"                      : gesamt_rmsd,
        # Prosmart related variables
        "prosmart_length_number"           : prosmart_length_number,
        "prosmart_rmsd"                    : prosmart_rmsd,
        "prosmart_seqid"                   : prosmart_seqid,
        "prosmart_procrustes"              : procrustes,
        "prosmart_flexible"                : flexible,
        "prosmart_cluster0_fragments"      : fragments0,
        "prosmart_cluster0_mean_cos_theta" : mean_cos_theta0,
        "prosmart_cluster0_sd_cos_theta"   : sd_cos_theta0,
        "prosmart_cluster1_fragments"      : fragments1,
        "prosmart_cluster1_mean_cos_theta" : mean_cos_theta1,
        "prosmart_cluster1_sd_cos_theta"   : sd_cos_theta1,
        "initial_rfree_afterSSM0"          : initial_rfree_afterSSM0,
        "final_rfree_afterSSM0"            : final_rfree_afterSSM0, 
        "initial_rwork_afterSSM0"          : initial_rwork_afterSSM0, 
        "final_rwork_afterSSM0"            : final_rwork_afterSSM0,
        "mean_phase_error_afterSSM0"       : mean_phase_error_afterSSM0,
        "f_map_correlation_afterSSM0"      : f_map_correlation_afterSSM0,
        "num_fragments_afterSSM0"          : num_fragments_afterSSM0,
        "num_res_built_afterSSM0"          : num_res_built_afterSSM0,
        "num_res_sequenced_afterSSM0"      : num_res_sequenced_afterSSM0,
        "num_res_unique_afterSSM0"         : num_res_unique_afterSSM0,
        "longest_fragments_afterSSM0"      : longest_fragments_afterSSM0,
        "percent_chain_complete_afterSSM0" : percent_chain_complete_afterSSM0,
        "percent_res_complete_afterSSM0" : percent_res_complete_afterSSM0,
        "initial_rfree_refmac0_afterSSM0_Buccaneer" : initial_rfree_refmac0_afterSSM0_Buccaneer,
        "final_rfree_refmac0_afterSSM0_Buccaneer" : final_rfree_refmac0_afterSSM0_Buccaneer,
        "initial_rwork_refmac0_afterSSM0_Buccaneer" : initial_rwork_refmac0_afterSSM0_Buccaneer,
        "final_rwork_refmac0_afterSSM0_Buccaneer" : final_rwork_refmac0_afterSSM0_Buccaneer,
        "mean_phase_error_afterSSM0_Buccaneer_refmac0" : mean_phase_error_afterSSM0_Buccaneer_refmac0,
        "f_map_correlation_afterSSM0_Buccaneer_refmac0" : f_map_correlation_afterSSM0_Buccaneer_refmac0,
        "initial_rfree_refmac_afterSSM0_Buccaneer" : initial_rfree_refmac_afterSSM0_Buccaneer,
        "final_rfree_refmac_afterSSM0_Buccaneer" : final_rfree_refmac_afterSSM0_Buccaneer,
        "initial_rwork_refmac_afterSSM0_Buccaneer" : initial_rwork_refmac_afterSSM0_Buccaneer,
        "final_rwork_refmac_afterSSM0_Buccaneer" : final_rwork_refmac_afterSSM0_Buccaneer,
        "mean_phase_error_afterSSM0_Buccaneer_refmac" : mean_phase_error_afterSSM0_Buccaneer_refmac,
        "f_map_correlation_afterSSM0_Buccaneer_refmac" : f_map_correlation_afterSSM0_Buccaneer_refmac,
        "initial_rfree_afterSSM"           : initial_rfree_afterSSM,
        "final_rfree_afterSSM"             : final_rfree_afterSSM,
        "initial_rwork_afterSSM"           : initial_rwork_afterSSM,
        "final_rwork_afterSSM"             : final_rwork_afterSSM,
        "mean_phase_error_afterSSM"        : mean_phase_error_afterSSM,
        "f_map_correlation_afterSSM"       : f_map_correlation_afterSSM,
        "num_fragments_afterSSM"           : num_fragments_afterSSM,
        "num_res_built_afterSSM"           : num_res_built_afterSSM,
        "num_res_sequenced_afterSSM"       : num_res_sequenced_afterSSM,
        "num_res_unique_afterSSM"          : num_res_unique_afterSSM,
        "longest_fragments_afterSSM"       : longest_fragments_afterSSM,
        "percent_chain_complete_afterSSM" : percent_chain_complete_afterSSM,
        "percent_res_complete_afterSSM" : percent_res_complete_afterSSM,
        "initial_rfree_refmac0_afterSSM_Buccaneer" : initial_rfree_refmac0_afterSSM_Buccaneer,
        "final_rfree_refmac0_afterSSM_Buccaneer" : final_rfree_refmac0_afterSSM_Buccaneer,
        "initial_rwork_refmac0_afterSSM_Buccaneer" : initial_rwork_refmac0_afterSSM_Buccaneer,
        "final_rwork_refmac0_afterSSM_Buccaneer" : final_rwork_refmac0_afterSSM_Buccaneer,
        "mean_phase_error_afterSSM_Buccaneer_refmac0" : mean_phase_error_afterSSM_Buccaneer_refmac0,
        "f_map_correlation_afterSSM_Buccaneer_refmac0" : f_map_correlation_afterSSM_Buccaneer_refmac0,
        "initial_rfree_refmac_afterSSM_Buccaneer" : initial_rfree_refmac_afterSSM_Buccaneer,
        "final_rfree_refmac_afterSSM_Buccaneer" : final_rfree_refmac_afterSSM_Buccaneer,
        "initial_rwork_refmac_afterSSM_Buccaneer" : initial_rwork_refmac_afterSSM_Buccaneer,
        "final_rwork_refmac_afterSSM_Buccaneer" : final_rwork_refmac_afterSSM_Buccaneer,
        "mean_phase_error_afterSSM_Buccaneer_refmac" : mean_phase_error_afterSSM_Buccaneer_refmac,
        "f_map_correlation_afterSSM_Buccaneer_refmac" : f_map_correlation_afterSSM_Buccaneer_refmac,
        # Molrep related variables
        "molrep_TF_sig"                    : molrep_TF_sig,
        "molrep_contrast"                  : molrep_contrast,
        "molrep_corrD"                     : molrep_corrD,
        "molrep_corrF"                     : molrep_corrF,
        "molrep_final_cc"                  : molrep_final_cc,
        "molrep_packing_coeff"             : molrep_packing_coeff,
        "initial_rfree_afterMolrep0"       : initial_rfree_afterMolrep0,
        "final_rfree_afterMolrep0"         : final_rfree_afterMolrep0,
        "initial_rwork_afterMolrep0"       : initial_rwork_afterMolrep0,
        "final_rwork_afterMolrep0"         : final_rwork_afterMolrep0,
        "mean_phase_error_afterMolrep0"    : mean_phase_error_afterMolrep0,
        "f_map_correlation_afterMolrep0"   : f_map_correlation_afterMolrep0,
        "num_fragments_afterMolrep0"          : num_fragments_afterMolrep0,
        "num_res_built_afterMolrep0"          : num_res_built_afterMolrep0,
        "num_res_sequenced_afterMolrep0"      : num_res_sequenced_afterMolrep0,
        "num_res_unique_afterMolrep0"         : num_res_unique_afterMolrep0,
        "longest_fragments_afterMolrep0"      : longest_fragments_afterMolrep0,
        "percent_chain_complete_afterMolrep0" : percent_chain_complete_afterMolrep0,
        "percent_res_complete_afterMolrep0" : percent_res_complete_afterMolrep0,
        "initial_rfree_refmac0_afterMolrep0_Buccaneer" : initial_rfree_refmac0_afterMolrep0_Buccaneer,
        "final_rfree_refmac0_afterMolrep0_Buccaneer" : final_rfree_refmac0_afterMolrep0_Buccaneer,
        "initial_rwork_refmac0_afterMolrep0_Buccaneer" : initial_rwork_refmac0_afterMolrep0_Buccaneer,
        "final_rwork_refmac0_afterMolrep0_Buccaneer" : final_rwork_refmac0_afterMolrep0_Buccaneer,
        "mean_phase_error_afterMolrep0_Buccaneer_refmac0" : mean_phase_error_afterMolrep0_Buccaneer_refmac0,
        "f_map_correlation_afterMolrep0_Buccaneer_refmac0" : f_map_correlation_afterMolrep0_Buccaneer_refmac0,
        "initial_rfree_refmac_afterMolrep0_Buccaneer" : initial_rfree_refmac_afterMolrep0_Buccaneer,
        "final_rfree_refmac_afterMolrep0_Buccaneer" : final_rfree_refmac_afterMolrep0_Buccaneer,
        "initial_rwork_refmac_afterMolrep0_Buccaneer" : initial_rwork_refmac_afterMolrep0_Buccaneer,
        "final_rwork_refmac_afterMolrep0_Buccaneer" : final_rwork_refmac_afterMolrep0_Buccaneer,
        "mean_phase_error_afterMolrep0_Buccaneer_refmac" : mean_phase_error_afterMolrep0_Buccaneer_refmac,
        "f_map_correlation_afterMolrep0_Buccaneer_refmac" : f_map_correlation_afterMolrep0_Buccaneer_refmac,
        "initial_rfree_afterMolrep"           : initial_rfree_afterMolrep,
        "final_rfree_afterMolrep"             : final_rfree_afterMolrep,
        "initial_rwork_afterMolrep"           : initial_rwork_afterMolrep,
        "final_rwork_afterMolrep"             : final_rwork_afterMolrep,
        "mean_phase_error_afterMolrep"        : mean_phase_error_afterMolrep,
        "f_map_correlation_afterMolrep"       : f_map_correlation_afterMolrep,
        "num_fragments_afterMolrep"           : num_fragments_afterMolrep,
        "num_res_built_afterMolrep"           : num_res_built_afterMolrep,
        "num_res_sequenced_afterMolrep"       : num_res_sequenced_afterMolrep,
        "num_res_unique_afterMolrep"          : num_res_unique_afterMolrep,
        "longest_fragments_afterMolrep"       : longest_fragments_afterMolrep,
        "percent_chain_complete_afterMolrep" : percent_chain_complete_afterMolrep,
        "percent_res_complete_afterMolrep" : percent_res_complete_afterMolrep,
        "initial_rfree_refmac0_afterMolrep_Buccaneer" : initial_rfree_refmac0_afterMolrep_Buccaneer,
        "final_rfree_refmac0_afterMolrep_Buccaneer" : final_rfree_refmac0_afterMolrep_Buccaneer,
        "initial_rwork_refmac0_afterMolrep_Buccaneer" : initial_rwork_refmac0_afterMolrep_Buccaneer,
        "final_rwork_refmac0_afterMolrep_Buccaneer" : final_rwork_refmac0_afterMolrep_Buccaneer,
        "mean_phase_error_afterMolrep_Buccaneer_refmac0" : mean_phase_error_afterMolrep_Buccaneer_refmac0,
        "f_map_correlation_afterMolrep_Buccaneer_refmac0" : f_map_correlation_afterMolrep_Buccaneer_refmac0,
        "initial_rfree_refmac_afterMolrep_Buccaneer" : initial_rfree_refmac_afterMolrep_Buccaneer,
        "final_rfree_refmac_afterMolrep_Buccaneer" : final_rfree_refmac_afterMolrep_Buccaneer,
        "initial_rwork_refmac_afterMolrep_Buccaneer" : initial_rwork_refmac_afterMolrep_Buccaneer,
        "final_rwork_refmac_afterMolrep_Buccaneer" : final_rwork_refmac_afterMolrep_Buccaneer,
        "mean_phase_error_afterMolrep_Buccaneer_refmac" : mean_phase_error_afterMolrep_Buccaneer_refmac,
        "f_map_correlation_afterMolrep_Buccaneer_refmac" : f_map_correlation_afterMolrep_Buccaneer_refmac,
        # Phaser related variables
        "phaser_ellg"                      : phaser_ellg,
        "phaser_llg"                       : phaser_llg,
        "phaser_rmsd"                      : phaser_rmsd,
        "initial_rfree_afterMR0"           : initial_rfree_afterMR0,
        "final_rfree_afterMR0"             : final_rfree_afterMR0,     
        "initial_rwork_afterMR0"           : initial_rwork_afterMR0,
        "final_rwork_afterMR0"             : final_rwork_afterMR0,
        "mean_phase_error_afterMR0"        : mean_phase_error_afterMR0,
        "f_map_correlation_afterMR0"       : f_map_correlation_afterMR0,  
        "num_fragments_afterMR0"          : num_fragments_afterMR0,
        "num_res_built_afterMR0"          : num_res_built_afterMR0,
        "num_res_sequenced_afterMR0"      : num_res_sequenced_afterMR0,
        "num_res_unique_afterMR0"         : num_res_unique_afterMR0,
        "longest_fragments_afterMR0"      : longest_fragments_afterMR0,
        "percent_chain_complete_afterMR0" : percent_chain_complete_afterMR0,
        "percent_res_complete_afterMR0" : percent_res_complete_afterMR0,
        "initial_rfree_refmac0_afterMR0_Buccaneer" : initial_rfree_refmac0_afterMR0_Buccaneer,
        "final_rfree_refmac0_afterMR0_Buccaneer" : final_rfree_refmac0_afterMR0_Buccaneer,
        "initial_rwork_refmac0_afterMR0_Buccaneer" : initial_rwork_refmac0_afterMR0_Buccaneer,
        "final_rwork_refmac0_afterMR0_Buccaneer" : final_rwork_refmac0_afterMR0_Buccaneer,
        "mean_phase_error_afterMR0_Buccaneer_refmac0" : mean_phase_error_afterMR0_Buccaneer_refmac0,
        "f_map_correlation_afterMR0_Buccaneer_refmac0" : f_map_correlation_afterMR0_Buccaneer_refmac0,
        "initial_rfree_refmac_afterMR0_Buccaneer" : initial_rfree_refmac_afterMR0_Buccaneer,
        "final_rfree_refmac_afterMR0_Buccaneer" : final_rfree_refmac_afterMR0_Buccaneer,
        "initial_rwork_refmac_afterMR0_Buccaneer" : initial_rwork_refmac_afterMR0_Buccaneer,
        "final_rwork_refmac_afterMR0_Buccaneer" : final_rwork_refmac_afterMR0_Buccaneer,
        "mean_phase_error_afterMR0_Buccaneer_refmac" : mean_phase_error_afterMR0_Buccaneer_refmac,
        "f_map_correlation_afterMR0_Buccaneer_refmac" : f_map_correlation_afterMR0_Buccaneer_refmac,
        "initial_rfree_afterMR"           : initial_rfree_afterMR,
        "final_rfree_afterMR"             : final_rfree_afterMR,
        "initial_rwork_afterMR"           : initial_rwork_afterMR,
        "final_rwork_afterMR"             : final_rwork_afterMR,
        "mean_phase_error_afterMR"        : mean_phase_error_afterMR,
        "f_map_correlation_afterMR"       : f_map_correlation_afterMR,
        "num_fragments_afterMR"           : num_fragments_afterMR,
        "num_res_built_afterMR"           : num_res_built_afterMR,
        "num_res_sequenced_afterMR"       : num_res_sequenced_afterMR,
        "num_res_unique_afterMR"          : num_res_unique_afterMR,
        "longest_fragments_afterMR"       : longest_fragments_afterMR,
        "percent_chain_complete_afterMR" : percent_chain_complete_afterMR,
        "percent_res_complete_afterMR" : percent_res_complete_afterMR,
        "initial_rfree_refmac0_afterMR_Buccaneer" : initial_rfree_refmac0_afterMR_Buccaneer,
        "final_rfree_refmac0_afterMR_Buccaneer" : final_rfree_refmac0_afterMR_Buccaneer,
        "initial_rwork_refmac0_afterMR_Buccaneer" : initial_rwork_refmac0_afterMR_Buccaneer,
        "final_rwork_refmac0_afterMR_Buccaneer" : final_rwork_refmac0_afterMR_Buccaneer,
        "mean_phase_error_afterMR_Buccaneer_refmac0" : mean_phase_error_afterMR_Buccaneer_refmac0,
        "f_map_correlation_afterMR_Buccaneer_refmac0" : f_map_correlation_afterMR_Buccaneer_refmac0,
        "initial_rfree_refmac_afterMR_Buccaneer" : initial_rfree_refmac_afterMR_Buccaneer,
        "final_rfree_refmac_afterMR_Buccaneer" : final_rfree_refmac_afterMR_Buccaneer,
        "initial_rwork_refmac_afterMR_Buccaneer" : initial_rwork_refmac_afterMR_Buccaneer,
        "final_rwork_refmac_afterMR_Buccaneer" : final_rwork_refmac_afterMR_Buccaneer,
        "mean_phase_error_afterMR_Buccaneer_refmac" : mean_phase_error_afterMR_Buccaneer_refmac,
        "f_map_correlation_afterMR_Buccaneer_refmac" : f_map_correlation_afterMR_Buccaneer_refmac,
#        "mr_success_lable"                 : mr_success_lable,
#        "refinement_success_lable"         : refinement_success_lable
        }

    # writing stats in the homologue dict to the database
    for entry in homologue_dict:
      cur.execute('''
        UPDATE homologue_stats
        SET "%s" = "%s"
        WHERE homologue_name_id = "%s";
        ''' % (entry, homologue_dict[entry], homologue_pk))

    self.handle.commit()
