#!/bin/env python3

import sqlite3
import os

class Initialiser(object):
  '''
  A class to initialise the database
  '''

  def __init__(self, overwrite=False):
    '''
    Get the database handle
    '''

    # Check if we need to init
    if not os.path.exists('results.sqlite') or overwrite:
      init = True
    else:
      init = False

    # Get the handle
    self.handle = sqlite3.connect('results.sqlite')

    # Initialise if we need to
    if init:
      self._initialise()

  def _initialise(self):
    '''
    Initialise the table; do not add column labels yet
    '''

    # Get a cursor
    cur = self.handle.cursor()

    # Execute the commands to initialise the table
    cur.executescript('''
      DROP TABLE IF EXISTS pdb_id;
      DROP TABLE IF EXISTS homologue_name;
      DROP TABLE IF EXISTS pdb_redo_stats;
      DROP TABLE IF EXISTS target_stats;
      DROP TABLE IF EXISTS homologue_stats;
      
      CREATE TABLE pdb_id (
          id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT UNIQUE,
          pdb_id  TEXT UNIQUE
      );
      CREATE TABLE homologue_name (
          id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT UNIQUE,
          pdb_id_id INTEGER,
          homologue_name  TEXT,
          FOREIGN KEY (pdb_id_id) REFERENCES pdb_id(id)
      );
      CREATE TABLE pdb_redo_stats (
          pdb_id_id INTEGER,
          rWork_depo TEXT,
          rFree_depo TEXT,
          rWork_tls TEXT,
          rFree_tls TEXT,
          rWork_final TEXT,
          rFree_final TEXT,
          completeness TEXT,          
          FOREIGN KEY (pdb_id_id) REFERENCES pdb_id(id)
      );
      CREATE TABLE target_stats (
          pdb_id_id INTEGER,
          number_of_copies INTEGER,
          sequence_length INTEGER,
          sequence TEXT,
          number_of_homologues INTEGER,
          resolution FLOAT,
          multiplicity FLOAT,
          Isigma FLOAT,
          completeness FLOAT,
          FOREIGN KEY (pdb_id_id) REFERENCES pdb_id(id)
      );
      CREATE TABLE homologue_stats (
          homologue_name_id INTEGER,
          gesamt_length INTEGER,
          gesamt_qscore FLOAT,
          gesamt_rmsd FLOAT,
          gesamt_seqid FLOAT,
          prosmart_length_number INTEGER,
          prosmart_rmsd FLOAT,
          prosmart_seqid FLOAT,
          prosmart_procrustes FLOAT,
          prosmart_flexible FLOAT,
          prosmart_cluster0_fragments FLOAT,
          prosmart_cluster0_mean_cos_theta FLOAT,
          prosmart_cluster0_sd_cos_theta FLOAT,
          prosmart_cluster1_fragments FLOAT,
          prosmart_cluster1_mean_cos_theta FLOAT,
          prosmart_cluster1_sd_cos_theta FLOAT,
          initial_rfree_afterSSM0 FLOAT,
          final_rfree_afterSSM0 FLOAT,
          initial_rwork_afterSSM0 FLOAT,
          final_rwork_afterSSM0 FLOAT,
          mean_phase_error_afterSSM0 FLOAT,
          f_map_correlation_afterSSM0 FLOAT,
          num_fragments_afterSSM0 FLOAT,
          num_res_built_afterSSM0 FLOAT,
          num_res_sequenced_afterSSM0 FLOAT,
          num_res_unique_afterSSM0 FLOAT,
          longest_fragments_afterSSM0 FLOAT,
          percent_chain_complete_afterSSM0 FLOAT,
          percent_res_complete_afterSSM0 FLOAT,
          initial_rfree_refmac0_afterSSM0_Buccaneer FLOAT,
          final_rfree_refmac0_afterSSM0_Buccaneer FLOAT,
          initial_rwork_refmac0_afterSSM0_Buccaneer FLOAT,
          final_rwork_refmac0_afterSSM0_Buccaneer FLOAT,
          mean_phase_error_afterSSM0_Buccaneer_refmac0 FLOAT,
          f_map_correlation_afterSSM0_Buccaneer_refmac0 FLOAT,
          initial_rfree_refmac_afterSSM0_Buccaneer FLOAT,
          final_rfree_refmac_afterSSM0_Buccaneer FLOAT,
          initial_rwork_refmac_afterSSM0_Buccaneer FLOAT,
          final_rwork_refmac_afterSSM0_Buccaneer FLOAT,
          mean_phase_error_afterSSM0_Buccaneer_refmac FLOAT,
          f_map_correlation_afterSSM0_Buccaneer_refmac FLOAT,
          initial_rfree_afterSSM FLOAT,
          final_rfree_afterSSM FLOAT,
          initial_rwork_afterSSM FLOAT,
          final_rwork_afterSSM FLOAT,
          mean_phase_error_afterSSM FLOAT,
          f_map_correlation_afterSSM FLOAT,
          num_fragments_afterSSM FLOAT,
          num_res_built_afterSSM FLOAT,
          num_res_sequenced_afterSSM FLOAT,
          num_res_unique_afterSSM FLOAT,
          longest_fragments_afterSSM FLOAT,
          initial_rfree_refmac0_afterSSM_Buccaneer FLOAT,
          percent_chain_complete_afterSSM FLOAT,
          percent_res_complete_afterSSM FLOAT,
          final_rfree_refmac0_afterSSM_Buccaneer FLOAT,
          initial_rwork_refmac0_afterSSM_Buccaneer FLOAT,
          final_rwork_refmac0_afterSSM_Buccaneer FLOAT,
          mean_phase_error_afterSSM_Buccaneer_refmac0 FLOAT,
          f_map_correlation_afterSSM_Buccaneer_refmac0 FLOAT,
          initial_rfree_refmac_afterSSM_Buccaneer FLOAT,
          final_rfree_refmac_afterSSM_Buccaneer FLOAT,
          initial_rwork_refmac_afterSSM_Buccaneer FLOAT,
          final_rwork_refmac_afterSSM_Buccaneer FLOAT,
          mean_phase_error_afterSSM_Buccaneer_refmac FLOAT,
          f_map_correlation_afterSSM_Buccaneer_refmac FLOAT,
          molrep_TF_sig FLOAT,
          molrep_contrast FLOAT,
          molrep_corrD FLOAT,
          molrep_corrF FLOAT,
          molrep_final_cc FLOAT,
          molrep_packing_coeff FLOAT,
          initial_rfree_afterMolrep0 FLOAT,
          final_rfree_afterMolrep0 FLOAT,
          initial_rwork_afterMolrep0 FLOAT,
          final_rwork_afterMolrep0 FLOAT,
          mean_phase_error_afterMolrep0 FLOAT,
          f_map_correlation_afterMolrep0 FLOAT,
          num_fragments_afterMolrep0 FLOAT,
          num_res_built_afterMolrep0 FLOAT,
          num_res_sequenced_afterMolrep0 FLOAT,
          num_res_unique_afterMolrep0 FLOAT,
          longest_fragments_afterMolrep0 FLOAT,
          percent_chain_complete_afterMolrep0 FLOAT,
          percent_res_complete_afterMolrep0 FLOAT,
          initial_rfree_refmac0_afterMolrep0_Buccaneer FLOAT,
          final_rfree_refmac0_afterMolrep0_Buccaneer FLOAT,
          initial_rwork_refmac0_afterMolrep0_Buccaneer FLOAT,
          final_rwork_refmac0_afterMolrep0_Buccaneer FLOAT,
          mean_phase_error_afterMolrep0_Buccaneer_refmac0 FLOAT,
          f_map_correlation_afterMolrep0_Buccaneer_refmac0 FLOAT,
          initial_rfree_refmac_afterMolrep0_Buccaneer FLOAT,
          final_rfree_refmac_afterMolrep0_Buccaneer FLOAT,
          initial_rwork_refmac_afterMolrep0_Buccaneer FLOAT,
          final_rwork_refmac_afterMolrep0_Buccaneer FLOAT,
          mean_phase_error_afterMolrep0_Buccaneer_refmac FLOAT,
          f_map_correlation_afterMolrep0_Buccaneer_refmac FLOAT,
          initial_rfree_afterMolrep FLOAT,
          final_rfree_afterMolrep FLOAT,
          initial_rwork_afterMolrep FLOAT,
          final_rwork_afterMolrep FLOAT,
          mean_phase_error_afterMolrep FLOAT,
          f_map_correlation_afterMolrep FLOAT,
          num_fragments_afterMolrep FLOAT,
          num_res_built_afterMolrep FLOAT,
          num_res_sequenced_afterMolrep FLOAT,
          num_res_unique_afterMolrep FLOAT,
          longest_fragments_afterMolrep FLOAT,
          percent_chain_complete_afterMolrep FLOAT,
          percent_res_complete_afterMolrep FLOAT,
          initial_rfree_refmac0_afterMolrep_Buccaneer FLOAT,
          final_rfree_refmac0_afterMolrep_Buccaneer FLOAT,
          initial_rwork_refmac0_afterMolrep_Buccaneer FLOAT,
          final_rwork_refmac0_afterMolrep_Buccaneer FLOAT,
          mean_phase_error_afterMolrep_Buccaneer_refmac0 FLOAT,
          f_map_correlation_afterMolrep_Buccaneer_refmac0 FLOAT,
          initial_rfree_refmac_afterMolrep_Buccaneer FLOAT,
          final_rfree_refmac_afterMolrep_Buccaneer FLOAT,
          initial_rwork_refmac_afterMolrep_Buccaneer FLOAT,
          final_rwork_refmac_afterMolrep_Buccaneer FLOAT,
          mean_phase_error_afterMolrep_Buccaneer_refmac FLOAT,
          f_map_correlation_afterMolrep_Buccaneer_refmac FLOAT,
          phaser_ellg FLOAT,
          phaser_llg FLOAT,
          phaser_rmsd FLOAT,
          initial_rfree_afterMR0 FLOAT,
          final_rfree_afterMR0 FLOAT,
          initial_rwork_afterMR0 FLOAT,
          final_rwork_afterMR0 FLOAT,
          mean_phase_error_afterMR0 FLOAT,
          f_map_correlation_afterMR0 FLOAT,
          num_fragments_afterMR0 FLOAT,
          num_res_built_afterMR0 FLOAT,
          num_res_sequenced_afterMR0 FLOAT,
          num_res_unique_afterMR0 FLOAT,
          longest_fragments_afterMR0 FLOAT,
          percent_chain_complete_afterMR0 FLOAT,
          percent_res_complete_afterMR0 FLOAT,
          initial_rfree_refmac0_afterMR0_Buccaneer FLOAT,
          final_rfree_refmac0_afterMR0_Buccaneer FLOAT,
          initial_rwork_refmac0_afterMR0_Buccaneer FLOAT,
          final_rwork_refmac0_afterMR0_Buccaneer FLOAT,
          mean_phase_error_afterMR0_Buccaneer_refmac0 FLOAT,
          f_map_correlation_afterMR0_Buccaneer_refmac0 FLOAT,
          initial_rfree_refmac_afterMR0_Buccaneer FLOAT,
          final_rfree_refmac_afterMR0_Buccaneer FLOAT,
          initial_rwork_refmac_afterMR0_Buccaneer FLOAT,
          final_rwork_refmac_afterMR0_Buccaneer FLOAT,
          mean_phase_error_afterMR0_Buccaneer_refmac FLOAT,
          f_map_correlation_afterMR0_Buccaneer_refmac FLOAT,
          initial_rfree_afterMR FLOAT,
          final_rfree_afterMR FLOAT,
          initial_rwork_afterMR FLOAT,
          final_rwork_afterMR FLOAT,
          mean_phase_error_afterMR FLOAT,
          f_map_correlation_afterMR FLOAT,
          num_fragments_afterMR FLOAT,
          num_res_built_afterMR FLOAT,
          num_res_sequenced_afterMR FLOAT,
          num_res_unique_afterMR FLOAT,
          longest_fragments_afterMR FLOAT,
          percent_chain_complete_afterMR FLOAT,
          percent_res_complete_afterMR FLOAT,
          initial_rfree_refmac0_afterMR_Buccaneer FLOAT,
          final_rfree_refmac0_afterMR_Buccaneer FLOAT,
          initial_rwork_refmac0_afterMR_Buccaneer FLOAT,
          final_rwork_refmac0_afterMR_Buccaneer FLOAT,
          mean_phase_error_afterMR_Buccaneer_refmac0 FLOAT,
          f_map_correlation_afterMR_Buccaneer_refmac0 FLOAT,
          initial_rfree_refmac_afterMR_Buccaneer FLOAT,
          final_rfree_refmac_afterMR_Buccaneer FLOAT,
          initial_rwork_refmac_afterMR_Buccaneer FLOAT,
          final_rwork_refmac_afterMR_Buccaneer FLOAT,
          mean_phase_error_afterMR_Buccaneer_refmac FLOAT,
          f_map_correlation_afterMR_Buccaneer_refmac FLOAT,
          mr_success_lable FLOAT,
          refinement_success_lable FLOAT,
          FOREIGN KEY (homologue_name_id) REFERENCES homologue_name(id)
      );
      ''')
