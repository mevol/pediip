#!/bin/env python3

import os
import json
import re
import pandas as pd
from modules.create_mr_set.utils.utils import ProgressBar
from itertools import islice

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
    cur = self.handle.cursor()

    target_pdb = homologue.split("_")[1]
    
    cur.execute('''
      SELECT id FROM pdb_id
      WHERE pdb_id="%s"
      ''' % (target_pdb))
    pdb_id = cur.fetchone()[0]
    
    homologue_pdb = homologue.split("_")[-2]
    homologue_chain = homologue.split("_")[-1]
    homologue_name = homologue_pdb+"_"+homologue_chain
    
    cur.execute('''
      INSERT OR IGNORE INTO homologue_name (homologue_name)
      VALUES ("%s")
      ''' % (homologue_name))
    
    cur.execute('''
      SELECT id FROM homologue_name
      WHERE homologue_name.homologue_name="%s"
      ''' % (homologue_name))
    homologue_pk = cur.fetchall()[-1][0]

    cur.execute('''
      UPDATE homologue_name
      SET pdb_id_id = "%s"
      WHERE id = "%s"
      ''' % (pdb_id, homologue_pk))    

    cur.executescript( '''
      INSERT OR IGNORE INTO homologue_stats (homologue_name_id)
      VALUES (%s);
      ''' % (homologue_pk))


#    h_main = homologue.replace("_", "/")
#    print("Main directory", h_main)
    
    main_dir = "structures/%s/chains/%s/homologues/" % (target_pdb, homologue_chain)
    
    h_dir = os.path.join(main_dir, str(homologue_name))   
    h_meta_json = os.path.join(h_dir, "metadata.json")
    phaser_log = os.path.join(h_dir, "phaser.log")
    prosmart_dir = os.path.join(h_dir, "prosmart")
    prosmart_log = os.path.join(prosmart_dir, "prosmart_align_logfile.txt")


    gesamt_length = 0
    gesamt_qscore = 0
    gesamt_seqid = 0
    gesamt_rmsd = 0
    prosmart_length_number = 0
    prosmart_rmsd = 0
    prosmart_seqid = 0
    initial_rfree_afterSSM0 = 0
    final_rfree_afterSSM0 = 0 
    initial_rwork_afterSSM0 = 0 
    final_rwork_afterSSM0 = 0
    mean_phase_error_afterSSM0 = 0
    f_map_correlation_afterSSM0 = 0
    initial_rfree_afterSSM = 0
    final_rfree_afterSSM = 0
    initial_rwork_afterSSM = 0
    final_rwork_afterSSM = 0
    mean_phase_error_afterSSM = 0
    f_map_correlation_afterSSM = 0
    molrep_TF_sig = 0
    molrep_contrast = 0
    molrep_corrD = 0
    molrep_corrF = 0
    molrep_final_cc = 0
    molrep_packing_coeff = 0
    initial_rfree_afterMolrep0 = 0
    final_rfree_afterMolrep0 = 0
    initial_rwork_afterMolrep0 = 0
    final_rwork_afterMolrep0 = 0
    mean_phase_error_afterMolrep0 = 0
    f_map_correlation_afterMolrep0 = 0
    initial_rfree_afterMolrep = 0
    final_rfree_afterMolrep = 0
    initial_rwork_afterMolrep = 0
    final_rwork_afterMolrep = 0
    mean_phase_error_afterMolrep = 0
    f_map_correlation_afterMolrep = 0
    phaser_ellg = 0
    phaser_llg = 0
    phaser_rmsd = 0
    initial_rfree_afterMR0 = 0
    final_rfree_afterMR0 = 0     
    initial_rwork_afterMR0 = 0
    final_rwork_afterMR0 = 0
    mean_phase_error_afterMR0 = 0
    f_map_correlation_afterMR0 = 0  
    initial_rfree_afterMR = 0
    final_rfree_afterMR = 0     
    initial_rwork_afterMR = 0
    final_rwork_afterMR = 0
    mean_phase_error_afterMR = 0
    f_map_correlation_afterMR = 0
    mr_success_lable = 3
    refinement_success_lable = 3
    procrustes = 0
    flexible = 0



    if os.path.exists(h_meta_json):
      with open(h_meta_json, "r") as h_json:
        h_reader = json.load(h_json)          
        if "gesamt_length" in h_reader:
          gesamt_length = h_reader["gesamt_length"]
#            else:
#              gesamt_length = 0
        if "gesamt_qscore" in h_reader:
          gesamt_qscore = h_reader["gesamt_qscore"]
#            else:
#              gesamt_qscore = 0
        if "gesamt_seqid" in h_reader:
          gesamt_seqid = h_reader["gesamt_seqid"]
#            else:
#              gesamt_seqid = 0
        if "gesamt_rmsd" in h_reader:
          gesamt_rmsd = h_reader["gesamt_rmsd"]
#            else:
#              gesamt_rmsd = 0            
        if "initial_rfree" in h_reader:
          initial_rfree = h_reader["initial_rfree"]
#            else:
#              initial_rfree = 0
        if  "prosmart_length_number" in h_reader:
          prosmart_length_number = h_reader["prosmart_length_number"]
        if  "prosmart_rmsd" in h_reader:
          prosmart_rmsd = h_reader["prosmart_rmsd"]
        if  "prosmart_seqid" in h_reader:
          prosmart_seqid = h_reader["prosmart_seqid"]
        if  "initial_rfree_afterSSM0" in h_reader:
          initial_rfree_afterSSM0 = h_reader["initial_rfree_afterSSM0"]
        if  "final_rfree_afterSSM0" in h_reader:
          final_rfree_afterSSM0 = h_reader["final_rfree_afterSSM0"]
        if  "initial_rwork_afterSSM0" in h_reader:
          initial_rwork_afterSSM0 = h_reader["initial_rwork_afterSSM0"]
        if  "final_rwork_afterSSM0" in h_reader:
          final_rwork_afterSSM0 = h_reader["final_rwork_afterSSM0"]
        if  "mean_phase_error_afterSSM0" in h_reader:
          mean_phase_error_afterSSM0 = h_reader["mean_phase_error_afterSSM0"]
        if  "f_map_correlation_afterSSM0" in h_reader:
          f_map_correlation_afterSSM0 = h_reader["f_map_correlation_afterSSM0"]
        if  "initial_rfree_afterSSM" in h_reader:
          initial_rfree_afterSSM = h_reader["initial_rfree_afterSSM"]
        if  "final_rfree_afterSSM" in h_reader:
          final_rfree_afterSSM = h_reader["final_rfree_afterSSM"]
        if  "initial_rwork_afterSSM" in h_reader:
          initial_rwork_afterSSM = h_reader["initial_rwork_afterSSM"]
        if  "final_rwork_afterSSM" in h_reader:
          final_rwork_afterSSM = h_reader["final_rwork_afterSSM"]
        if  "mean_phase_error_afterSSM" in h_reader:
          mean_phase_error_afterSSM = h_reader["mean_phase_error_afterSSM"]
        if  "f_map_correlation_afterSSM" in h_reader:
          f_map_correlation_afterSSM = h_reader["f_map_correlation_afterSSM"]
        if "molrep_TF_sig" in h_reader:
          molrep_TF_sig = h_reader["molrep_TF_sig"]
        if "molrep_contrast" in h_reader:
          molrep_contrast = h_reader["molrep_contrast"]
        if "molrep_corrD" in h_reader:
          molrep_corrD = h_reader["molrep_corrD"]
        if "molrep_corrF" in h_reader:
          molrep_corrF = h_reader["molrep_corrF"]
        if "molrep_final_cc" in h_reader:
          molrep_final_cc = h_reader["molrep_final_cc"]
        if "molrep_packing_coeff" in h_reader:
          molrep_packing_coeff = h_reader["molrep_packing_coeff"]
        if "initial_rfree_afterMolrep0" in h_reader:
          initial_rfree_afterMolrep0 = h_reader["initial_rfree_afterMolrep0"]
        if "final_rfree_afterMolrep0" in h_reader:
          final_rfree_afterMolrep0 = h_reader["final_rfree_afterMolrep0"]
        if "initial_rwork_afterMolrep0" in h_reader:
          initial_rwork_afterMolrep0 = h_reader["initial_rwork_afterMolrep0"]
        if "final_rwork_afterMolrep0" in h_reader:
          final_rwork_afterMolrep0 = h_reader["final_rwork_afterMolrep0"]
        if "mean_phase_error_afterMolrep0" in h_reader:
          mean_phase_error_afterMolrep0 = h_reader["mean_phase_error_afterMolrep0"]
        if "f_map_correlation_afterMolrep0" in h_reader:
          f_map_correlation_afterMolrep0 = h_reader["f_map_correlation_afterMolrep0"]
        if "initial_rfree_afterMolrep" in h_reader:
          initial_rfree_afterMolrep = h_reader["initial_rfree_afterMolrep"]
        if "final_rfree_afterMolrep" in h_reader:
          final_rfree_afterMolrep = h_reader["final_rfree_afterMolrep"]
        if "initial_rwork_afterMolrep" in h_reader:
          initial_rwork_afterMolrep = h_reader["initial_rwork_afterMolrep"]
        if "final_rwork_afterMolrep" in h_reader:
          final_rwork_afterMolrep = h_reader["final_rwork_afterMolrep"]
        if "mean_phase_error_afterMolrep" in h_reader:
          mean_phase_error_afterMolrep = h_reader["mean_phase_error_afterMolrep"]
        if "f_map_correlation_afterMolrep" in h_reader:
          f_map_correlation_afterMolrep = h_reader["f_map_correlation_afterMolrep"]
        if "phaser_llg" in h_reader:
          phaser_llg = h_reader["phaser_llg"]
        else:
          phaser_llg = 0.0            
        if phaser_llg is None:
          phaser_llg = 0.0  
        if "phaser_rmsd" in h_reader:
          phaser_rmsd = h_reader["phaser_rmsd"]
#            else:
#              phaser_rmsd = 0
        if phaser_rmsd is None:
          phaser_rmsd = 0
        if "initial_rfree_afterMR0" in h_reader:
          initial_rfree_afterMR0 = h_reader["initial_rfree_afterMR0"]
        if "final_rfree_afterMR0" in h_reader:
          final_rfree_afterMR0 = h_reader["final_rfree_afterMR0"]
        if "initial_rwork_afterMR0" in h_reader:
          initial_rwork_afterMR0 = h_reader["initial_rwork_afterMR0"]
        if "final_rwork_afterMR0" in h_reader:
          final_rwork_afterMR0 = h_reader["final_rwork_afterMR0"]
        if "mean_phase_error_afterMR0" in h_reader:
          mean_phase_error_afterMR0 = h_reader["mean_phase_error_afterMR0"]
        if "f_map_correlation_afterMR0" in h_reader:
          f_map_correlation_afterMR0 = h_reader["f_map_correlation_afterMR0"]
        if "initial_rfree_afterMR" in h_reader:
          initial_rfree_afterMR = h_reader["initial_rfree_afterMR"]
        if "final_rfree_afterMR" in h_reader:
          final_rfree_afterMR = h_reader["final_rfree_afterMR"]
        if "initial_rwork_afterMR" in h_reader:
          initial_rwork_afterMR = h_reader["initial_rwork_afterMR"]
        if "final_rwork_afterMR" in h_reader:
          final_rwork_afterMR = h_reader["final_rwork_afterMR"]
        if "mean_phase_error_afterMR" in h_reader:
          mean_phase_error_afterMR = h_reader["mean_phase_error_afterMR"]
        if "f_map_correlation_afterMR" in h_reader:
          f_map_correlation_afterMR = h_reader["f_map_correlation_afterMR"]

        if prosmart_rmsd > 0 or molrep_contrast > 0:
          mr_success_lable = 2
          if final_rfree_afterSSM <= 0.5 or final_rfree_afterMolrep <= 0.5:
            refinement_success_lable = "2a"
          if final_rfree_afterSSM > 0.5 or final_rfree_afterSSM == 0 or final_rfree_afterMolrep > 0.5 or final_rfree_afterMolrep == 0:
            refinement_success_lable = "2b"

        #if final_rfree <= initial_rfree and phaser_llg >= 60.0:
        if phaser_llg >= 60.0:  
          #mr_success_lable = 1
          mr_success_lable = 1
          if final_rfree_afterMR <= 0.5:
            refinement_success_lable = "1a"
          if final_rfree_afterMR > 0.5 or final_rfree_afterMR == 0:
            refinement_success_lable = "1b"          
          
          





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
          


    if os.path.exists(phaser_log):
      with open(phaser_log, "r") as p_log:
        for line in p_log:
          if line.rstrip() == "   eLLG: eLLG of chain alone":
            phaser_ellg = list(islice(p_log, 2))[1].split()[0]
    else:
      phaser_ellg = 0        

    if os.path.exists(prosmart_log):
      print(prosmart_log)
      with open(prosmart_log, "r") as pro_log:
        print("Opened PROSMART log")
        for line in pro_log:
          #print(line)
          #match = re.match(line, "Average residue scores:", 1)
          #print(match)
          if line.strip().startswith("Average residue scores:"):
            print(line)
            procrustes = list(islice(pro_log, 2))[0].split()#[0]
            print(6666666666666666666, procrustes)
            clusters = list(islice(pro_log, 2))[1].split()#[0]
            print(clusters)
            
    else:
      procrustes = 0        


    homologue_dict = {
        "gesamt_length"                  : gesamt_length,
        "gesamt_qscore"                  : gesamt_qscore,
        "gesamt_seqid"                   : gesamt_seqid,
        "gesamt_rmsd"                    : gesamt_rmsd,
        "prosmart_length_number"         : prosmart_length_number,
        "prosmart_rmsd"                  : prosmart_rmsd,
        "prosmart_seqid"                 : prosmart_seqid,
        "initial_rfree_afterSSM0"        : initial_rfree_afterSSM0,
        "final_rfree_afterSSM0"          : final_rfree_afterSSM0, 
        "initial_rwork_afterSSM0"        : initial_rwork_afterSSM0, 
        "final_rwork_afterSSM0"          : final_rwork_afterSSM0,
        "mean_phase_error_afterSSM0"     : mean_phase_error_afterSSM0,
        "f_map_correlation_afterSSM0"    : f_map_correlation_afterSSM0,
        "initial_rfree_afterSSM"         : initial_rfree_afterSSM,
        "final_rfree_afterSSM"           : final_rfree_afterSSM,
        "initial_rwork_afterSSM"         : initial_rwork_afterSSM,
        "final_rwork_afterSSM"           : final_rwork_afterSSM,
        "mean_phase_error_afterSSM"      : mean_phase_error_afterSSM,
        "f_map_correlation_afterSSM"     : f_map_correlation_afterSSM,
        "molrep_TF_sig"                  : molrep_TF_sig,
        "molrep_contrast"                : molrep_contrast,
        "molrep_corrD"                   : molrep_corrD,
        "molrep_corrF"                   : molrep_corrF,
        "molrep_final_cc"                : molrep_final_cc,
        "molrep_packing_coeff"           : molrep_packing_coeff,
        "initial_rfree_afterMolrep0"     : initial_rfree_afterMolrep0,
        "final_rfree_afterMolrep0"       : final_rfree_afterMolrep0,
        "initial_rwork_afterMolrep0"     : initial_rwork_afterMolrep0,
        "final_rwork_afterMolrep0"       : final_rwork_afterMolrep0,
        "mean_phase_error_afterMolrep0"  : mean_phase_error_afterMolrep0,
        "f_map_correlation_afterMolrep0" : f_map_correlation_afterMolrep0,
        "initial_rfree_afterMolrep"      : initial_rfree_afterMolrep,
        "final_rfree_afterMolrep"        : final_rfree_afterMolrep,
        "initial_rwork_afterMolrep"      : initial_rwork_afterMolrep,
        "final_rwork_afterMolrep"        : final_rwork_afterMolrep,
        "mean_phase_error_afterMolrep"   : mean_phase_error_afterMolrep,
        "f_map_correlation_afterMolrep"  : f_map_correlation_afterMolrep,
        "phaser_ellg"                    : phaser_ellg,
        "phaser_llg"                     : phaser_llg,
        "phaser_rmsd"                    : phaser_rmsd,
        "initial_rfree_afterMR0"         : initial_rfree_afterMR0,
        "final_rfree_afterMR0"           : final_rfree_afterMR0,     
        "initial_rwork_afterMR0"         : initial_rwork_afterMR0,
        "final_rwork_afterMR0"           : final_rwork_afterMR0,
        "mean_phase_error_afterMR0"      : mean_phase_error_afterMR0,
        "f_map_correlation_afterMR0"     : f_map_correlation_afterMR0,  
        "initial_rfree_afterMR"          : initial_rfree_afterMR,
        "final_rfree_afterMR"            : final_rfree_afterMR,
        "initial_rwork_afterMR"          : initial_rwork_afterMR,
        "final_rwork_afterMR"            : final_rwork_afterMR,
        "mean_phase_error_afterMR"       : mean_phase_error_afterMR,
        "f_map_correlation_afterMR"      : f_map_correlation_afterMR,
        "mr_success_lable"               : mr_success_lable,
        "refinement_success_lable"       : refinement_success_lable
                 }             

    for entry in homologue_dict:
      cur.execute('''
        UPDATE homologue_stats
        SET "%s" = "%s"
        WHERE homologue_name_id = "%s";
        ''' % (entry, homologue_dict[entry], homologue_pk))
  
    self.handle.commit()
