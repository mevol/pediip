#!/usr/bin/env python3

import argparse
import Bio.Seq
import Bio.SeqIO
import Bio.SeqRecord
import datetime
import gemmi
import glob
import gzip
import modules.create_mr_set.utils.models as models
import os
import modules.create_mr_set.utils.pdbtools as pdbtools
import random
import modules.create_mr_set.utils.rcsb as rcsb
import sys
import modules.create_mr_set.tasks.tasks as tasks
import urllib.request
import modules.create_mr_set.utils.utils as utils
import uuid
import xml.etree.ElementTree as ET
import shutil
import re

## MR

def path_coords(pdb_id, args):
  pdb = pdb_id.lower()
  return os.path.join(args.pdb_coords, pdb[1:3], "%s_final.pdb" % pdb)

# Superpose search model onto target using GESAMT
def superpose_homologue(key, homologue, args):
  xyzin1 = homologue.chain.structure.path("refmac.pdb")
  chain1 = homologue.chain.id
  xyzin2 = path_coords(homologue.hit_pdb, args)
  chain2 = homologue.hit_chain
  prefix = homologue.path("gesamt")
  result = tasks.superpose(xyzin1, chain1, xyzin2, chain2, prefix)
  homologue.jobs["gesamt"] = result
  if "ERROR" not in result and "qscore" in result:
    homologue.add_metadata("gesamt_qscore", result["qscore"])
    homologue.add_metadata("gesamt_rmsd", result["rmsd"])
    homologue.add_metadata("gesamt_length", result["length"])
    homologue.add_metadata("gesamt_seqid", result["seqid"])
  return key, homologue

# Superpose search model onto target using prosmart
def superpose_prosmart(key, homologue, args):
  xyzin1 = homologue.chain.structure.path("refmac.pdb")
  chain1 = homologue.chain.id
  xyzin2 = path_coords(homologue.hit_pdb, args)
  chain2 = homologue.hit_chain
  prefix = homologue.path("prosmart")
  result = tasks.superpose_like_mr(xyzin1, chain1, xyzin2, chain2, prefix)
  homologue.jobs["prosmart"] = result
  #print(result)
  if "ERROR" not in result and "rmsd" in result:
    homologue.add_metadata("prosmart_rmsd", result["rmsd"])
    homologue.add_metadata("prosmart_length_number", result["alignment_length"])
    homologue.add_metadata("prosmart_seqid", result["seqid"])
    xyz_path = "prosmart/Output_Files/Superposition/PDB_files/refmac_*_*_final_*/*Cluster0.pdb"
    homologue_dir = homologue.path(xyz_path)
    current = os.getcwd()
    combined = os.path.join(current, homologue_dir)
    try:
      os.path.exists(glob.glob(combined)[0])
    except IndexError:
      pass
    else:
      pro_out = glob.glob(combined)[0]
      prosmart_file = homologue.path("prosmart.pdb")
      with open(xyzin1, "r") as target_pdb:
        for line in target_pdb:
          if "CRYST1" in line:
            cryst_card = line
      with open(pro_out, "r") as out:
        content = out.read()
      with open(prosmart_file, "a") as to_add:
        to_add.writelines(cryst_card)
        to_add.writelines(content)
  return key, homologue

# Superpose search model onto target using Molrep
def superpose_molrep(key, homologue, args):
  hit_coords = path_coords(homologue.hit_pdb, args)
  xyzin1 = hit_coords
  xyzin2 = homologue.chain.structure.path("refmac.pdb")
  prefix = homologue.path("")
  result = tasks.superpose_molrep(xyzin1, xyzin2, prefix)
  homologue.jobs["molrep"] = result
  if "ERROR" not in result:
    homologue.add_metadata("molrep_corrF", result["corrF"])
    homologue.add_metadata("molrep_corrD", result["corrD"])
    homologue.add_metadata("molrep_TF_sig", result["TF_sig"])
    homologue.add_metadata("molrep_final_cc", result["final_cc"])
    homologue.add_metadata("molrep_packing_coeff", result["packing_coeff"])
    homologue.add_metadata("molrep_contrast", result["contrast"])
  if os.path.exists(homologue.path("molrep.pdb")):
    molrep_file = homologue.path("molrep.pdb")        
    with open(xyzin2) as target_pdb:
      for line in target_pdb:
        if "CRYST1" in line:
          cryst_card = line
    with open(molrep_file, "r") as m1_file:
      for line in m1_file:
        if "CRYST1" in line:
          m_cryst_card = line          
    with open(molrep_file, "r") as m2_file:
      content = m2_file.read()
      content = content.replace(str(m_cryst_card), str(cryst_card))      
    new_molrep_file = homologue.path("molrep_CRYST1_replace.pdb")
    with open(new_molrep_file, "w") as out_file:
      out_file.write(content)
  return key, homologue

#run Sculptor to edit MR search model
def prepare_sculptor_alignment(key, homologue, args):
  if os.path.exists(homologue.path("gesamt.seq")):
    seqin = homologue.path("gesamt.seq")
    seqout = homologue.path("sculptor.aln")
    records = list(Bio.SeqIO.parse(seqin, "fasta"))
    for record in records:
      record.seq = Bio.Seq.Seq(str(record.seq).upper())
    Bio.SeqIO.write(records, seqout, "clustal")
  return key, homologue

def trim_model(key, homologue, args):
  model = path_coords(homologue.hit_pdb, args)
  chain = homologue.hit_chain
  alignment = homologue.path("sculptor.aln")
  prefix = homologue.path("sculptor")
  result = tasks.trim_model(model, chain, alignment, prefix)
  homologue.jobs["sculptor"] = result
  return key, homologue

#run MR using Sculptor model
def mr(key, homologue, args):
  hklin = homologue.chain.structure.path("refmac.mtz")
  try:
    os.path.exists(glob.glob(homologue.path("sculptor*.pdb"))[0])
  except IndexError:
    pass
  else:  
    xyzin = glob.glob(homologue.path("sculptor*.pdb"))[0]
    identity = homologue.metadata["gesamt_seqid"]
    prefix = homologue.path("phaser")
    copies = homologue.chain.metadata["copies"]
    atom_counts = pdbtools.count_elements(homologue.chain.structure.path("refmac.pdb"))
    result = tasks.mr(hklin, xyzin, identity, prefix, copies, atom_counts)
    homologue.jobs["phaser"] = result
    if "llg" in result:
      homologue.add_metadata("phaser_llg", result["llg"])
      homologue.add_metadata("phaser_rmsd", result["rmsd"])
  return key, homologue

#refine MR solution using 100 cycles jelly body
def refine_placed_model_jelly(key, homologue, args):
  hklin = homologue.chain.structure.path("refmac.mtz")
  xyzin = homologue.path("phaser.1.pdb")
  prefix = homologue.path("refmac_afterMR")
  result = tasks.refine_jelly(hklin, xyzin, prefix)
  homologue.jobs["refmac"] = result
  if "final_rfree" in result:
    homologue.add_metadata("final_rfree_afterMR", result["final_rfree"])
    homologue.add_metadata("final_rwork_afterMR", result["final_rwork"])
    homologue.add_metadata("initial_rfree_afterMR", result["initial_rfree"])
    homologue.add_metadata("initial_rwork_afterMR", result["initial_rwork"])
  return key, homologue

#refine MR solution using 0 cycles
def refine_placed_model_zero(key, homologue, args):
  hklin = homologue.chain.structure.path("refmac.mtz")
  xyzin = homologue.path("phaser.1.pdb")
  prefix = homologue.path("refmac_afterMR0")
  result = tasks.refine_zero(hklin, xyzin, prefix)
  homologue.jobs["refmac"] = result 
  if "final_rfree" in result:
    homologue.add_metadata("final_rfree_afterMR0", result["final_rfree"])
    homologue.add_metadata("final_rwork_afterMR0", result["final_rwork"])
    homologue.add_metadata("initial_rfree_afterMR0", result["initial_rfree"])
    homologue.add_metadata("initial_rwork_afterMR0", result["initial_rwork"])
  return key, homologue

#refine prosmart SSM solution using 100 cycles jelly body
def refine_ssm_model_jelly(key, homologue, args):
  hklin = homologue.chain.structure.path("refmac.mtz")
  xyzin = homologue.path("prosmart.pdb")
  prefix = homologue.path("refmac_afterSSM")
  result = tasks.refine_jelly(hklin, xyzin, prefix)
  homologue.jobs["refmac"] = result
  if "final_rfree" in result:
    homologue.add_metadata("final_rfree_afterSSM", result["final_rfree"])
    homologue.add_metadata("final_rwork_afterSSM", result["final_rwork"])
    homologue.add_metadata("initial_rfree_afterSSM", result["initial_rfree"])
    homologue.add_metadata("initial_rwork_afterSSM", result["initial_rwork"])
  return key, homologue

#refine prosmart SSM solution using 0 cycles
def refine_ssm_model_zero(key, homologue, args):
  hklin = homologue.chain.structure.path("refmac.mtz")
  xyzin = homologue.path("prosmart.pdb")
  prefix = homologue.path("refmac_afterSSM0")
  result = tasks.refine_zero(hklin, xyzin, prefix)
  homologue.jobs["refmac"] = result
  if "final_rfree" in result:
    homologue.add_metadata("final_rfree_afterSSM0", result["final_rfree"])
    homologue.add_metadata("final_rwork_afterSSM0", result["final_rwork"])
    homologue.add_metadata("initial_rfree_afterSSM0", result["initial_rfree"])
    homologue.add_metadata("initial_rwork_afterSSM0", result["initial_rwork"])
  return key, homologue

#refine Molrep SSM solution using 100 cycles jelly body
def refine_molrep_model_jelly(key, homologue, args):
  hklin = homologue.chain.structure.path("refmac.mtz")
  xyzin = homologue.path("molrep_CRYST1_replace.pdb")
  prefix = homologue.path("refmac_afterMolrep")
  result = tasks.refine_jelly(hklin, xyzin, prefix)
  homologue.jobs["refmac"] = result
  if "final_rfree" in result:
    homologue.add_metadata("final_rfree_afterMolrep", result["final_rfree"])
    homologue.add_metadata("final_rwork_afterMolrep", result["final_rwork"])
    homologue.add_metadata("initial_rfree_afterMolrep", result["initial_rfree"])
    homologue.add_metadata("initial_rwork_afterMolrep", result["initial_rwork"])
  return key, homologue

#refine Molrep SSM solution using 0 cycles
def refine_molrep_model_zero(key, homologue, args):
  hklin = homologue.chain.structure.path("refmac.mtz")
  xyzin = homologue.path("molrep_CRYST1_replace.pdb")
  prefix = homologue.path("refmac_afterMolrep0")
  result = tasks.refine_zero(hklin, xyzin, prefix)
  homologue.jobs["refmac"] = result
  if "final_rfree" in result:
    homologue.add_metadata("final_rfree_afterMolrep0", result["final_rfree"])
    homologue.add_metadata("final_rwork_afterMolrep0", result["final_rwork"])
    homologue.add_metadata("initial_rfree_afterMolrep0", result["initial_rfree"])
    homologue.add_metadata("initial_rwork_afterMolrep0", result["initial_rwork"])
  return key, homologue

#combine phases PDB-redo target MR_jelly_body
def write_combined_mtz_afterMR_jelly(key, homologue, args):
  prefix = homologue.path("gemmijoin_afterMR")
  if not os.path.exists(homologue.path("refmac_afterMR.mtz")):
    pass
  else:    
    result = tasks.combine_mtz(prefix, [
      (homologue.chain.structure.path("refmac.mtz")),
      (homologue.path("refmac_afterMR.mtz"))])
    homologue.jobs["gemmijoin"] = result
  return key, homologue

#compare phases PDB-redo target MR_jelly_body
def compare_phases_afterMR_jelly(key, homologue, args):
  hklin = homologue.path("gemmijoin_afterMR.mtz")
  fo = "FP,SIGFP"
  wrk_hl = "PHWT_model,FOM_model"
  ref_hl = "PHWT_ref,FOM_ref"
  prefix = homologue.path("cphasematch_afterMR")
  result = tasks.compare_phases(hklin, fo, wrk_hl, ref_hl, prefix)
  homologue.jobs["cphasematch"] = result
  if "mean_phase_error" in result:
    homologue.add_metadata("mean_phase_error_afterMR", result["mean_phase_error"])
    homologue.add_metadata("f_map_correlation_afterMR", result["f_map_correlation"])
  return key, homologue

#combine phases PDB-redo target MR_zero
def write_combined_mtz_afterMR_zero(key, homologue, args):
  prefix = homologue.path("gemmijoin_afterMR0")
  if not os.path.exists(homologue.path("refmac_afterMR0.mtz")):
    pass
  else:  
    result = tasks.combine_mtz(prefix, [
      (homologue.chain.structure.path("refmac.mtz")),
      (homologue.path("refmac_afterMR0.mtz"))])
    homologue.jobs["gemmijoin"] = result
  return key, homologue

#compare phases PDB-redo target MR_zero
def compare_phases_afterMR_zero(key, homologue, args):
  hklin = homologue.path("gemmijoin_afterMR0.mtz")
  fo = "FP,SIGFP"
  wrk_hl = "PHWT_model,FOM_model"
  ref_hl = "PHWT_ref,FOM_ref"
  prefix = homologue.path("cphasematch_afterMR0")
  result = tasks.compare_phases(hklin, fo, wrk_hl, ref_hl, prefix)
  homologue.jobs["cphasematch"] = result
  if "mean_phase_error" in result:
    homologue.add_metadata("mean_phase_error_afterMR0", result["mean_phase_error"])
    homologue.add_metadata("f_map_correlation_afterMR0", result["f_map_correlation"])
  return key, homologue

#combine phases PDB-redo target prosmart_SSM_jelly_body
def write_combined_mtz_afterSSM_jelly(key, homologue, args):
  prefix = homologue.path("gemmijoin_afterSSM")
  if not os.path.exists(homologue.path("refmac_afterSSM.mtz")):
    pass
  else:  
    result = tasks.combine_mtz(prefix, [
      (homologue.chain.structure.path("refmac.mtz")),
      (homologue.path("refmac_afterSSM.mtz"))])
    homologue.jobs["gemmijoin"] = result
  return key, homologue

#compare phases PDB-redo target prosmart_SSM_jelly_body
def compare_phases_afterSSM_jelly(key, homologue, args):
  hklin = homologue.path("gemmijoin_afterSSM.mtz")
  fo = "FP,SIGFP"
  wrk_hl = "PHWT_model,FOM_model"
  ref_hl = "PHWT_ref,FOM_ref"
  prefix = homologue.path("cphasematch_afterSSM")
  result = tasks.compare_phases(hklin, fo, wrk_hl, ref_hl, prefix)
  homologue.jobs["cphasematch"] = result
  if "mean_phase_error" in result:
    homologue.add_metadata("mean_phase_error_afterSSM", result["mean_phase_error"])
    homologue.add_metadata("f_map_correlation_afterSSM", result["f_map_correlation"])
  return key, homologue

#combine phases PDB-redo target prosmart_SSM_zero
def write_combined_mtz_afterSSM_zero(key, homologue, args):
  prefix = homologue.path("gemmijoin_afterSSM0")
  if not os.path.exists(homologue.path("refmac_afterSSM0.mtz")):
    pass
  else:  
    result = tasks.combine_mtz(prefix, [
      (homologue.chain.structure.path("refmac.mtz")),
      (homologue.path("refmac_afterSSM0.mtz"))])
    homologue.jobs["gemmijoin"] = result
  return key, homologue

#compare phases PDB-redo target prosmart_SSM_zero
def compare_phases_afterSSM_zero(key, homologue, args):
  hklin = homologue.path("gemmijoin_afterSSM0.mtz")
  fo = "FP,SIGFP"
  wrk_hl = "PHWT_model,FOM_model"
  ref_hl = "PHWT_ref,FOM_ref"
  prefix = homologue.path("cphasematch_afterSSM0")
  result = tasks.compare_phases(hklin, fo, wrk_hl, ref_hl, prefix)
  homologue.jobs["cphasematch"] = result
  if "mean_phase_error" in result:
    homologue.add_metadata("mean_phase_error_afterSSM0", result["mean_phase_error"])
    homologue.add_metadata("f_map_correlation_afterSSM0", result["f_map_correlation"])
  return key, homologue

#combine phases PDB-redo target prosmart_Molrep_jelly_body
def write_combined_mtz_afterMolrep_jelly(key, homologue, args):
  prefix = homologue.path("gemmijoin_afterMolrep")
  if not os.path.exists(homologue.path("refmac_afterMolrep.mtz")):
    pass
  else:  
    result = tasks.combine_mtz(prefix, [
      (homologue.chain.structure.path("refmac.mtz")),
      (homologue.path("refmac_afterMolrep.mtz"))])
    homologue.jobs["gemmijoin"] = result
  return key, homologue

#compare phases PDB-redo target prosmart_Molrep_jelly_body
def compare_phases_afterMolrep_jelly(key, homologue, args):
  hklin = homologue.path("gemmijoin_afterMolrep.mtz")
  fo = "FP,SIGFP"
  wrk_hl = "PHWT_model,FOM_model"
  ref_hl = "PHWT_ref,FOM_ref"
  prefix = homologue.path("cphasematch_afterMolrep")
  result = tasks.compare_phases(hklin, fo, wrk_hl, ref_hl, prefix)
  homologue.jobs["cphasematch"] = result
  if "mean_phase_error" in result:
    homologue.add_metadata("mean_phase_error_afterMolrep", result["mean_phase_error"])
    homologue.add_metadata("f_map_correlation_afterMolrep", result["f_map_correlation"])
  return key, homologue

#combine phases PDB-redo target prosmart_Molrep_zero
def write_combined_mtz_afterMolrep_zero(key, homologue, args):
  prefix = homologue.path("gemmijoin_afterMolrep0")
  if not os.path.exists(homologue.path("refmac_afterMolrep0.mtz")):
    pass
  else:  
    result = tasks.combine_mtz(prefix, [
      (homologue.chain.structure.path("refmac.mtz")),
      (homologue.path("refmac_afterMolrep0.mtz"))])
    homologue.jobs["gemmijoin"] = result
  return key, homologue

#compare phases PDB-redo target prosmart_Molrep_zero
def compare_phases_afterMolrep_zero(key, homologue, args):
  hklin = homologue.path("gemmijoin_afterMolrep0.mtz")
  fo = "FP,SIGFP"
  wrk_hl = "PHWT_model,FOM_model"
  ref_hl = "PHWT_ref,FOM_ref"
  prefix = homologue.path("cphasematch_afterMolrep0")
  result = tasks.compare_phases(hklin, fo, wrk_hl, ref_hl, prefix)
  homologue.jobs["cphasematch"] = result
  if "mean_phase_error" in result:
    homologue.add_metadata("mean_phase_error_afterMolrep0", result["mean_phase_error"])
    homologue.add_metadata("f_map_correlation_afterMolrep0", result["f_map_correlation"])
  return key, homologue

#build MR solution after 0-cycles jelly body
def buccaneer_mr_after_refmac_zero(key, homologue, args):
  hklin = homologue.path("refmac_afterMR0.mtz")
  xyzin = homologue.path("refmac_afterMR0.pdb")
  fo = "FP,SIGFP"
  wrk_hl = "PHWT,FOM"
  seqin = glob.glob(homologue.path(os.path.join(
                                   "prosmart/Output_Files/Sequence", "refmac*.txt")))[0]
  prefix = homologue.path("buccaneer_afterMR0")
  result = tasks.buccaneer(hklin, xyzin, fo, wrk_hl, seqin, prefix)
  #result = tasks.buccaneer(hklin, xyzin, fo, wrk_hl, prefix)

  homologue.jobs["buccaneer"] = result
  if "final_rfree" in result:
    homologue.add_metadata("final_rfree_buccaneer_afterMR0", result["final_rfree"])
    homologue.add_metadata("final_rwork_buccaneer_afterMR0", result["final_rwork"])
    homologue.add_metadata("initial_rfree_buccaneer_afterMR0", result["initial_rfree"])
    homologue.add_metadata("initial_rwork_buccaneer_afterMR0", result["initial_rwork"])
  return key, homologue

#build MR solution after 100-cycles jelly body
def buccaneer_mr_after_refmac_jelly(key, homologue, args):
  hklin = homologue.path("refmac_afterMR.mtz")
  xyzin = homologue.path("refmac_afterMR.pdb")
  fo = "FP,SIGFP"
  wrk_hl = "PHWT,FOM"
  seqin = glob.glob(homologue.path(os.path.join(
                                   "prosmart/Output_Files/Sequence", "refmac*.txt")))[0]
  prefix = homologue.path("buccaneer_afterMR")
  result = tasks.buccaneer(hklin, xyzin, fo, wrk_hl, seqin, prefix)
  #result = tasks.buccaneer(hklin, xyzin, fo, wrk_hl, prefix)

  homologue.jobs["buccaneer"] = result
  if "final_rfree" in result:
    homologue.add_metadata("final_rfree_buccaneer_afterMR", result["final_rfree"])
    homologue.add_metadata("final_rwork_buccaneer_afterMR", result["final_rwork"])
    homologue.add_metadata("initial_rfree_buccaneer_afterMR", result["initial_rfree"])
    homologue.add_metadata("initial_rwork_buccaneer_afterMR", result["initial_rwork"])
  return key, homologue

#build Molrep solution after 0-cycles jelly body
def buccaneer_molrep_after_refmac_zero(key, homologue, args):
  hklin = homologue.path("refmac_afterMolrep0.mtz")
  xyzin = homologue.path("refmac_afterMolrep0.pdb")
  fo = "FP,SIGFP"
  wrk_hl = "PHWT,FOM"
  seqin = glob.glob(homologue.path(os.path.join(
                                   "prosmart/Output_Files/Sequence", "refmac*.txt")))[0]
  prefix = homologue.path("buccaneer_afterMolrep0")
  result = tasks.buccaneer(hklin, xyzin, fo, wrk_hl, seqin, prefix)
  #result = tasks.buccaneer(hklin, xyzin, fo, wrk_hl, prefix)

  homologue.jobs["buccaneer"] = result
  if "final_rfree" in result:
    homologue.add_metadata("final_rfree_buccaneer_afterMolrep0", result["final_rfree"])
    homologue.add_metadata("final_rwork_buccaneer_afterMolrep0", result["final_rwork"])
    homologue.add_metadata("initial_rfree_buccaneer_afterMolrep0", result["initial_rfree"])
    homologue.add_metadata("initial_rwork_buccaneer_afterMolrep0", result["initial_rwork"])
  return key, homologue

#build Molrep solution after 100-cycles jelly body
def buccaneer_molrep_after_refmac_jelly(key, homologue, args):
  print(2222222)
  hklin = homologue.path("refmac_afterMolrep.mtz")
  xyzin = homologue.path("refmac_afterMolrep.pdb")
  fo = "FP,SIGFP"
  wrk_hl = "PHWT,FOM"
  seqin = glob.glob(homologue.path(os.path.join(
                                   "prosmart/Output_Files/Sequence", "refmac*.txt")))[0]
  prefix = homologue.path("buccaneer_afterMolrep")
  result = tasks.buccaneer(hklin, xyzin, fo, wrk_hl, seqin, prefix)
  #result = tasks.buccaneer(hklin, xyzin, fo, wrk_hl, prefix)

  homologue.jobs["buccaneer"] = result
  if "final_rfree" in result:
    homologue.add_metadata("final_rfree_buccaneer_afterMolrep", result["final_rfree"])
    homologue.add_metadata("final_rwork_buccaneer_afterMolrep", result["final_rwork"])
    homologue.add_metadata("initial_rfree_buccaneer_afterMolrep", result["initial_rfree"])
    homologue.add_metadata("initial_rwork_buccaneer_afterMolrep", result["initial_rwork"])
  return key, homologue

#####################################################################
# this is the original execution plan that works;
#def prepare_and_do_mr(homologues, args):
#  utils.print_section_title("Preparing Models")
#  utils.parallel("Superposing homologues for sequence alignments", superpose_homologue, homologues, args, args.jobs)
#  utils.parallel("Preparing alignments for sculptor", prepare_sculptor_alignment, homologues, args, args.jobs)
#  utils.parallel("Trimming input models with sculptor", trim_model, homologues, args, args.jobs)
#  
#  utils.print_section_title("SSM with Prosmart")
#  utils.parallel("Superpose target and model with prosmart", superpose_prosmart, homologues, args, args.jobs)
#  utils.parallel("Refining SSM models zero cycles", refine_ssm_model_zero, homologues, args, args.jobs)
#  utils.parallel("Combining SSM phases into a single MTZ file zero cycles", write_combined_mtz_afterSSM_zero, homologues, args, args.jobs)
#  utils.parallel("Comparing SSM phases with cphasematch", compare_phases_afterSSM_zero, homologues, args, args.jobs)
#  utils.parallel("Refining SSM models jelly body", refine_ssm_model_jelly, homologues, args, args.jobs)
#  utils.parallel("Combining SSM phases into a single MTZ file after jelly body", write_combined_mtz_afterSSM_jelly, homologues, args, args.jobs)
#  utils.parallel("Comparing SSM phases with cphasematch after jelly body", compare_phases_afterSSM_jelly, homologues, args, args.jobs)
#
#  utils.print_section_title("SSM with Molrep")
#  utils.parallel("Superpose target and model with Molrep", superpose_molrep, homologues, args, args.jobs)
#  utils.parallel("Refining Molrep models zero cycles", refine_molrep_model_zero, homologues, args, args.jobs)
#  utils.parallel("Combining Molrep phases into a single MTZ file zero cycles", write_combined_mtz_afterMolrep_zero, homologues, args, args.jobs)
#  utils.parallel("Comparing Molrep phases with cphasematch", compare_phases_afterMolrep_zero, homologues, args, args.jobs)
#  utils.parallel("Refining Molrep models jelly body", refine_molrep_model_jelly, homologues, args, args.jobs)
#  utils.parallel("Combining Molrep phases into a single MTZ file after jelly body", write_combined_mtz_afterMolrep_jelly, homologues, args, args.jobs)
#  utils.parallel("Comparing Molrep phases with cphasematch after jelly body", compare_phases_afterMolrep_jelly, homologues, args, args.jobs)
#
#
#  if not args.stop_before_mr:
#    utils.print_section_title("Performing Molecular Replacement")
#    utils.parallel("Performing molecular replacement with phaser", mr, homologues, args, int(args.jobs / 4))
#    utils.parallel("Refining placed models zero cycles", refine_placed_model_zero, homologues, args, args.jobs)
#    utils.parallel("Combining MR phases into a single MTZ file zero cycles", write_combined_mtz_afterMR_zero, homologues, args, args.jobs)
#    utils.parallel("Comparing MR phases with cphasematch zero cycles", compare_phases_afterMR_zero, homologues, args, args.jobs)
#    utils.parallel("Refining placed models jelly body", refine_placed_model_jelly, homologues, args, args.jobs)    
#    utils.parallel("Combining MR phases into a single MTZ file after jelly body", write_combined_mtz_afterMR_jelly, homologues, args, args.jobs)
#    utils.parallel("Comparing MR phases with cphasematch after jelly body", compare_phases_afterMR_jelly, homologues, args, args.jobs)
#  utils.remove_errors(homologues)
#  print("")

def run_mr_pipelines(key, homologue, args):
  if not os.path.exists(homologue.path("JOB_IS_DONE.txt")):
       
    superpose_homologue(key, homologue, args)
    prepare_sculptor_alignment(key, homologue, args)
    trim_model(key, homologue, args)
    
    superpose_prosmart(key, homologue, args)
    refine_ssm_model_zero(key, homologue, args)
    write_combined_mtz_afterSSM_zero(key, homologue, args)
    compare_phases_afterSSM_zero(key, homologue, args)
    refine_ssm_model_jelly(key, homologue, args)
    write_combined_mtz_afterSSM_jelly(key, homologue, args)
    compare_phases_afterSSM_jelly(key, homologue, args)
    
    superpose_molrep(key, homologue, args)
    refine_molrep_model_zero(key, homologue, args)
    write_combined_mtz_afterMolrep_zero(key, homologue, args)
    compare_phases_afterMolrep_zero(key, homologue, args)
    refine_molrep_model_jelly(key, homologue, args)
    write_combined_mtz_afterMolrep_jelly(key, homologue, args)
    compare_phases_afterMolrep_jelly(key, homologue, args)
    
    mr(key, homologue, args)
    refine_placed_model_zero(key, homologue, args)
    write_combined_mtz_afterMR_zero(key, homologue, args)
    compare_phases_afterMR_zero(key, homologue, args)
    refine_placed_model_jelly(key, homologue, args)
    write_combined_mtz_afterMR_jelly(key, homologue, args)
    compare_phases_afterMR_jelly(key, homologue, args)
    
    with open(homologue.path("JOB_IS_DONE.txt"), "w") as out_file:
      line = "job is done"
      out_file.writelines(line)
  else:
    print("MR and SSM already done.")
    pass

  if not os.path.exists(homologue.path("BUILD_WITH_BUCCANEER.TXT")):
    print("Starting Buccaneer")
    buccaneer_mr_after_refmac_zero(key, homologue, args)
    buccaneer_mr_after_refmac_jelly(key, homologue, args)
    buccaneer_molrep_after_refmac_zero(key, homologue, args)
    buccaneer_molrep_after_refmac_jelly(key, homologue, args)
    with open(homologue.path("BUILD_WITH_BUCCANEER.txt"), "w") as out_file:
      line = "job is done"
      out_file.writelines(line)

  return key, homologue
  

def prepare_and_do_mr(homologues, args):
  utils.print_section_title("Running set of MR pipelines")
  utils.parallel("Iterating over homologues", run_mr_pipelines, homologues, args, args.jobs)
  #utils.parallel("Iterating over homologues", run_mr_pipelines, homologues, args)

#def prepare_and_do_mr(homologues, args):
  #utils.print_section_title("Running set of MR pipelines")
  #utils.parallel("Superposing homologues for sequence alignments", superpose_homologue, homologues, args, args.jobs)
  #utils.parallel("Preparing alignments for sculptor", prepare_sculptor_alignment, homologues, args, args.jobs)
  #utils.parallel("Trimming input models with sculptor", trim_model, homologues, args, args.jobs)
  
  #utils.print_section_title("SSM with Prosmart")
  #utils.parallel("Superpose target and model with prosmart", superpose_prosmart, homologues, args, args.jobs)
  #utils.parallel("Refining SSM models zero cycles", refine_ssm_model_zero, homologues, args, args.jobs)
  #utils.parallel("Combining SSM phases into a single MTZ file zero cycles", write_combined_mtz_afterSSM_zero, homologues, args, args.jobs)
  #utils.parallel("Comparing SSM phases with cphasematch", compare_phases_afterSSM_zero, homologues, args, args.jobs)
  #utils.parallel("Refining SSM models jelly body", refine_ssm_model_jelly, homologues, args, args.jobs)
  #utils.parallel("Combining SSM phases into a single MTZ file after jelly body", write_combined_mtz_afterSSM_jelly, homologues, args, args.jobs)
  #utils.parallel("Comparing SSM phases with cphasematch after jelly body", compare_phases_afterSSM_jelly, homologues, args, args.jobs)

  #utils.print_section_title("SSM with Molrep")
  #utils.parallel("Superpose target and model with Molrep", superpose_molrep, homologues, args, args.jobs)
  #utils.parallel("Refining Molrep models zero cycles", refine_molrep_model_zero, homologues, args, args.jobs)
  #utils.parallel("Combining Molrep phases into a single MTZ file zero cycles", write_combined_mtz_afterMolrep_zero, homologues, args, args.jobs)
  #utils.parallel("Comparing Molrep phases with cphasematch", compare_phases_afterMolrep_zero, homologues, args, args.jobs)
  #utils.parallel("Refining Molrep models jelly body", refine_molrep_model_jelly, homologues, args, args.jobs)
  #utils.parallel("Combining Molrep phases into a single MTZ file after jelly body", write_combined_mtz_afterMolrep_jelly, homologues, args, args.jobs)
  #utils.parallel("Comparing Molrep phases with cphasematch after jelly body", compare_phases_afterMolrep_jelly, homologues, args, args.jobs)


  #if not args.stop_before_mr:
    #utils.print_section_title("Performing Molecular Replacement")
    #utils.parallel("Performing molecular replacement with phaser", mr, homologues, args, int(args.jobs / 4))
    #utils.parallel("Refining placed models zero cycles", refine_placed_model_zero, homologues, args, args.jobs)
    #utils.parallel("Combining MR phases into a single MTZ file zero cycles", write_combined_mtz_afterMR_zero, homologues, args, args.jobs)
    #utils.parallel("Comparing MR phases with cphasematch zero cycles", compare_phases_afterMR_zero, homologues, args, args.jobs)
    #utils.parallel("Refining placed models jelly body", refine_placed_model_jelly, homologues, args, args.jobs)    
    #utils.parallel("Combining MR phases into a single MTZ file after jelly body", write_combined_mtz_afterMR_jelly, homologues, args, args.jobs)
    #utils.parallel("Comparing MR phases with cphasematch after jelly body", compare_phases_afterMR_jelly, homologues, args, args.jobs)
  utils.remove_errors(homologues)
  print("")
