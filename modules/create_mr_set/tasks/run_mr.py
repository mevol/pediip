#!/usr/bin/env python3

import Bio.Seq
import Bio.SeqIO
import glob
import modules.create_mr_set.utils.models as models
import os
import modules.create_mr_set.utils.pdbtools as pdbtools
import modules.create_mr_set.utils.rcsb as rcsb
import modules.create_mr_set.tasks.tasks as tasks
import modules.create_mr_set.utils.utils as utils

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

##########################################################################################
# Build with Buccaneer after MR and first refinement
##########################################################################################

#build MR solution after 100-cycles jelly body
def buccaneer_mr_after_refmac_jelly(key, homologue, args):
  hklin = homologue.path("refmac_afterMR.mtz")
  xyzin = homologue.path("refmac_afterMR.pdb")
  fo = "FP,SIGFP"
  wrk_hl = "PHWT,FOM"
  seqin = homologue.chain.structure.path("unique.fasta")
  prefix = homologue.path("buccaneer_afterMR")
  result = tasks.buccaneer(hklin, xyzin, fo, wrk_hl, seqin, prefix)
  homologue.jobs["buccaneer"] = result
  if "num_res_built" in result:
    homologue.add_metadata("num_res_built_afterMR", result["num_res_built"])
    homologue.add_metadata("num_fragments_afterMR", result["num_fragments"])
    homologue.add_metadata("longest_fragments_afterMR", result["longest_fragments"])
    homologue.add_metadata("num_res_sequenced_afterMR", result["num_res_sequenced"])
    homologue.add_metadata("num_res_unique_afterMR", result["num_res_unique"])
    homologue.add_metadata("percent_res_complete_afterMR", result["percent_res_complete"])
    homologue.add_metadata("percent_chain_complete_afterMR", result["percent_chain_complete"])
  return key, homologue

##########################################################################################
# Build with Buccaneer after Molrep and first refinement
##########################################################################################

#build Molrep solution after 100-cycles jelly body
def buccaneer_molrep_after_refmac_jelly(key, homologue, args):
  hklin = homologue.path("refmac_afterMolrep.mtz")
  xyzin = homologue.path("refmac_afterMolrep.pdb")
  fo = "FP,SIGFP"
  wrk_hl = "PHWT,FOM"
  seqin = homologue.chain.structure.path("unique.fasta")
  prefix = homologue.path("buccaneer_afterMolrep")
  result = tasks.buccaneer(hklin, xyzin, fo, wrk_hl, seqin, prefix)
  homologue.jobs["buccaneer"] = result
  if "num_res_built" in result:
    homologue.add_metadata("num_res_built_afterMolrep", result["num_res_built"])
    homologue.add_metadata("num_fragments_afterMolrep", result["num_fragments"])
    homologue.add_metadata("longest_fragments_afterMolrep", result["longest_fragments"])
    homologue.add_metadata("num_res_sequenced_afterMolrep", result["num_res_sequenced"])
    homologue.add_metadata("num_res_unique_afterMolrep", result["num_res_unique"])
    homologue.add_metadata("percent_res_complete_afterMolrep", result["percent_res_complete"])
    homologue.add_metadata("percent_chain_complete_afterMolrep", result["percent_chain_complete"])
  return key, homologue

##########################################################################################
# Build with Buccaneer after Prosmart and first refinement
##########################################################################################

#build Prosmart solution after 100-cycles jelly body
def buccaneer_ssm_after_refmac_jelly(key, homologue, args):
  hklin = homologue.path("refmac_afterSSM.mtz")
  xyzin = homologue.path("refmac_afterSSM.pdb")
  fo = "FP,SIGFP"
  wrk_hl = "PHWT,FOM"
  seqin = homologue.chain.structure.path("unique.fasta")
  prefix = homologue.path("buccaneer_afterSSM")
  result = tasks.buccaneer(hklin, xyzin, fo, wrk_hl, seqin, prefix)
  homologue.jobs["buccaneer"] = result
  if "num_res_built" in result:
    homologue.add_metadata("num_res_built_afterSSM", result["num_res_built"])
    homologue.add_metadata("num_fragments_afterSSM", result["num_fragments"])
    homologue.add_metadata("longest_fragments_afterSSM", result["longest_fragments"])
    homologue.add_metadata("num_res_sequenced_afterSSM", result["num_res_sequenced"])
    homologue.add_metadata("num_res_unique_afterSSM", result["num_res_unique"])
    homologue.add_metadata("percent_res_complete_afterSSM", result["percent_res_complete"])
    homologue.add_metadata("percent_chain_complete_afterSSM", result["percent_chain_complete"])
  return key, homologue

##########################################################################################
# Restraint refinement after Buccaneer after Phaser and first refinement
##########################################################################################
#refine MR-Buccaneer solution using 20 cycles restraint refinement;
# first refinement 100 jelly body
def refine_placed_model_jelly_buccaneer_restraint(key, homologue, args):
  hklin = homologue.chain.structure.path("refmac.mtz")
  xyzin = homologue.path("buccaneer_afterMR.pdb")
  prefix = homologue.path("refmac_default_afterMR_Buccaneer")
  result = tasks.refine_default(hklin, xyzin, prefix)
  homologue.jobs["refmac"] = result
  if "final_rfree" in result:
    homologue.add_metadata("final_rfree_refmac_default_afterMR_Buccaneer", result["final_rfree"])
    homologue.add_metadata("final_rwork_refmac_default_afterMR_Buccaneer", result["final_rwork"])
    homologue.add_metadata("initial_rfree_refmac_default_afterMR_Buccaneer", result["initial_rfree"])
    homologue.add_metadata("initial_rwork_refmac_default_afterMR_Buccaneer", result["initial_rwork"])
  return key, homologue

##########################################################################################
# Restraint refinement after Buccaneer after Molrep and first refinement
##########################################################################################
#refine Molrep-Buccaneer solution using 20 cycles restraint refinement;
# first refinement 100 jelly body
def refine_molrep_model_jelly_buccaneer_restraint(key, homologue, args):
  hklin = homologue.chain.structure.path("refmac.mtz")
  xyzin = homologue.path("buccaneer_afterMolrep.pdb")
  prefix = homologue.path("refmac_default_afterMolrep_Buccaneer")
  result = tasks.refine_default(hklin, xyzin, prefix)
  homologue.jobs["refmac"] = result
  if "final_rfree" in result:
    homologue.add_metadata("final_rfree_refmac_default_afterMolrep_Buccaneer", result["final_rfree"])
    homologue.add_metadata("final_rwork_refmac_default_afterMolrep_Buccaneer", result["final_rwork"])
    homologue.add_metadata("initial_rfree_refmac_default_afterMolrep_Buccaneer", result["initial_rfree"])
    homologue.add_metadata("initial_rwork_refmac_default_afterMolrep_Buccaneer", result["initial_rwork"])
  return key, homologue

##########################################################################################
# Restraint refinement after Buccaneer after Prosmart and first refinement
##########################################################################################
#refine Prosmart-Buccaneer solution using 20 cycles restraint refinement;
# first refinement 100 jelly body
def refine_prosmart_model_jelly_buccaneer_restraint(key, homologue, args):
  hklin = homologue.chain.structure.path("refmac.mtz")
  xyzin = homologue.path("buccaneer_afterSSM.pdb")
  prefix = homologue.path("refmac_default_afterSSM_Buccaneer")
  result = tasks.refine_default(hklin, xyzin, prefix)
  homologue.jobs["refmac"] = result
  if "final_rfree" in result:
    homologue.add_metadata("final_rfree_refmac_default_afterSSM_Buccaneer", result["final_rfree"])
    homologue.add_metadata("final_rwork_refmac_default_afterSSM_Buccaneer", result["final_rwork"])
    homologue.add_metadata("initial_rfree_refmac_default_afterSSM_Buccaneer", result["initial_rfree"])
    homologue.add_metadata("initial_rwork_refmac_default_afterSSM_Buccaneer", result["initial_rwork"])
  return key, homologue

##########################################################################################
# Combine phases after refinement and Buccaneer after MR and refine
##########################################################################################
#combine phases PDB-redo target Buccaneer_afterMR_jelly_body
def write_combined_mtz_afterMR_buccaneer_jelly(key, homologue, args):
  prefix = homologue.path("gemmijoin_refmac_default_afterMR_Buccaneer")
  if not os.path.exists(homologue.path("refmac_default_afterMR_Buccaneer.mtz")):
    pass
  else:    
    result = tasks.combine_mtz(prefix, [
      (homologue.chain.structure.path("refmac.mtz")),
      (homologue.path("refmac_default_afterMR_Buccaneer.mtz"))])
    homologue.jobs["gemmijoin"] = result
  return key, homologue

##########################################################################################
# Combine phases after refinement and Buccaneer after Molrep and refine
##########################################################################################
#combine phases PDB-redo target Buccaneer_Molrep_jelly_body
def write_combined_mtz_afterMolrep_buccaneer_jelly(key, homologue, args):
  prefix = homologue.path("gemmijoin_refmac_default_afterMolrep_Buccaneer")
  if not os.path.exists(homologue.path("refmac_default_afterMolrep_Buccaneer.mtz")):
    pass
  else:    
    result = tasks.combine_mtz(prefix, [
      (homologue.chain.structure.path("refmac.mtz")),
      (homologue.path("refmac_default_afterMolrep_Buccaneer.mtz"))])
    homologue.jobs["gemmijoin"] = result
  return key, homologue

##########################################################################################
# Combine phases after refinement and Buccaneer after Prosmart and refine
##########################################################################################
#combine phases PDB-redo target Buccaneer_SSM_jelly_body
def write_combined_mtz_afterSSM_buccaneer_jelly(key, homologue, args):
  prefix = homologue.path("gemmijoin_refmac_default_afterSSM_Buccaneer")
  if not os.path.exists(homologue.path("refmac_default_afterSSM_Buccaneer.mtz")):
    pass
  else:    
    result = tasks.combine_mtz(prefix, [
      (homologue.chain.structure.path("refmac.mtz")),
      (homologue.path("refmac_default_afterSSM_Buccaneer.mtz"))])
    homologue.jobs["gemmijoin"] = result
  return key, homologue

##########################################################################################
# Compare phases after refinement and Buccaneer after MR and refine
##########################################################################################
#compare phases PDB-redo target MR_jelly_body
def compare_phases_afterMR_buccaneer_jelly(key, homologue, args):
  hklin = homologue.path("gemmijoin_refmac_default_afterMR_Buccaneer.mtz")
  fo = "FP,SIGFP"
  wrk_hl = "PHWT_model,FOM_model"
  ref_hl = "PHWT_ref,FOM_ref"
  prefix = homologue.path("cphasematch_refmac_default_afterMR_Buccaneer")
  result = tasks.compare_phases(hklin, fo, wrk_hl, ref_hl, prefix)
  homologue.jobs["cphasematch"] = result
  if "mean_phase_error" in result:
    homologue.add_metadata("mean_phase_error_afterMR_Buccaneer_refmac_default",
                           result["mean_phase_error"])
    homologue.add_metadata("f_map_correlation_afterMR_Buccaneer_refmac_default",
                           result["f_map_correlation"])
  return key, homologue
  
##########################################################################################
# Compare phases after refinement and Buccaneer after Molrep and refine
##########################################################################################
#compare phases PDB-redo target Molrep_jelly_body
def compare_phases_afterMolrep_buccaneer_jelly(key, homologue, args):
  hklin = homologue.path("gemmijoin_refmac_default_afterMolrep_Buccaneer.mtz")
  fo = "FP,SIGFP"
  wrk_hl = "PHWT_model,FOM_model"
  ref_hl = "PHWT_ref,FOM_ref"
  prefix = homologue.path("cphasematch_refmac_default_afterMolrep_Buccaneer")
  result = tasks.compare_phases(hklin, fo, wrk_hl, ref_hl, prefix)
  homologue.jobs["cphasematch"] = result
  if "mean_phase_error" in result:
    homologue.add_metadata("mean_phase_error_afterMolrep_Buccaneer_refmac_default",
                           result["mean_phase_error"])
    homologue.add_metadata("f_map_correlation_afterMolrep_Buccaneer_refmac_default",
                           result["f_map_correlation"])
  return key, homologue


##########################################################################################
# Compare phases after refinement and Buccaneer after Prosmart and refine
##########################################################################################
#compare phases PDB-redo target SSM_jelly_body
def compare_phases_afterSSM_buccaneer_jelly(key, homologue, args):
  hklin = homologue.path("gemmijoin_refmac_default_afterSSM_Buccaneer.mtz")
  fo = "FP,SIGFP"
  wrk_hl = "PHWT_model,FOM_model"
  ref_hl = "PHWT_ref,FOM_ref"
  prefix = homologue.path("cphasematch_refmac_default_afterSSM_Buccaneer")
  result = tasks.compare_phases(hklin, fo, wrk_hl, ref_hl, prefix)
  homologue.jobs["cphasematch"] = result
  if "mean_phase_error" in result:
    homologue.add_metadata("mean_phase_error_afterSSM_Buccaneer_refmac_default",
                           result["mean_phase_error"])
    homologue.add_metadata("f_map_correlation_afterSSM_Buccaneer_refmac_default",
                           result["f_map_correlation"])
  return key, homologue


def run_mr_pipelines(key, homologue, args):
  #print("Working on homologue: ", homologue)
  #print("\n")
#  try:
#    os.path.exists(homologue.path("JOB_IS_DONE.txt")) == True
#    print("MR and SSM have been done")
#  except:
#    # superpose homologue on to PDB-redo target (ground truth); use sculptor to trim model
#    # to use in MR
#    print("Running MR and SSM")
#    superpose_homologue(key, homologue, args)
#    prepare_sculptor_alignment(key, homologue, args)
#    trim_model(key, homologue, args)
#
#    # superpose homologue on to PDB-redo target using Prosmart; refine placed model with 0-cycles and
#    # 100 cycles jelly body in Refmac against PDB-redo ground truth MTZ;
#    #combine the refinement phases (from 0-cycles as well
#    # as jelly body) with the ground truth from PDB-redo; calculate phase angle between
#    # the phases from 0-cycle refinement and PDB-redo ground truth as well as jelly body
#    # and PDB-redo ground truth
#    superpose_prosmart(key, homologue, args)
#    refine_ssm_model_zero(key, homologue, args)
#    write_combined_mtz_afterSSM_zero(key, homologue, args)
#    compare_phases_afterSSM_zero(key, homologue, args)
#    refine_ssm_model_jelly(key, homologue, args)
#    write_combined_mtz_afterSSM_jelly(key, homologue, args)
#    compare_phases_afterSSM_jelly(key, homologue, args)
#
#    # superpose homologue on to PDB-redo target using Molrep; refine placed model with 0-cycles and
#    # 100 cycles jelly body in Refmac against PDB-redo ground truth MTZ;
#    #combine the refinement phases (from 0-cycles as well
#    # as jelly body) with the ground truth from PDB-redo; calculate phase angle between
#    # the phases from 0-cycle refinement and PDB-redo ground truth as well as jelly body
#    # and PDB-redo ground truth
#    superpose_molrep(key, homologue, args)
#    refine_molrep_model_zero(key, homologue, args)
#    write_combined_mtz_afterMolrep_zero(key, homologue, args)
#    compare_phases_afterMolrep_zero(key, homologue, args)
#    refine_molrep_model_jelly(key, homologue, args)
#    write_combined_mtz_afterMolrep_jelly(key, homologue, args)
#    compare_phases_afterMolrep_jelly(key, homologue, args)
#
#    # MR of homologue on to PDB-redo target using Phaser; refine placed model with 0-cycles and
#    # 100 cycles jelly body in Refmac against PDB-redo ground truth MTZ;
#    # combine the refinement phases (from 0-cycles as well
#    # as jelly body) with the ground truth from PDB-redo; calculate phase angle between
#    # the phases from 0-cycle refinement and PDB-redo ground truth as well as jelly body
#    # and PDB-redo ground truth
#    mr(key, homologue, args)
#    refine_placed_model_zero(key, homologue, args)
#    write_combined_mtz_afterMR_zero(key, homologue, args)
#    compare_phases_afterMR_zero(key, homologue, args)
#    refine_placed_model_jelly(key, homologue, args)
#    write_combined_mtz_afterMR_jelly(key, homologue, args)
#    compare_phases_afterMR_jelly(key, homologue, args)
#    
#    with open(homologue.path("JOB_IS_DONE.txt"), "w") as out_file:
#      line = "job is done"
#      out_file.writelines(line)
#  else:
#    print("\n")
#    print("MR and SSM already done.")
#    print("\n")

  if not os.path.exists(homologue.path("BUILD_WITH_BUCCANEER.TXT")):# == True:
#  try:
#    os.path.exists(homologue.path("BUILD_WITH_BUCCANEER.TXT")) == True
#    print("Building with Buccaneer has been done")
#  except:
    print("Building with Buccaneer missing homologues")
    if os.path.exists(homologue.path("refmac_afterMR.mtz")):
      print("Building MR result")
      print("\n")
      # Phaser-placed model after 100 cycles jelly body Refmac refinement; built with Buccaneer;
      # refined with Refmac 0-cycle and 100 cycles jelly body using the PDB-redo ground truth
      # MTZ;
      # combine the refinement phases (from 0-cycles as well
      # as jelly body) with the ground truth from PDB-redo; calculate phase angle between
      # the phases from 0-cycle refinement and PDB-redo ground truth as well as jelly body
      # and PDB-redo ground truth
#      buccaneer_mr_after_refmac_jelly(key, homologue, args)
#      refine_placed_model_jelly_buccaneer_restraint(key, homologue, args)
#      write_combined_mtz_afterMR_buccaneer_jelly(key, homologue, args)
#      compare_phases_afterMR_buccaneer_jelly(key, homologue, args)
    else:
      print("Building Molrep result")
      print("\n")
      #os.path.exists(homologue.path("refmac_afterMolrep.mtz"))
      # Molrep-placed model after 100 cycles jelly body Refmac refinement; built with Buccaneer;
      # refined with Refmac 0-cycle and 100 cycles jelly body using the PDB-redo ground truth
      # MTZ;
      # combine the refinement phases (from 0-cycles as well
      # as jelly body) with the ground truth from PDB-redo; calculate phase angle between
      # the phases from 0-cycle refinement and PDB-redo ground truth as well as jelly body
      # and PDB-redo ground truth
#      buccaneer_molrep_after_refmac_jelly(key, homologue, args)
#      refine_molrep_model_jelly_buccaneer_restraint(key, homologue, args)
#      write_combined_mtz_afterMolrep_buccaneer_jelly(key, homologue, args)
#      compare_phases_afterMolrep_buccaneer_jelly(key, homologue, args)
    if not os.path.exists(homologue.path("refmac_afterMR.mtz")) and not os.path.exists(homologue.path("refmac_afterMolrep.mtz")):
      print("Building Prosmart result")
      print("\n")
      # Prosmart-placed model after 100 cycles jelly body Refmac refinement; built with Buccaneer;
      # refined with Refmac 0-cycle and 100 cycles jelly body using the PDB-redo ground truth
      # MTZ;
      # combine the refinement phases (from 0-cycles as well
      # as jelly body) with the ground truth from PDB-redo; calculate phase angle between
      # the phases from 0-cycle refinement and PDB-redo ground truth as well as jelly body
      # and PDB-redo ground truth
#      buccaneer_ssm_after_refmac_jelly(key, homologue, args)
#      refine_prosmart_model_jelly_buccaneer_restraint(key, homologue, args)
#      write_combined_mtz_afterSSM_buccaneer_jelly(key, homologue, args)
#      compare_phases_afterSSM_buccaneer_jelly(key, homologue, args)
#    with open(homologue.path("BUILD_WITH_BUCCANEER.txt"), "w") as out_file:
#      line = "job is done"
#      out_file.writelines(line)
    print("Finished building")
  else:
    print("\n")
    print("No MR or SSM results found.")
    print("\n")
#    pass

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

