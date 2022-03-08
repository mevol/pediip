#!/usr/bin/env python3

import numpy 
import pandas
import gemmi
import os
import modules.create_mr_set.utils.pdbtools as pdbtools
import re
import shutil
import modules.create_mr_set.utils.utils as utils
import uuid
import xml.etree.ElementTree as ET
from itertools import islice
from linecache import getline


def add_freer_flag(hklin, prefix):
  result = {
    "hklout": "%s.mtz" % prefix,
    "stdout": "%s.log" % prefix,
    "stderr": "%s.err" % prefix,
  }
  utils.run("freerflag", [
    "hklin", hklin,
    "hklout", result["hklout"],
  ], [
    "END"
  ], stdout=result["stdout"], stderr=result["stderr"])
  return result

def cif2mtz(hklin, prefix):
  result = {
    "hklout": "%s.mtz" % prefix,
    "stdout": "%s.log" % prefix,
    "stderr": "%s.err" % prefix,
  }
  utils.run("cif2mtz", [
    "hklin", hklin,
    "hklout", result["hklout"],
  ], [
    "END"
  ], stdout=result["stdout"], stderr=result["stderr"])
  if not os.path.exists(result["hklout"]):
    return { "error": "No MTZ file produced" }
  with open(result["stderr"]) as f:
    for line in f:
      line = line.strip()
      if line[:8] == "cif2mtz:":
        return { "error": line[8:].strip() }
  return result

def combine_mtz(prefix, columns):
  """Combine columns from multiple MTZ files with gemmijoin"""
  result = {
    "hklout": "%s.mtz" % prefix,
    "stdout": "%s.log" % prefix,
    "stderr": "%s.err" % prefix,
  }
  reference = columns[0]
  reference_mtz = gemmi.read_mtz_file(reference)  
  reference_data = numpy.array(reference_mtz, copy=False)  
  reference_df = pandas.DataFrame(data=reference_data, columns=reference_mtz.column_labels())
  reference_sel_col_df = reference_df[['H', 'K', 'L', 'FP', 'SIGFP', 'FREE', 'PHWT', 'FOM']]

  model = columns[1]
  model_mtz = gemmi.read_mtz_file(model)
  model_data = numpy.array(model_mtz, copy=False)
  model_df = pandas.DataFrame(data=model_data, columns=model_mtz.column_labels())
  model_sel_col_df = model_df[['PHWT', 'FOM']]
  model_sel_col_df = model_sel_col_df.rename(columns={'PHWT': 'PHWT_model', 'FOM': 'FOM_model'})

  combined_df = pandas.concat([reference_sel_col_df, model_sel_col_df], ignore_index=True, axis=1)  
  
  combined_array = combined_df.to_numpy()

  mtz_new = gemmi.Mtz()
  mtz_new.add_dataset('combined')
  mtz_new.add_column('H', 'H', dataset_id=0)
  mtz_new.add_column('K', 'H', dataset_id=0)
  mtz_new.add_column('L', 'H', dataset_id=0)
  mtz_new.add_column('FP', 'F', dataset_id=0)
  mtz_new.add_column('SIGFP', 'Q', dataset_id=0)
  mtz_new.add_column('FREE', 'I', dataset_id=0)
  mtz_new.add_column('PHWT_ref', 'P', dataset_id=0)
  mtz_new.add_column('FOM_ref', 'W', dataset_id=0)
  mtz_new.add_column('PHWT_model', 'P', dataset_id=0)
  mtz_new.add_column('FOM_model', 'W', dataset_id=0)

  unit_cell = reference_mtz.cell
  space_group = reference_mtz.spacegroup
  mtz_new.spacegroup = space_group
  mtz_new.cell = unit_cell
  mtz_new.set_data(combined_array)  
  
  mtz_new.write_to_file(result["hklout"])
  
  if not os.path.exists(result["hklout"]):
    return { "error": "No reflection data produced" }
  return result


def compare_phases(hklin, fo, wrk_hl, ref_hl, prefix):
  """Compare two sets of phases with cphasematch"""
  result = {
    "stdout": "%s.log" % prefix,
    "stderr": "%s.err" % prefix,
  }
  utils.run("cphasematch", [
    "-mtzin", hklin,
    "-colin-fo", fo,
    "-colin-phifom-1", wrk_hl,
    "-colin-phifom-2", ref_hl,
  ], stdout=result["stdout"], stderr=result["stderr"])
  with open(result["stdout"]) as f:
    for line in f:
      if "Overall statistics:" in line:
        headers = next(f).split()
        values = next(f).split()
        result["mean_phase_error"] = float(values[headers.index("w1<dphi>")])
        result["f_map_correlation"] = float(values[headers.index("wFcorr")])
  return result


def convert_amplitudes(hklin, seqin, prefix):
  result = {
    "hklout": "%s.mtz" % prefix,
    "stdout": "%s.log" % prefix,
    "stderr": "%s.err" % prefix,
  }
  mtz = gemmi.read_mtz_file(hklin)
  labels = [col.label for col in mtz.columns]
  column_sets = [
    ["FP", "SIGFP"],
    ["I", "SIGI"],
    ["F(+)", "SIGF(+)", "F(-)", "SIGF(-)"],
    ["I(+)", "SIGI(+)", "I(-)", "SIGI(-)"],
  ]
  columns = next((s for s in column_sets if all(l in labels for l in s)), None)
  if columns is None:
    return { "error": "Can't find columns to convert to F,SIGF" }
  arguments = [
    "-hklin", hklin,
    "-seqin", seqin,
    "-hklout", result["hklout"],
  ]
  if len(columns) == 2:
    arguments.extend(["-colin", "/*/*/[%s]" % ",".join(columns)])
    result["colout"] = ["F", "SIGF"]
  else:
    arguments.extend(["-colano", "/*/*/[%s]" % ",".join(columns)])
    result["colout"] = ["FMEAN", "SIGFMEAN"]
  if columns[0][0] == "F":
    arguments.append("-amplitudes")
  utils.run("ctruncate", arguments, stdout=result["stdout"], stderr=result["stderr"])
  with open(result["stderr"]) as f:
    line = f.readline().strip()
    if len(line) > 0:
      return { "error": line }
  return result


def mr(hklin, xyzin, identity, prefix, copies, atom_counts):
  """Perform molecular replacement with PHASER"""
  result = {
    "xyzout": "%s.1.pdb" % prefix,
    "hklout": "%s.1.mtz" % prefix,
    "solout": "%s.sol" % prefix,
    "stdout": "%s.log" % prefix,
    "stderr": "%s.err" % prefix,
    "llg": None,
    "rmsd": None,
  }
  keywords = [
    "MODE MR_AUTO",
    "HKLIN %s" % hklin,
    "ENSEMBLE model PDBFILE %s IDENTITY %s" % (xyzin, identity),
    "SEARCH ENSEMBLE model NUM %d" % copies,
    "ROOT %s" % prefix,
    "PURGE ROT NUMBER 1",
    "PURGE TRA NUMBER 1",
    "PURGE RNP NUMBER 1",
    "JOBS 4",
  ]
  for atom in atom_counts:
    if atom.strip() == "X": continue
    keywords.append("COMPOSITION ATOM %-2s NUMBER %d" % (atom, atom_counts[atom]))
  utils.run("phaser", stdin=keywords, stdout=result["stdout"], stderr=result["stderr"])
  if not os.path.exists(result["xyzout"]):
    with open(result["stdout"]) as f: log = f.read()
    if not "EXIT STATUS:" in log:
      return { "error": "Early termination" }
    elif "EXIT STATUS: SUCCESS" in log:
      return { "error": "No solution found" }
    elif "INPUT ERROR: No scattering in coordinate file" in log:
      return { "error": "No scattering in input coordinates" }
    elif "INPUT ERROR: Structure Factors of Models" in log:
      return { "error": "Bad ensemble given as input" }
    elif "F is negative" in log:
      return { "error": "Reflection(s) with negative F" }
    else:
      return { "error": "No coordinates produced" }
  with open(result["xyzout"]) as f:
    for line in f:
      if line[:26] == "REMARK Log-Likelihood Gain":
        result["llg"] = float(line.split()[-1])
        break
  with open(result["solout"]) as f:
    for line in f:
      split = line.split()
      if "#RMSD" in split:
        i = split.index("#RMSD")
        result["rmsd"] = float(split[i+1])
        break
  return result


def refine_jelly(hklin, xyzin, prefix, cycles=100):
  """Refine a structure with REFMAC"""
  result = {
    "hklout": "%s.mtz" % prefix,
    "xyzout": "%s.pdb" % prefix,
    "xmlout": "%s.xml" % prefix,
    "stdout": "%s.log" % prefix,
    "stderr": "%s.err" % prefix,
  }
  utils.run("refmac5", [
    "HKLIN", hklin,
    "XYZIN", xyzin,
    "HKLOUT", result["hklout"],
    "XYZOUT", result["xyzout"],
    "XMLOUT", result["xmlout"],
  ], [
    "NCYCLES %d" % cycles,
    "RIDGE DISTANCE INCLUDE all",
    "RIDGE DISTANCE SIGMA 0.02",
    "RIDGE DISTANCE DMAX 4.0",
    "MAKE NEWLIGAND NOEXIT",
    "PHOUT",
    "END"
  ], stdout=result["stdout"], stderr=result["stderr"])
  with open(result["stderr"]) as f:
    for line in f:
      line = line.strip()
      if line[:7] == "Refmac:":
        return { "error": line[7:].strip() }
  for output in ("hklout", "xyzout", "xmlout"):
    if not os.path.exists(result[output]):
      return { "error": "Output file missing: %s" % result[output] }
  root = ET.parse(result["xmlout"]).getroot()
  rworks = list(root.iter("r_factor"))
  rfrees = list(root.iter("r_free"))
  result["data_completeness"] = float(list(root.iter("data_completeness"))[0].text)
  result["final_rfree"] = float(rfrees[-1].text)
  result["final_rwork"] = float(rworks[-1].text)
  result["initial_rfree"] = float(rfrees[0].text)
  result["initial_rwork"] = float(rworks[0].text)
  return result

def refine_zero(hklin, xyzin, prefix, cycles=0):
  """Refine a structure with REFMAC"""
  result = {
    "hklout": "%s.mtz" % prefix,
    "xyzout": "%s.pdb" % prefix,
    "xmlout": "%s.xml" % prefix,
    "stdout": "%s.log" % prefix,
    "stderr": "%s.err" % prefix,
  }
  utils.run("refmac5", [
    "HKLIN", hklin,
    "XYZIN", xyzin,
    "HKLOUT", result["hklout"],
    "XYZOUT", result["xyzout"],
    "XMLOUT", result["xmlout"],
  ], [
    "NCYCLES %d" % cycles,
    "MAKE NEWLIGAND NOEXIT",
    "PHOUT",
    "END"
  ], stdout=result["stdout"], stderr=result["stderr"])
  with open(result["stderr"]) as f:
    for line in f:
      line = line.strip()
      if line[:7] == "Refmac:":
        return { "error": line[7:].strip() }
  for output in ("hklout", "xyzout", "xmlout"):
    if not os.path.exists(result[output]):
      return { "error": "Output file missing: %s" % result[output] }
  root = ET.parse(result["xmlout"]).getroot()
  rworks = list(root.iter("r_factor"))
  rfrees = list(root.iter("r_free"))
  result["data_completeness"] = float(list(root.iter("data_completeness"))[0].text)
  result["final_rfree"] = float(rfrees[-1].text)
  result["final_rwork"] = float(rworks[-1].text)
  result["initial_rfree"] = float(rfrees[0].text)
  result["initial_rwork"] = float(rworks[0].text)
  return result


def remove_unl_residues(xyzin, prefix):
  result = {
    "xyzout": "%s.pdb" % prefix
  }
  shutil.copy(xyzin, result["xyzout"])
  utils.run("sed", ["-i", "/^HET.*UNL/d", result["xyzout"]])
  utils.run("sed", ["-i", "/^ATOM.*UNL/d", result["xyzout"]])
  utils.run("sed", ["-i", "/^REMARK 500.*UNL/d", result["xyzout"]])
  return result


def select_and_rename_columns(hklin, colin, colout, prefix):
  result = {
    "hklout": "%s.mtz" % prefix,
    "stdout": "%s.log" % prefix,
    "stderr": "%s.err" % prefix,
  }
  labi = " ".join("E%d=%s" % (i+1, colin[i]) for i in range(len(colin)))
  labo = " ".join("E%d=%s" % (i+1, colout[i]) for i in range(len(colout)))
  utils.run("cad", [
    "hklin1", hklin,
    "hklout", result["hklout"]
  ], [
    "LABI FILE_NUMBER 1 %s" % labi,
    "XNAME FILE_NUMBER 1 ALL=",
    "DNAME FILE_NUMBER 1 ALL=",
    "LABO FILE_NUMBER 1 %s" % labo,
    "END",
  ], stdout=result["stdout"], stderr=result["stderr"])
  return result

def structural_homologues(xyzin, chain, prefix, archive, threads="auto"):
  """Search for structural homologues with a GESAMT achrive search"""
  result = {
    "txtout": "%s.txt" % prefix,
    "stdout": "%s.log" % prefix,
    "stderr": "%s.err" % prefix,
  }
  utils.run("gesamt", [
    xyzin, "-s", "//%s" % chain,
    "-archive", archive,
    "-nthreads=%s" % threads,
    "-o", result["txtout"],
  ], stdout=result["stdout"], stderr=result["stderr"])
  return result

def superpose(xyzin1, chain1, xyzin2, chain2, prefix):
  """Superpose one chain over another with GESAMT"""
  result = {
    "xyzout": "%s.pdb" % prefix,
    "seqout": "%s.seq" % prefix,
    "stdout": "%s.log" % prefix,
    "stderr": "%s.err" % prefix,
  }
  utils.run("gesamt", [
    xyzin1, "-s", "//%s" % chain1,
    xyzin2, "-s", "//%s" % chain2,
    "-o", result["xyzout"],
    "-a", result["seqout"],
  ], stdout=result["stdout"], stderr=result["stderr"])
  with open(result["stdout"]) as f: log = f.read()
  if "DISSIMILAR and cannot be reasonably aligned" in log:
    return { "error": "GESAMT: Query and target are too dissimilar" }
  for output in ("xyzout", "seqout"):
    if not os.path.exists(result[output]):
      return { "error": "Output file missing: %s" % result[output] }
  match = re.search(r" Q-score          : (\d+\.\d+)", log)
  result["qscore"] = float(match.group(1))
  match = re.search(r" RMSD             : (\d+\.\d+)", log)
  result["rmsd"] = float(match.group(1))
  match = re.search(r" Aligned residues : (\d+)", log)
  result["length"] = int(match.group(1))
  match = re.search(r" Sequence Id:     : (\d+\.\d+)", log)
  result["seqid"] = float(match.group(1))
  return result

def superpose_like_mr(xyzin1, chain1, xyzin2, chain2, prefix):
  """Superpose one chain over another with prosmart"""
  #print("Running Prosmart on", xyzin1, xyzin2)
  result = {
    "outdir": "%s" % prefix,
    "stdout": "%s.log" % prefix,
    "stderr": "%s.err" % prefix,
  }
  utils.run("prosmart", [
    "-a",
    "-p1", xyzin1,
    "-p2", xyzin2,
    "-c1", chain1,
    "-c2", chain2,
    "-o", result["outdir"],
  ], stdout=result["stdout"], stderr=result["stderr"])
  with open(result["stdout"]) as f:
    log = f.read()
  if not "prosmart_align_logfile.txt" in log:
    return { "error": "prosmart: file missing" }
  with open(os.path.join(result["outdir"], "prosmart_align_logfile.txt")) as f2:
    out = f2.read()
  if "Rigid alignment global RMSD =" not in out:
    result["rmsd"] = 0.0
  else:  
    match = re.search(r"Rigid alignment global RMSD = (\d+\.\d+)", out)
    result["rmsd"] = float(match.group(1))
  if "Alignment:" not in out:
    result["alignment_length"] = 0
  else:
    match = re.search(r"Alignment: (\d+)", out)
    result["alignment_length"] = int(match.group(1))
  if "Strict sequence identity of aligned residues:" not in out:
    result["seqid"] = 0.0
  else:  
    match = re.search(r"Strict sequence identity of aligned residues: (\d+\.\d+)", out)
    result["seqid"] = float(match.group(1))
  return result

def superpose_molrep(xyzin1, xyzin2, prefix):
  """Superpose one chain over another with Molrep"""
  result = {
    "xyzout": "%s.pdb" % prefix,
    "outdir": "%s" % prefix,
    "stdout": "%s.log" % prefix,
    "stderr": "%s.err" % prefix,
  }
  utils.run("molrep", [
    "-m", xyzin1,
    "-mx", xyzin2,
    "-po", result["outdir"],
    "-ps", result["outdir"],
  ], stdout=result["stdout"], stderr=result["stderr"])
  with open(result["stdout"]) as f:
    log = f.read()  
  if " corrF,corrD" not in log:
    result["corrF"] = 0.0
    result["corrD"] = 0.0
  else:  
    match = re.search(r" corrF,corrD =   ((\d+\.\d+)\s*(\d+\.\d+))", log)
    result["corrF"] = float(match.group(1).split(" ")[0])
    result["corrD"] = float(match.group(1).split(" ")[-1])  
  if " TF/sig       =" not in log:
    result["TF_sig"] = 0.0
  else:
    match = re.search(r" TF/sig       =   (\s*[+-]?(\d+\.\d+))", log)
    if match == None:
      result["TF_sig"] = 0.0
    else:  
      result["TF_sig"] = float(match.group(1))
  if " Final CC     =" not in log:
    result["final_cc"] = 0.0
  else:  
    match = re.search(r" Final CC     =   (\d+\.\d+)", log)
    result["final_cc"] = float(match.group(1))
  if " Packing_Coef =" not in log:
    result["packing_coeff"] = 0.0
  else:  
    match = re.search(r" Packing_Coef =   (\d+\.\d+)", log)
    result["packing_coeff"] = float(match.group(1))
  match = re.search(r" Contrast     = (\s*(\d+\.\d+))", log)
  if match == None:
    result["contrast"] = 0.0
  else:
    result["contrast"] = float(match.group(1))
  return result


def trim_model(model, chain, alignment, prefix):
  """Trim a molecular replacement model with SCULPTOR"""
  result = {
    "xyzout": "%s_%s.pdb" % (prefix, os.path.basename(model)[:-4]),
    "stdout": "%s.log" % prefix,
    "stderr": "%s.err" % prefix,
  }
  utils.run("phaser.sculptor", ["--stdin"], [
    "input {",
    "  model {"
    "    file_name = %s" % model,
    "    selection = chain %s" % chain,
    "    remove_alternate_conformations = True",
    "  }",
    "  alignment {",
    "    file_name = %s" % alignment,
    "    target_index = 1",
    "  }",
    "}",
    "output {",
    "  folder = %s" % os.path.dirname(prefix),
    "  root = '%s'" % os.path.basename(prefix),
    "}",
  ], stdout=result["stdout"], stderr=result["stderr"])
  if not os.path.exists(result["xyzout"]):
    return { "error": "No trimmed coordinates produced" }
  if not pdbtools.has_atoms(result["xyzout"]):
    return { "error": "SCULPTOR: No atoms in trimmed coordinates" }
  return result


def buccaneer(hklin, xyzin, fo, wrk_hl, seqin, prefix):
#def buccaneer(hklin, xyzin, fo, wrk_hl, prefix):

  """Running automated model building with Buccaneer"""
  result = {
    "stdout": "%s.log" % prefix,
    "stderr": "%s.err" % prefix,
  }
  utils.run("cbuccaneer", [
    "-mtzin", hklin,
    "-pdbin", xyzin,
    "-colin-fo", fo,
    "-colin-phifom", wrk_hl,
    "-seqin", seqin,
    "-pdbout", "%s.pdb" % prefix,
  ], stdout=result["stdout"], stderr=result["stderr"])
  with open(result["stdout"]) as f:
    for ind, line in enumerate(f, 1):
      if line.strip().startswith("$TEXT:Result: $$ $$"):
        stats = list(islice(f, 5))
        print(stats)
        split1 = stats[0].split()
        split2 = stats[1].split()
        split3 = stats[2].split()
        split4 = stats[3].split()
        split5 = stats[4].split()
        built = split1[0]
        fragments = split1[5]
        longest = split1[-2]
        sequenced = split2[0]
        unique = split3[0]
        residue_completeness = split4[-1].strip("%")
        chain_completeness = split5[-2].strip("%")
#        print(chain_completeness)
        result["num_res_built"] = int(built)
        result["num_fragments"] = int(fragments)
        result["longest_fragments"] = int(longest)
        result["num_res_sequenced"] = int(sequenced)
        result["num_res_unique"] = int(unique)
        result["percent_res_complete"] = float(residue_completeness)
        result["percent_chain_complete"] = float(chain_completeness)
  return result
