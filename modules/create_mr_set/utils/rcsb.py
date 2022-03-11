import argparse
import copy
import csv
import operator
import collections
import json
import os
import urllib.request
import gzip
import sqlite3
import modules.create_mr_set.utils.utils as utils
import multiprocessing
from functools import partial

_structures = None

class _Structure:
  def __init__(self, row):
    self.id = row["structureId"]
    self.resolution = float(row["resolution"])
    self.rwork = float(row["rWork"])
    self.rfree = float(row["rFree"])
    self.chains = {}

class _Chain:
  def __init__(self, row):
    self.id = row["chainId"]
    self.cluster95 = row["clusterNumber95"]
    self.cluster90 = row["clusterNumber90"]
    self.cluster70 = row["clusterNumber70"]
    self.cluster50 = row["clusterNumber50"]
    self.cluster40 = row["clusterNumber40"]
    self.cluster30 = row["clusterNumber30"]

def _get_structures(args):
  global _structures
  #global add_pdb_redo_info
  #db = "/dls/science/users/ghp45345/pediip/modules/db_files/pdb_redo_db.sqlite"
  
  download_columns = ["entityMacromoleculeType",
                      "experimentalTechnique",
                      "resolution",
                      "rWork",
                      "rFree",
                      "clusterNumber95",
                      "clusterNumber90",
                      "clusterNumber70",
                      "clusterNumber50",
                      "clusterNumber40",
                      "clusterNumber30"]  
 
  if not os.path.exists("pdb-chains.csv"):  
    download_custom_report(download_columns, "pdb-chains.csv")

  _structures = {}  

  headers = ["structureId",
             "chainId",
             "entityMacromoleculeType",
             "experimentalTechnique",
             "resolution",
             "rWork",
             "rFree",
             "clusterNumber95",
             "clusterNumber90",
             "clusterNumber70",
             "clusterNumber50",
             "clusterNumber40",
             "clusterNumber30"]
  #           "rWork_tls",
  #           "rFree_tls",
  #           "rWork_final",
  #           "rFree_final",
  #           "completeness_final"]

  xray_poly_lst = []

  with open("pdb-chains.csv") as f_1:
    reader_1 = csv.reader(f_1)           
    for row in reader_1:
      if "X-RAY DIFFRACTION" in row and "Polypeptide(L)" in row:  
        xray_poly_lst.append(row)

  print("remove no X-ray and no poly-peptide", len(xray_poly_lst))

  
  rm_empty_str_lst = []  
  for row in xray_poly_lst:
    if "" not in row:
      rm_empty_str_lst.append(row)

  print("remove missing values", len (rm_empty_str_lst))

  
  rm_high_rFree = []  
  for row in rm_empty_str_lst:
    if float(row[6]) <= 0.250:
      rm_high_rFree.append(row)

  print("remove rFree > 0.25", len(rm_high_rFree))

  counter = collections.Counter()
  for row in rm_high_rFree:
    this_id = row[0].strip().upper()
    counter[this_id] += 1

  final_lst = []
  for row in rm_high_rFree:
    this_id = row[0].strip().upper()
    if counter[this_id] > 1:
        continue
    final_lst.append(row)


  print("remove duplicate PDB entries", len(final_lst))

#  added_pdb_info_lst = []  
#
#  manager = multiprocessing.Manager()
#  added_pdb_info_lst = manager.list()
#
#  def add_pdb_redo_info(entry):
#    structure_id = entry[0]
#    
#    pdb_id = None
#    
#    handle = sqlite3.connect(db)
#
#    cur = handle.cursor()
#
#    cur.execute('''
#      SELECT id
#      FROM pdb_id
#      WHERE pdb_id="%s"
#      ''' % (structure_id))
#    get_id = cur.fetchone()
#    if get_id is not None:
#      pdb_id = get_id[0]
#      cur.execute('''
#        SELECT rWork_tls, rFree_tls, rWork_final, rFree_final, completeness
#        FROM pdb_redo_stats
#        WHERE pdb_id_id="%s"
#        ''' % (pdb_id))
#      stats = cur.fetchall()
#    
#      tls_rWork = stats[0][0]
#      tls_rFree = stats[0][1]
#      final_rWork = stats[0][2]
#      final_rFree = stats[0][3]
#      final_completeness = stats[0][4]
#    
#      entry.append(tls_rWork)
#      entry.append(tls_rFree)
#      entry.append(final_rWork)
#      entry.append(final_rFree)
#      entry.append(final_completeness)
#      
#      added_pdb_info_lst.append(entry)
#
#  pool = multiprocessing.Pool()
#  pool.map(add_pdb_redo_info, final_lst)
#  pool.close()
#  pool.join()
#  
#  rm_low_completeness = []
#  
#  for row in added_pdb_info_lst:
#    if float(row[17]) >= 90.0:
#      rm_low_completeness.append(row)
#
#  print("remove low completeness", len(rm_low_completeness))
#
#  no_rfree_increase = []
#
#  for row in rm_low_completeness:
#    if row[6] >= row[14]:
#      no_rfree_increase.append(row)
#      
#  print("remove increased Rfree", len(no_rfree_increase))
      
  with open("pdb-chains-short.csv", "w", newline = "") as out_2:
    writer_2 = csv.writer(out_2)
    writer_2.writerow(headers)
    #for row in no_rfree_increase:
    for row in final_lst:
      writer_2.writerow(row)
                     
  with open("pdb-chains-short.csv") as f_2:   
    for row in csv.DictReader(f_2):
      structure_id = row["structureId"]
      chain_id = row["chainId"]
      if structure_id not in _structures:
        _structures[structure_id] = _Structure(row)
      _structures[structure_id].chains[chain_id] = _Chain(row)
          
def download_custom_report(columns, path):
  """Download a custom report for all structures in the PDB"""
  url = "https://www.rcsb.org/pdb/rest/customReport.xml?pdbids=*&"
  url += "customReportColumns=%s&" % ",".join(columns)
  url += "format=csv&service=wsfile"
  urllib.request.urlretrieve(url, path)

def structures(args):
  """Get a dictionary of X-ray structures with protein chains"""
  if _structures is None:
    _get_structures(args)
  return copy.deepcopy(_structures)

def cluster_number(structure_id, chain_id, cluster_level):
  """Return the cluster number for an individual protein chain"""
  assert(cluster_level in {95, 90, 70, 50, 40, 30})
  if _structures is None: _get_structures()
  if structure_id in _structures:
    if chain_id in _structures[structure_id].chains:
      attr = "cluster%d" % cluster_level
      return getattr(_structures[structure_id].chains[chain_id], attr)
