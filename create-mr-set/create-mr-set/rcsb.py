import copy
import csv
import os
import urllib.request

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

def _get_structures():
  global _structures
  columns = [
    "experimentalTechnique",
    "resolution",
    "rWork",
    "rFree",
    "entityMacromoleculeType",
    "clusterNumber95",
    "clusterNumber90",
    "clusterNumber70",
    "clusterNumber50",
    "clusterNumber40",
    "clusterNumber30",
  ]
  if not os.path.exists("pdb-chains.csv"):
    download_custom_report(columns, "pdb-chains.csv")
  _structures = {}
  with open("pdb-chains.csv") as f:
    for row in csv.DictReader(f):
      if any(row[column] == "" for column in columns): continue
      if row["experimentalTechnique"] != "X-RAY DIFFRACTION": continue
      if row["entityMacromoleculeType"] != "Polypeptide(L)": continue
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

def structures():
  """Get a dictionary of X-ray structures with protein chains"""
  if _structures is None: _get_structures()
  return copy.deepcopy(_structures)

def cluster_number(structure_id, chain_id, cluster_level):
  """Return the cluster number for an individual protein chain"""
  assert(cluster_level in {95, 90, 70, 50, 40, 30})
  if _structures is None: _get_structures()
  if structure_id in _structures:
    if chain_id in _structures[structure_id].chains:
      attr = "cluster%d" % cluster_level
      return getattr(_structures[structure_id].chains[chain_id], attr)
