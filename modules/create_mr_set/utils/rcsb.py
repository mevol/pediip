import copy
import csv
import json
import os
import urllib.request
import gzip
import modules.create_mr_set.utils.utils as utils

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
    #add those below as empty keys to dict below
    self.cluster95 = row["clusterNumber95"]
    self.cluster90 = row["clusterNumber90"]
    self.cluster70 = row["clusterNumber70"]
    self.cluster50 = row["clusterNumber50"]
    self.cluster40 = row["clusterNumber40"]
    self.cluster30 = row["clusterNumber30"]

#def _get_structures(pdb_dir):
def _get_structures():

  # get PDB structures and write to csv file with  column labels below
  # set "_structures" as global variable
  global _structures
  
  #structures_keys = ["structureId",
  #                   "chainId",
  #                   "completeness",          #COMPLETED
  #                   "highRes",               #RESOLUTION
  #                   "rWork",                 #RFACT
  #                   "rFree"]                 #RFREE

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
                      "clusterNumber30",
                      ]  
 
  if not os.path.exists("pdb-chains.csv"):  
    download_custom_report(download_columns, "pdb-chains.csv")

  _structures = {}  

  with open("pdb-chains.csv") as f_1, open("pdb-chains-short.csv", "w") as out_1:
    reader_1 = csv.reader(f_1)
    writer_1 = csv.writer(out_1)
    writer_1.writerow(["structureId",
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
                     "clusterNumber30",])
    for row in reader_1:
      #if "X-RAY DIFFRACTION" in row and "Polypeptide(L)" in row and row[1]=="A":
      if "X-RAY DIFFRACTION" in row and "Polypeptide(L)" in row:  
        writer_1.writerow(row)
  
  pdb_lst = []
  with open("pdb-chains-short.csv", newline="") as f_2:    
    for row in csv.DictReader(f_2):
      pdb_lst.append(row["structureId"])


  with open("pdb-single-chains-short.csv", "w", newline = "") as out_2a:
    writer_2a = csv.writer(out_2a)
    writer_2a.writerow(["structureId",
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
                       "clusterNumber30",])


  
  for e in pdb_lst:
    print(e)
    counter = 0
    with open("pdb-chains-short.csv", newline="") as f_3:
#    with open("pdb-chains-short.csv", newline="") as f, open("pdb-single-chains-short.csv", "w") as out:
#      writer = csv.writer(out)
#      writer.writerow(["structureId",
#                       "chainId",
#                       "entityMacromoleculeType",
#                       "experimentalTechnique",
#                       "resolution",
#                       "rWork",
#                       "rFree",
#                       "clusterNumber95",
#                       "clusterNumber90",
#                       "clusterNumber70",
#                       "clusterNumber50",
#                       "clusterNumber40",
#                       "clusterNumber30",])

      for row in csv.DictReader(f_3):
        if row["structureId"] == str(e):
          print("Found match for", e)
          counter += 1
          print(counter)
    if counter > 1:
      print("more than 1 chain in", e)
      
    if counter == 1:
      print("single entry for", e) 
#      with open("pdb-chains-short.csv", newline="") as f, open("pdb-single-chains-short.csv", "w") as out:
#      with open("pdb-single-chains-short.csv", "w", newline="") as out_2:
#        writer_2 = csv.writer(out_2)
#        writer_2.writerow(["structureId",
#                       "chainId",
#                       "entityMacromoleculeType",
#                       "experimentalTechnique",
#                       "resolution",
#                       "rWork",
#                       "rFree",
#                       "clusterNumber95",
#                       "clusterNumber90",
#                       "clusterNumber70",
#                       "clusterNumber50",
#                       "clusterNumber40",
#                       "clusterNumber30",])

      
      with open("pdb-chains-short.csv", newline="") as f_4, open("pdb-single-chains-short.csv", "a", newline="") as out_3:
        #for row in csv.DictReader(f_4):
        reader_4 = csv.reader(f_4)
        writer_3 = csv.writer(out_3)
#        writer_3.writerow(["structureId",
#                       "chainId",
#                       "entityMacromoleculeType",
#                       "experimentalTechnique",
#                       "resolution",
#                       "rWork",
#                       "rFree",
#                       "clusterNumber95",
#                       "clusterNumber90",
#                       "clusterNumber70",
#                       "clusterNumber50",
#                       "clusterNumber40",
#                       "clusterNumber30",])

        for row in reader_4:  
          #if row["structureId"] == str(e):
          if str(e) in row:
            print(row)
            writer_3.writerow(row)
            #with open("pdb-single-chains-short.csv", "a", newline="") as out_3: 
            #  writer_3 = csv.writer(out_3)
            #  writer_3.writerow(row)


            #writer = csv.writer(out)


  
#  if counter > 1:
#    print("more than 1 chain in", e)    
  
#  if counter == 1:    
#    with open("pdb-single-chains-short.csv", "w") as out:
#      writer = csv.writer(out)
#      writer.writerow(["structureId",
#                       "chainId",
#                       "entityMacromoleculeType",
#                       "experimentalTechnique",
#                       "resolution",
#                       "rWork",
#                       "rFree",
#                       "clusterNumber95",
#                       "clusterNumber90",
#                       "clusterNumber70",
#                       "clusterNumber50",
#                       "clusterNumber40",
#                       "clusterNumber30",])
#    
#    if counter == 1 and str(e) == row["structureId"]:
#        print(row)
      
        
        

#  #with open("pdb-chains-short.csv", newline="") as f, open("pdb-single-chains-short.csv", "w") as out:
#  with open("pdb-chains-short.csv", newline="") as f:  
#    #reader = csv.reader(f)
#    #writer = csv.writer(out)
#    writer.writerow(["structureId",
#                     "chainId",
#                     "entityMacromoleculeType",
#                     "experimentalTechnique",
#                     "resolution",
#                     "rWork",
#                     "rFree",
#                     "clusterNumber95",
#                     "clusterNumber90",
#                     "clusterNumber70",
#                     "clusterNumber50",
#                     "clusterNumber40",
#                     "clusterNumber30",])
#    
#    
#
#    #with open('filename.csv') as f:
#    data = list(csv.reader(f))
#    new_data = [a for i, a in enumerate(data) if a not in data[:i]]
#    with open("pdb-single-chains-short.csv", "w") as out:
#    #with open('filename.csv', 'w') as t:
#      writer = csv.writer(out)
#      writer.writerow(["structureId",
#                     "chainId",
#                     "entityMacromoleculeType",
#                     "experimentalTechnique",
#                     "resolution",
#                     "rWork",
#                     "rFree",
#                     "clusterNumber95",
#                     "clusterNumber90",
#                     "clusterNumber70",
#                     "clusterNumber50",
#                     "clusterNumber40",
#                     "clusterNumber30",])
#
#      writer.writerows(new_data)
    
    
    
      
    #print(pdb_lst)
    
    #counter = 0
    
    #for e in pdb_lst:
    #  print(e)
    #  print(reader)
      
    #for row in reader:
    #  print(row)
    #    # Check the first (0-th) column.
    #    if row[0] == str(e):
    #      # Found the row we were looking for.
    #      counter += 1
    #      print(counter)

      
      
      
#      search = findall(str(e), reader)
#      print(search)
      
      
      
#      print(e)
#      for row in csv.DictReader(f):
#        print(row)
#        if str(e) == row["structureId"]:
#          #print(e)
#          counter += 1
#          print(counter)
    
    
    
                     
  with open("pdb-single-chains-short.csv") as f:   
    for row in csv.DictReader(f):
      structure_id = row["structureId"]
      chain_id = row["chainId"]
      if structure_id not in _structures:
        _structures[structure_id] = _Structure(row)
      _structures[structure_id].chains[chain_id] = _Chain(row)

#  for subdirs, dirs, files in os.walk(pdb_dir):
#    for filename in files:
#      if filename == "data.json":
#        file_path = os.path.join(subdirs, filename)
#        pdb_name = str(file_path.split("/")[-2].upper())
#        json_data = json.load(open(file_path))
#        with open("pdb-chains-short.csv") as f:
#          for row in csv.DictReader(f):
#            if pdb_name == row["structureId"]:
#              row["highRes"] = json_data["RESOLUTION"]
#              row["completeness"] = json_data["COMPLETED"]
#              row["rWork"] = json_data["RFACT"]
#              row["rFree"] = json_data["RFREE"]
#            print(row)
          
def download_custom_report(columns, path):
  """Download a custom report for all structures in the PDB"""
  url = "https://www.rcsb.org/pdb/rest/customReport.xml?pdbids=*&"
  url += "customReportColumns=%s&" % ",".join(columns)
  url += "format=csv&service=wsfile"
  urllib.request.urlretrieve(url, path)

# This is my alternative; looking at structures of local PDB copy
#def structures(pdb_dir):
def structures():
  """Get a dictionary of X-ray structures with protein chains"""
  if _structures is None:
    #_get_structures(pdb_dir)
    _get_structures()
  return copy.deepcopy(_structures)

def cluster_number(structure_id, chain_id, cluster_level):
  """Return the cluster number for an individual protein chain"""
  assert(cluster_level in {95, 90, 70, 50, 40, 30})
  if _structures is None: _get_structures()
  if structure_id in _structures:
    if chain_id in _structures[structure_id].chains:
      attr = "cluster%d" % cluster_level
      return getattr(_structures[structure_id].chains[chain_id], attr)
