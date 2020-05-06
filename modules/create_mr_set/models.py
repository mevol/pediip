import json
import os
import shutil

class _Base:
  def __init__(self, directory):
    self.global_id = directory.replace("/", "_")
    self.directory = directory
    self.metadata = {}
    if os.path.exists(self.path("metadata.json")):
      with open(self.path("metadata.json")) as f:
        self.metadata = json.load(f)
    self.jobs = {}
    os.makedirs(self.directory, exist_ok=True)

  def path(self, filename):
    return os.path.join(self.directory, filename)

  def add_metadata(self, key, value):
    self.metadata[key] = value
    with open(self.path("metadata.json"), "w") as f:
      json.dump(self.metadata, f, sort_keys=True, indent=2)

  def remove_directory(self):
    shutil.rmtree(self.directory)

class Structure(_Base):
  def __init__(self, rcsb_structure):
    self.id = rcsb_structure.id
    super().__init__(os.path.join("structures", self.id))
    existing_chains = os.listdir(self.path("chains"))
    if len(existing_chains) > 0:
      self.chains = {cid: Chain(cid, self) for cid in existing_chains}
    else:
      self.chains = {cid: Chain(cid, self) for cid in rcsb_structure.chains}
      self.add_metadata("reported_resolution", rcsb_structure.resolution)
      self.add_metadata("reported_rwork", rcsb_structure.rwork)
      self.add_metadata("reported_rfree", rcsb_structure.rfree)

class Chain(_Base):
  def __init__(self, chain_id, structure):
    self.id = chain_id
    super().__init__(os.path.join(structure.directory, "chains", self.id))
    self.structure = structure
    self.homologues = {}

class Homologue(_Base):
  def __init__(self, hit_pdb, hit_chain, chain):
    self.hit_pdb = hit_pdb
    self.hit_chain = hit_chain
    self.id = "%s_%s" % (hit_pdb, hit_chain)
    super().__init__(os.path.join(chain.directory, "homologues", self.id))
    self.chain = chain
    self.chain.homologues[self.id] = self
    self.todo = len(self.metadata) == 0
