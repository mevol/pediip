import gemmi

def count_elements(xyzin):
  model = gemmi.read_structure(xyzin)[0]
  counts = {}
  for chain in model:
    for residue in chain:
      for atom in residue:
        element = str(atom.element.name)
        if element not in counts:
          counts[element] = 0
        counts[element] += 1
  return counts

def has_atoms(pdbin):
  with open(pdbin) as f:
    for line in f:
      if line[:6] == "ATOM  ":
        return True
  return False
