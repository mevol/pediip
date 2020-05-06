import glob
import multiprocessing
import os
import subprocess
import sys
import traceback

class ProgressBar:
  def __init__(self, task, total, milestones=5):
    self.done = 0
    self.total = total
    step = int(total / milestones)
    self.milestones = {0 + i * step for i in range(milestones)}
    self.active = False
    print("%s ..." % task)
    if sys.stdout.isatty():
      self.draw()

  def increment(self):
    self.done += 1
    if sys.stdout.isatty() or self.done in self.milestones:
      self.draw()

  def draw(self):
    hashes = "#" * round(self.done / self.total * 60)
    dashes = "-" * (60 - len(hashes))
    bar = "|%s%s| %d/%d" % (hashes, dashes, self.done, self.total)
    if sys.stdout.isatty():
      if self.active:
        bar = "\r" + bar
      print(bar, end="")
      self.active = True
    else:
      print(bar)

  def finish(self):
    self.draw()
    if sys.stdout.isatty():
      print("")
    self.active = False

def run(executable, args=[], stdin=[], stdout=None, stderr=None):
  pstdin = subprocess.PIPE if len(stdin) > 0 else None
  pstdout = None if stdout is None else open(stdout, "w")
  pstderr = None if stderr is None else open(stderr, "w")
  command = [executable] + args
  p = subprocess.Popen(command,
    stdin=pstdin, stdout=pstdout, stderr=pstderr, encoding="utf8")
  if pstdin == subprocess.PIPE:
    for line in stdin:
      p.stdin.write(line + "\n")
    p.stdin.close()
  p.wait()

def parallel(title, func, dictionary, processes=None):
  progress_bar = ProgressBar(title, len(dictionary))
  def callback(item):
    key, value = item
    dictionary[key] = value
    progress_bar.increment()
  def error_callback(exc):
    traceback.print_exception(type(exc), exc, exc.__traceback__)
  pool = multiprocessing.Pool(processes)
  for key, value in dictionary.items():
    pool.apply_async(func, args=(key, value), callback=callback, error_callback=error_callback)
  pool.close()
  pool.join()
  progress_bar.finish()

def remove_errors(model_dictionary):
  model_type = type(next(iter(model_dictionary.values()))).__name__.lower()
  error_counts = {}
  for model_id, model in list(model_dictionary.items()):
    for job_id, result in model.jobs.items():
      if "error" in result:
        message = "%s: %s" % (job_id, result["error"])
        if message not in error_counts: error_counts[message] = 0
        error_counts[message] += 1
        model.add_metadata("error", message)
        del model_dictionary[model_id]
  if len(error_counts) > 0:
    print("Some %ss were removed due to errors:" % model_type)
    for error in error_counts:
      print("%s (%d removed)" % (error, error_counts[error]))
  if len(model_dictionary) < 1:
    sys.exit("No %ss left after removing errors!" % model_type)

def is_semet(pdbin):
  """Check if a PDB format file is a selenomethione derivative"""
  with open(pdbin) as f:
    for line in f:
      if line[:6] == "ATOM  " or line[:6] == "HETATM":
        if line[17:20] == "MET": return False
        if line[17:20] == "MSE": return True
  return False

def print_section_title(title):
  print("-" * len(title))
  print(title.upper())
  print("-" * len(title) + "\n")

def remove_files_starting_with(prefix):
  pattern = "%s*" % prefix
  for filename in glob.glob(pattern):
    os.remove(filename)
