#!/bin/env python3
import sqlite3
import os

handle = sqlite3.connect('pdb_redo_db.sqlite')

cur = handle.cursor()

cur.executescript('''
      DROP TABLE IF EXISTS pdb_id;
      DROP TABLE IF EXISTS pdb_redo_stats;
      
      CREATE TABLE pdb_id (
          id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT UNIQUE,
          pdb_id  TEXT UNIQUE
      );
      CREATE TABLE pdb_redo_stats (
          pdb_id_id INTEGER,
          rWork_depo TEXT,
          rFree_depo TEXT,
          rWork_tls TEXT,
          rFree_tls TEXT,
          rWork_final TEXT,
          rFree_final TEXT,
          completeness TEXT,          
          FOREIGN KEY (pdb_id_id) REFERENCES pdb_id(id)
      );
      ''')

all_data = "/dls/mx-scratch/ghp45345/pdb-redo-dump/others/alldata_noHeader.txt"
  
with open(all_data, "r") as data_file:
  data = data_file.read().split("\n")
  print(len(data))
  for sample in data:
    structure_id = sample.split(" ")[0].upper()
    rwork_deposited = sample.split(" ")[2]
    rfree_deposited = sample.split(" ")[3]
    rwork_tls = sample.split(" ")[9]
    rfree_tls = sample.split(" ")[10]
    rwork_final = sample.split(" ")[14]
    rfree_final = sample.split(" ")[15]
    completeness = sample.split(" ")[-21]

    cur.executescript( '''
      INSERT OR IGNORE INTO pdb_id
      (pdb_id) VALUES ("%s");
      '''% (structure_id))

    cur.executescript( '''
      INSERT OR IGNORE INTO pdb_redo_stats (pdb_id_id)
      SELECT id FROM pdb_id
      WHERE pdb_id.pdb_id="%s";
      ''' % (structure_id))

    pdb_redo_dict = {
        "rWork_depo"         : rwork_deposited,
        "rFree_depo"         : rfree_deposited,
        "rWork_tls"          : rwork_tls,
        "rFree_tls"          : rfree_tls,
        "rWork_final"        : rwork_final,
        "rFree_final"        : rfree_final,
        "completeness"       : completeness
                }

    cur.execute('''
      SELECT id FROM pdb_id
      WHERE pdb_id="%s"
      ''' % (structure_id))
    pdb_id = cur.fetchone()[0]
    
    for data in pdb_redo_dict:
      cur.execute('''
        UPDATE pdb_redo_stats
        SET "%s" = "%s"
        WHERE pdb_id_id = "%s";
        ''' % (data, pdb_redo_dict[data], pdb_id))
    handle.commit()
