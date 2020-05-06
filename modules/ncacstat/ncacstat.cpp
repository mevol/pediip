#include <clipper/clipper-ccp4.h>
#include <clipper/clipper-minimol.h>
#include <algorithm>

const int mmdbflags = ::mmdb::MMDBF_IgnoreBlankLines | ::mmdb::MMDBF_IgnoreDuplSeqNum | ::mmdb::MMDBF_IgnoreNonCoorPDBErrors | ::mmdb::MMDBF_IgnoreRemarks;

std::vector<std::string> known_types;

void define_known_types()
{
  known_types.push_back("ALA"); known_types.push_back("ARG");
  known_types.push_back("ASN"); known_types.push_back("ASP");
  known_types.push_back("CYS"); known_types.push_back("GLN");
  known_types.push_back("GLU"); known_types.push_back("GLY");
  known_types.push_back("HIS"); known_types.push_back("ILE");
  known_types.push_back("LEU"); known_types.push_back("LYS");
  known_types.push_back("MET"); known_types.push_back("MSE");
  known_types.push_back("PHE"); known_types.push_back("PRO");
  known_types.push_back("SER"); known_types.push_back("THR");
  known_types.push_back("TRP"); known_types.push_back("TYR");
  known_types.push_back("UNK"); known_types.push_back("VAL");
}

clipper::MiniMol read_mol(std::string path)
{
  clipper::MMDBfile mmdb;
  mmdb.SetFlag(mmdbflags);
  mmdb.read_file(path);
  clipper::MiniMol mol;
  mmdb.import_minimol(mol);
  return mol;
}

bool is_known_type(const clipper::MMonomer& monomer)
{
  clipper::String type = monomer.type();
  for (int i = 0; i < known_types.size(); i++) {
    if (type == known_types[i]) return true;
  }
  return false;
}

struct Residue
{
  clipper::String type;
  std::vector<clipper::Coord_frac> coords;
};

std::vector<Residue> residues(const clipper::MiniMol& mol)
{
  std::vector<Residue> residues;
  for (int c = 0; c < mol.size(); c++) {
    for (int r = 0; r < mol[c].size(); r++) {
      if (is_known_type(mol[c][r])) {
        int i1 = mol[c][r].lookup(" N  ", clipper::MM::ANY);
        int i2 = mol[c][r].lookup(" CA ", clipper::MM::ANY);
        int i3 = mol[c][r].lookup(" C  ", clipper::MM::ANY);
        if (i1 < 0 || i2 < 0 || i3 < 0) continue;
        Residue res;
        res.type = mol[c][r].type() == "MSE" ? "MET" : mol[c][r].type();
        res.coords.push_back(mol[c][r][i1].coord_orth().coord_frac(mol.cell()));
        res.coords.push_back(mol[c][r][i2].coord_orth().coord_frac(mol.cell()));
        res.coords.push_back(mol[c][r][i3].coord_orth().coord_frac(mol.cell()));
        residues.push_back(res);
      }
    }
  }
  return residues;
}

bool matching_coords(const Residue& res_ref, const Residue& res_wrk,
                     const double& cutoff, const clipper::Cell& cell,
                     const clipper::Spacegroup& spacegroup)
{
  double cutoff2 = cutoff * cutoff;
  for (int i = 0; i < res_ref.coords.size(); i++) {
    clipper::Coord_frac coord_ref = res_ref.coords[i];
    clipper::Coord_frac coord_wrk = res_wrk.coords[i].symmetry_copy_near(spacegroup, cell, coord_ref);
    double distance2 = (coord_ref - coord_wrk).lengthsq(cell);
    if (distance2 > cutoff2) return false;
  }
  return true;
}

bool matching_type(const Residue& res_ref, const Residue& res_wrk)
{
  if (res_ref.type == "UNK") return true;
  return res_ref.type == res_wrk.type;
}

int main( int argc, char** argv )
{
  std::string path_ref = argv[1];
  std::string path_wrk = argv[2];
  double cutoff = 1.0;
  if (argc > 3) cutoff = clipper::String(argv[3]).f();

  define_known_types();

  clipper::MiniMol mol_ref = read_mol(path_ref);
  clipper::MiniMol mol_wrk = read_mol(path_wrk);
  clipper::Spacegroup spacegroup = mol_wrk.spacegroup();
  clipper::Cell cell = mol_wrk.cell();
  std::vector<Residue> residues_ref = residues(mol_ref);
  std::vector<Residue> residues_wrk = residues(mol_wrk);
  int any_type = 0;
  int same_type = 0;
  for (int r1 = 0; r1 < residues_ref.size(); r1++) {
    Residue res_ref = residues_ref[r1];
    for (int r2 = 0; r2 < residues_wrk.size(); r2++) {
      Residue res_wrk = residues_wrk[r2];
      if (matching_coords(res_ref, res_wrk, cutoff, cell, spacegroup)) {
        any_type++;
        if (matching_type(res_ref, res_wrk)) {
          same_type++;
        }
        break;
      }
    }
  }
  std::cout << residues_ref.size() << " "
            << any_type << " "
            << same_type << std::endl;
}

  // clipper::Spacegroup spgr1 = mmol1.spacegroup();
  // clipper::Spacegroup spgr2 = mmol2.spacegroup();
  // clipper::Cell       cell1 = mmol1.cell();
  // clipper::Cell       cell2 = mmol2.cell();

  // // and turn into atom lists
  // typedef std::pair<clipper::String,clipper::Coord_frac> catype;
  // std::vector<catype> ca1, ca2;
  // for ( int c = 0; c < mmol1.size(); c++ )
  //   for ( int r = 0; r < mmol1[c].size(); r++ ) {
  //     int a = mmol1[c][r].lookup( " CA ", clipper::MM::ANY );
  //     clipper::String type = mmol1[c][r].type();
  //     if ( type == "MSE" ) type = "MET";
  //     if ( a >= 0 ) ca1.push_back(
  //       catype( type, mmol1[c][r][a].coord_orth().coord_frac(cell1) ) );
  //   }
  // for ( int c = 0; c < mmol2.size(); c++ )
  //   for ( int r = 0; r < mmol2[c].size(); r++ ) {
  //     int a = mmol2[c][r].lookup( " CA ", clipper::MM::ANY );
  //     clipper::String type = mmol2[c][r].type();
  //     if ( type == "MSE" ) type = "MET";
  //     if ( a >= 0 ) ca2.push_back(
  //       catype( type, mmol2[c][r][a].coord_orth().coord_frac(cell2) ) );
  //   }

  // // count atoms in lists
  // double n1, n2, n1n2, n2n1, n1m2, n2m1, r2;
  // n1 = ca1.size();
  // n2 = ca2.size();
  // n1n2 = n2n1 = n1m2 = n2m1 = r2 = 0.0;
  // clipper::Coord_frac cf0, cf1, cf2;
  // double d2, d2min, d2cut = pow(rad,2.0);
  // std::vector<int>  match12( ca1.size(),  -1 ), match21( ca2.size(),  -1 );
  // std::vector<bool> resid12( ca1.size(),false), resid21( ca2.size(),false);

  // // count atoms in one list but not the other
  // for ( int i = 0; i < ca1.size(); i++ ) {
  //   int jmin = -1;
  //   d2min = 1.0e9;
  //   cf0 = ca1[i].second;
  //   for ( int j = 0; j < ca2.size(); j++ ) {
  //     cf1 = ca2[j].second;
  //     cf2 = cf1.symmetry_copy_near( spgr2, cell2, cf0 );
  //     d2 = ( cf2 - cf0 ).lengthsq( cell2 );
  //     if ( d2 < d2min ) {
  //       jmin = j;
  //       d2min = d2;
  //     }
  //   }
  //   if ( d2min < d2cut ) { 
  //     match12[i] = jmin;
  //     resid12[i] = ( ca1[i].first == ca2[jmin].first );
  //     n1n2 += 1.0;
  //     if ( ca1[i].first == "UNK" || resid12[i] ) n1m2 += 1.0;
  //     r2 += d2min;
  //   }
  // }

  // // count atoms in one list but not the other
  // for ( int i = 0; i < ca2.size(); i++ ) {
  //   int jmin = -1;
  //   d2min = 1.0e9;
  //   cf0 = ca2[i].second;
  //   for ( int j = 0; j < ca1.size(); j++ ) {
  //     cf1 = ca1[j].second;
  //     cf2 = cf1.symmetry_copy_near( spgr1, cell1, cf0 );
  //     d2 = ( cf2 - cf0 ).lengthsq( cell1 );
  //     if ( d2 < d2min ) {
  //       jmin = j;
  //       d2min = d2;
  //     }
  //   }
  //   if ( d2min < d2cut ) {
  //     match21[i] = jmin;
  //     resid21[i] = ( ca2[i].first == ca1[jmin].first );
  //     n2n1 += 1.0;
  //     if ( ca2[i].first == "UNK" || resid21[i] ) n2m1 += 1.0;
  //     r2 += d2min;
  //   }
  // }

  // std::cout << n1 << " \t" << n2 << " \t" << n1n2 << " \t" << n2n1 << " \t\t" << sqrt(r2/(n1n2+n2n1)) << "   \t" << n1m2 << " \t" << n2m1 << "\n";
