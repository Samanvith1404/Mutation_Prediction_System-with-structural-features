# Data Pipeline — Structural Mutation Mapping

This folder contains the scripts used to build the SARS-CoV-2 Spike structural
mutation dataset: fetching reference structures, aligning GISAID strain
sequences against them, isolating Spike-specific mutations, mapping strains to
the best-matching PDB structure, and validating the result geometrically.

## Pipeline order

The scripts are meant to run roughly in this sequence (outputs of one feed the
next):

1. **`fetch_all_pdbs.py`**
   Queries the RCSB PDB search API for every structure containing a SARS-CoV-2
   Spike glycoprotein and saves the list of PDB IDs to `spike_pdb_ids.csv`.

2. **`map.py`**
   Downloads the canonical UniProt Spike reference sequence (P0DTC2), then
   walks a folder of GISAID strain JSON files, translates/normalizes each
   strain's sequence (including raw-nucleotide → protein translation when
   needed), globally aligns it against the reference, and writes an identity
   report (`alignment_report_with_sequences.csv`).

3. **`filter_85.py`**
   Filters that alignment report down to strains with ≥85% identity to the
   reference, producing the cleaned `final_training_set.csv`.

4. **`map_seq.py`**
   Downloads mmCIF structure files for each mapped PDB complex and extracts
   antigen/heavy-chain/light-chain amino-acid sequences directly from the
   3D coordinates, saving `final_structural_sequence_dataset.csv`.

5. **`map_pdb_seq.py`**
   Pairwise-aligns each GISAID Spike sequence against each PDB antigen
   sequence to build a mutation footprint (substitutions/insertions/
   deletions) relative to PDB residue numbering, keeping only high-identity
   (≥85%) matches (`gisaid_to_pdb_mutation_map.csv`).

6. **`filter_pdbmap.py`**
   For each strain in the training set, re-fetches full-length PDB sequences
   from the RCSB REST API and picks the single best-matching PDB structure
   by full-sequence similarity, attaching antigen/heavy/light chain IDs
   (`mapped_gisaid_to_best_pdb.csv`).

7. **`extract_aa.py`**
   Strips the mutation list down to Spike-only (`S:`) substitutions, dropping
   mutations from other genes (N, M, E, ORFs), producing the final training
   table (`final_spike_only_dataset.csv`).

8. **`validate_data.py`**
   Sanity-check/visualization step: loads the sphere-mapped dataset, parses
   each structure's Cα coordinates, and renders 3D matplotlib plots showing
   the antigen backbone, background mutations, and interface mutations
   (with search-radius spheres) for visual QA.

## Supporting / alternate-path scripts

- **`fetch_all_pdbs.py`** — see step 1 above (structure discovery).
- **`filter_pdbmap.py`** — alternate/updated version of the PDB-matching step
  using live RCSB sequence lookups instead of a pre-built matrix.
- **`95_same.py`** — standalone, multiprocessing-based large-scale pipeline
  that scans thousands of strain "shard" JSON files, translates each in all
  3 reading frames, and matches them against a cached PDB sequence database
  using a sliding-window identity check. Streams matches to
  `structural_mutation_map.jsonl` via a dedicated background writer process
  for high-throughput, low-memory disk I/O. Used for full-dataset scale runs
  rather than the smaller per-folder scripts above.

## File reference

| File | Purpose | Key input(s) | Key output(s) |
|---|---|---|---|
| `fetch_all_pdbs.py` | Discover all Spike-containing PDB entries via RCSB search API | — (web API) | `spike_pdb_ids.csv` |
| `map.py` | Align raw GISAID strain JSONs to the UniProt Spike reference | strain `.json` files, UniProt FASTA | `alignment_report_with_sequences.csv` |
| `filter_85.py` | Keep only strains ≥85% identity to reference | `alignment_report_with_sequences.csv` | `final_training_set.csv` |
| `map_seq.py` | Extract antigen/heavy/light sequences from downloaded mmCIF structures | `curated_antibody_matrix.csv`, `.cif` files | `final_structural_sequence_dataset.csv` |
| `map_pdb_seq.py` | Align GISAID sequences to PDB antigen sequences, derive mutation footprints | `final_structural_sequence_dataset.csv`, GISAID sequences | `gisaid_to_pdb_mutation_map.csv` |
| `filter_pdbmap.py` | Pick best-matching PDB per strain via live RCSB sequence fetch | `final_training_set.csv`, `curated_antibody_matrix.csv` | `mapped_gisaid_to_best_pdb.csv` |
| `extract_aa.py` | Isolate Spike-only (`S:`) mutation tokens | `final_dataset_with_mutations.csv` | `final_spike_only_dataset.csv` |
| `validate_data.py` | 3D visualization/QA of interface vs. background mutations | `sphere_radius_mapped_dataset.csv`, `.cif` files | PNG plots in `all_final_geometric_plots/` |
| `95_same.py` | Large-scale, multiprocess strain-to-PDB structural matching over sharded data | `cached_pdb_sequences.json`, shard `.json` files | `structural_mutation_map.jsonl` |

## Notes

- Several scripts (`map.py`, `map_seq.py`, `95_same.py`) reference local drive
  paths (e.g. `E:\NANI\...`, `D:\Evolution_antigens\...`) from the original
  development machine — update these constants at the top of each file before
  rerunning.
- The identity threshold is 90% in `map.py` but the downstream filter in
  `filter_85.py` uses 85% — check which threshold is authoritative before
  reusing the reference alignment report.
- `Bio.Align.PairwiseAligner` (Biopython) is used for global alignment in
  `map.py` and `map_pdb_seq.py`; `Bio.PDB.MMCIFParser` is used for structure
  parsing in `map_seq.py` and `validate_data.py`.
