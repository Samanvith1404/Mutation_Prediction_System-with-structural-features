import os
import re
import pandas as pd
from Bio import Align

# ============================================================
# CONFIGURATION
# ============================================================
PDB_SEQUENCE_CSV = "final_structural_sequence_dataset.csv"
GISAID_DATA_CSV = "gisaid_clean_sequences.csv"  # Replace with your actual GISAID dataset path
OUTPUT_MAPPED_CSV = "gisaid_to_pdb_mutation_map.csv"

# ============================================================
# 1. Initialize Pairwise Aligner Engine
# ============================================================
aligner = Align.PairwiseAligner()
aligner.mode = 'global'
aligner.open_gap_score = -10
aligner.extend_gap_score = -1

def map_sequence_coordinates(pdb_seq, gisaid_seq):
    """
    Aligns a GISAID variant sequence against the exact PDB sequence backbone.
    Returns a mutation footprint string and alignment identity score.
    """
    if not pdb_seq or pdb_seq == "Missing" or not gisaid_seq:
        return "Unknown", 0.0
        
    try:
        alignments = aligner.align(pdb_seq, gisaid_seq)
        best_alignment = alignments[0]
        score = best_alignment.score
        
        # Calculate approximate sequence identity percentage
        max_len = max(len(pdb_seq), len(gisaid_seq))
        identity_pct = (score / max_len) * 100 if max_len > 0 else 0.0
        
        # Trace exact point-mutations relative to PDB index spaces
        aligned_pdb, aligned_gisaid = best_alignment
        mutations = []
        
        pdb_idx = 1  # Track structural coordinate tracking indices
        for p_char, g_char in zip(aligned_pdb, aligned_gisaid):
            if p_char == "-":
                # Insertion in GISAID relative to PDB
                continue
            elif g_char == "-":
                # Deletion in GISAID relative to PDB
                mutations.append(f"del{pdb_idx}")
                pdb_idx += 1
            elif p_char != g_char:
                # Substitution substitution matrix point
                mutations.append(f"{p_char}{pdb_idx}{g_char}")
                pdb_idx += 1
            else:
                pdb_idx += 1
                
        mutation_profile = ",".join(mutations) if mutations else "WT_Match"
        return mutation_profile, round(identity_pct, 2)
        
    except Exception as e:
        return f"Alignment_Error ({str(e)})", 0.0

# ============================================================
# 2. Execution Pipeline
# ============================================================
def main():
    if not os.path.exists(PDB_SEQUENCE_CSV):
        print(f"❌ Error: Cannot locate '{PDB_SEQUENCE_CSV}'. Please ensure previous step finished cleanly.")
        return
        
    print("🚀 Loading PDB structural sequences and GISAID baseline maps...")
    pdb_df = pd.read_csv(PDB_SEQUENCE_CSV)
    
    # Check for placeholder GISAID data file; create template if missing for runtime protection
    if not os.path.exists(GISAID_DATA_CSV):
        print(f"⚠️ Notice: '{GISAID_DATA_CSV}' not found. Generating a mock dataset template for validation...")
        mock_gisaid = pd.DataFrame({
            "GISAID_ID": ["EPI_ISL_111111", "EPI_ISL_222222"],
            "Lineage": ["BA.5", "XBB.1.5"],
            "Spike_Sequence_AA": [
                pdb_df["Antigen_Sequence_AA"].iloc[0],  # Direct exact match test case
                pdb_df["Antigen_Sequence_AA"].iloc[0].replace("N", "K", 1)  # Synthetic single mutation test case
            ]
        })
        mock_gisaid.to_csv(GISAID_DATA_CSV, index=False)
        
    gisaid_df = pd.read_csv(GISAID_DATA_CSV)
    
    print(f"Loaded {len(pdb_df)} PDB target files and {len(gisaid_df)} GISAID evolutionary records.")
    print("Mapping coordinate alignments (this cross-references sequence spaces directly)...")
    
    mapping_records = []
    
    # Cross-product mapping loop (maps each GISAID sequence space variant to each verified structural template)
    for p_idx, p_row in pdb_df.iterrows():
        pdb_id = p_row["PDB_ID"]
        pdb_antigen_seq = str(p_row["Antigen_Sequence_AA"])
        
        if not pdb_antigen_seq or pdb_antigen_seq == "Missing":
            continue
            
        for g_idx, g_row in gisaid_df.iterrows():
            gisaid_id = g_row["GISAID_ID"]
            lineage = g_row.get("Lineage", "Unknown")
            gisaid_seq = str(g_row["Spike_Sequence_AA"])
            
            # Map mutations relative to this specific structural configuration template
            mut_profile, identity = map_sequence_coordinates(pdb_antigen_seq, gisaid_seq)
            
            # Keep records with strong structural sequence alignment identities (e.g., homology boundaries >= 85%)
            if identity >= 85.0:
                mapping_records.append({
                    "PDB_ID": pdb_id,
                    "GISAID_ID": gisaid_id,
                    "Lineage": lineage,
                    "Sequence_Identity_Pct": identity,
                    "PDB_Indexed_Mutations": mut_profile,
                    "Heavy_Chains_Mapped": p_row["Heavy_Chains"],
                    "Light_Chains_Mapped": p_row["Light_Chains"]
                })

    # Export mapping manifest sheet
    if mapping_records:
        out_df = pd.DataFrame(mapping_records)
        out_df.to_csv(OUTPUT_MAPPED_CSV, index=False)
        print("\n" + "=" * 95)
        print(f"📊 SEQUENCE MAPPING ALIGNMENT COMPLETE!")
        print(f"📁 Coordinated dataset matrix exported securely to: {OUTPUT_MAPPED_CSV}")
        print("=" * 95)
    else:
        print("❌ Pipeline completed, but no high-identity sequence mappings were resolved.")

if __name__ == "__main__":
    main()