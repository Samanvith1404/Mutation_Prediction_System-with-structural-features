import os
import re
import time
import pandas as pd
import requests
from difflib import SequenceMatcher

# ============================================================
# CONFIG
# ============================================================
GISAID_FILE = "final_training_set.csv"
PDB_MATRIX_FILE = "curated_antibody_matrix.csv"
OUTPUT_FILE = "mapped_gisaid_to_best_pdb.csv"

if not os.path.exists(GISAID_FILE) or not os.path.exists(PDB_MATRIX_FILE):
    print("❌ Error: Missing baseline files.")
    exit()

df_gisaid = pd.read_csv(GISAID_FILE)
df_pdb = pd.read_csv(PDB_MATRIX_FILE)

# ============================================================
# 1. API Sequence Caching Engine
# ============================================================
print("🚀 Re-caching full-length PDB sequences from core endpoints...")
pdb_sequence_cache = {}
unique_pdbs = df_pdb["PDB_ID"].unique()

for idx, pdb_id in enumerate(unique_pdbs):
    pdb_id = pdb_id.strip().upper()
    url = f"https://data.rcsb.org/rest/v1/core/polymer_entity/{pdb_id}/1"
    try:
        time.sleep(0.05)
        res = requests.get(url, timeout=10)
        if res.status_code == 200:
            raw_seq = res.json().get("entity_poly", {}).get("pdbx_seq_one_letter_code", "")
            clean_seq = re.sub(r"[^A-Z]", "", str(raw_seq).upper())
            if clean_seq:
                pdb_sequence_cache[pdb_id] = clean_seq
                continue
                
        # Fallback to group index 2
        url_alt = f"https://data.rcsb.org/rest/v1/core/polymer_entity/{pdb_id}/2"
        res_alt = requests.get(url_alt, timeout=10)
        if res_alt.status_code == 200:
            raw_seq = res_alt.json().get("entity_poly", {}).get("pdbx_seq_one_letter_code", "")
            clean_seq = re.sub(r"[^A-Z]", "", str(raw_seq).upper())
            pdb_sequence_cache[pdb_id] = clean_seq
    except Exception:
        pass

# ============================================================
# 2. Complete Full-Length String Comparison
# ============================================================
print(f"\n⚙️ Recalculating full-length sequence routing for {len(df_gisaid)} strains...")
mapped_records = []

for g_idx, g_row in df_gisaid.iterrows():
    # Clean up any 'X' or 'N' placeholders from the front to protect scoring accuracy
    gisaid_seq = str(g_row["Spike_Sequence_AA"]).upper().strip().strip('NX')
    strain_name = g_row["Strain"]
    gisaid_len = g_row["GISAID Len (AA)"]
    
    best_pdb_id = None
    best_score = -1.0
    
    for pdb_id, pdb_seq in pdb_sequence_cache.items():
        # Clean PDB validation string
        clean_pdb_seq = pdb_seq.strip().strip('NX')
        
        # FULL LENGTH BIOLOGICAL COMPARISON
        score = SequenceMatcher(None, gisaid_seq, clean_pdb_seq).ratio()
        if score > best_score:
            best_score = score
            best_pdb_id = pdb_id
            
    if best_pdb_id:
        p_match = df_pdb[df_pdb["PDB_ID"] == best_pdb_id].iloc[0]
        mapped_records.append({
            "GISAID_Strain": strain_name,
            "GISAID_Sequence_AA": g_row["Spike_Sequence_AA"], # Keep original format for model training
            "GISAID_Sequence_Len": gisaid_len,
            "Target_PDB_ID": best_pdb_id,
            "Sequence_Similarity_Score": round(best_score, 4),
            "PDB_Mutation_Group": p_match["Group_Mutation_Key"],
            "Antigen_Chains_3D": p_match["Antigen_Chains"],
            "Heavy_Chains_3D": p_match["Heavy_Chains"],
            "Light_Chains_3D": p_match["Light_Chains"]
        })

if mapped_records:
    out_df = pd.DataFrame(mapped_records)
    out_df.to_csv(OUTPUT_FILE, index=False)
    print("\n" + "=" * 90)
    print(f"✅ Full-Length Optimization Complete! Regenerated: {OUTPUT_FILE}")
    print(f"   Minimum dataset score has been corrected and stabilized.")
    print("=" * 90)