import os
import re
import time
import pandas as pd
import requests
from Bio.PDB import MMCIFParser

# ============================================================
# CONFIG
# ============================================================
INPUT_MATRIX_CSV = "curated_antibody_matrix.csv"
DOWNLOAD_DIR = "pdb_structures"
OUTPUT_SEQUENCE_CSV = "final_structural_sequence_dataset.csv"

# Create a clean directory for storing downloaded mmCIF structures
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

# Canonical 3-letter to 1-letter amino acid dictionary
AA_MAP = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
}

def extract_sequence_from_cif(cif_path, target_chains):
    """Parses a local mmCIF file and aggregates sequence strings for explicit chains."""
    if not os.path.exists(cif_path) or not target_chains:
        return ""
    
    parser = MMCIFParser(QUIET=True)
    try:
        structure = parser.get_structure("pdb_structure", cif_path)
        chain_sequences = []
        
        # Look through the first model in the coordinate file
        model = structure[0]
        for chain_id in target_chains:
            if chain_id in model:
                seq_chars = []
                for residue in model[chain_id]:
                    # Exclude water (HOH) and hetero-atoms/ligands from sequence lines
                    if residue.id[0] == " ": 
                        res_name = residue.get_resname().upper()
                        seq_chars.append(AA_MAP.get(res_name, 'X')) # 'X' acts as fallback for non-standard modifications
                
                if seq_chars:
                    chain_sequences.append("".join(seq_chars))
                    
        # Return the primary found chain's sequence to avoid double-dumping trimer duplicates
        return chain_sequences[0] if chain_sequences else ""
    except Exception as e:
        return ""

# ============================================================
# 1. Load Mapped Input Sheet
# ============================================================
if not os.path.exists(INPUT_MATRIX_CSV):
    print(f"Error: Could not find mapping manifest matrix '{INPUT_MATRIX_CSV}'.")
    exit()

df = pd.read_csv(INPUT_MATRIX_CSV)
total_rows = len(df)
print(f"Loaded {total_rows} structural complexes from file matrix manifest.")

final_dataset = []

# ============================================================
# 2. Dual Downloading and Direct Coordinate Sequence Extraction
# ============================================================
for idx, row in df.iterrows():
    pdb_id = str(row["PDB_ID"]).strip().upper()
    mutation_key = row["Group_Mutation_Key"]
    title = row["Complex_Title"]
    
    # Process comma-delimited mapping markers back to Python lists
    antigen_chains = [c.strip().upper() for c in str(row["Antigen_Chains"]).split(",") if c.strip()]
    heavy_chains = [c.strip().upper() for c in str(row["Heavy_Chains"]).split(",") if c.strip()] if str(row["Heavy_Chains"]) != "Unknown" else []
    light_chains = [c.strip().upper() for c in str(row["Light_Chains"]).split(",") if c.strip()] if str(row["Light_Chains"]) != "Unknown" else []
    
    cif_filename = f"{pdb_id.lower()}.cif"
    cif_path = os.path.join(DOWNLOAD_DIR, cif_filename)
    
    print(f"[{idx+1}/{total_rows}] Processing PDB Structure: {pdb_id} -> ", end="", flush=True)
    
    # Download file if it doesn't already exist in your directory cache
    if not os.path.exists(cif_path):
        url = f"https://files.rcsb.org/download/{pdb_id}.cif"
        try:
            time.sleep(0.2) # Micro pacing break
            res = requests.get(url, timeout=20)
            if res.status_code == 200:
                with open(cif_path, "wb") as f:
                    f.write(res.content)
            else:
                print(f"⚠️ Download failed (HTTP Status: {res.status_code})")
                continue
        except Exception as e:
            print(f"❌ Network issue encountered: {str(e)}")
            continue

    # Extract sequences right from the coordinate file's backbone coordinates
    antigen_seq = extract_sequence_from_cif(cif_path, antigen_chains)
    heavy_seq = extract_sequence_from_cif(cif_path, heavy_chains)
    light_seq = extract_sequence_from_cif(cif_path, light_chains)
    
    final_dataset.append({
        "PDB_ID": pdb_id,
        "Group_Mutation_Key": mutation_key,
        "Complex_Title": title,
        "Antigen_Chains": row["Antigen_Chains"],
        "Heavy_Chains": row["Heavy_Chains"],
        "Light_Chains": row["Light_Chains"],
        "Antigen_Sequence_AA": antigen_seq if antigen_seq else "Missing",
        "Heavy_Sequence_AA": heavy_seq if heavy_seq else "Missing",
        "Light_Sequence_AA": light_seq if light_seq else "Missing"
    })
    print(f"✅ Saved (Antigen: {len(antigen_seq)} AA | Heavy: {len(heavy_seq)} AA | Light: {len(light_seq)} AA)")

# ============================================================
# 3. Save Final Dataset Sheet
# ============================================================
if final_dataset:
    out_df = pd.DataFrame(final_dataset)
    out_df.to_csv(OUTPUT_SEQUENCE_CSV, index=False)
    print("\n" + "=" * 95)
    print(f"📊 PIPELINE COMPLETION MATRIX SUCCESSFUL!")
    print(f"📁 Local structural structures saved to folder: /{DOWNLOAD_DIR}")
    print(f"📁 Sequence alignment sheet written cleanly to: {OUTPUT_SEQUENCE_CSV}")
    print("=" * 95)