import os
import json
import re
import csv
import requests
from Bio.Align import PairwiseAligner

# ============================================================
# CONFIG
# ============================================================
FOLDER_PATH = r"E:\NANI\shards\I\N" 
UNIPROT_URL = "https://rest.uniprot.org/uniprotkb/P0DTC2.fasta"
OUTPUT_CSV = "alignment_report_with_sequences.csv"

# Comprehensive Amino Acid Translation Map for 3-Letter Code Fallbacks
AA3_TO_AA1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "SEC": "U", "PYL": "O", "ASX": "B", "GLX": "Z", "XLE": "J", "UNK": "X"
}

# Standard Genetic Codon Translation Table
CODON_TABLE = {
    'ATA':'I', 'ATC':'I', 'ATT':'I', 'ATG':'M', 'ACA':'T', 'ACC':'T', 'ACG':'T', 'ACT':'T',
    'AAC':'N', 'AAT':'N', 'AAA':'K', 'AAG':'K', 'AGC':'S', 'AGT':'S', 'AGA':'R', 'AGG':'R',
    'CTA':'L', 'CTC':'L', 'CTG':'L', 'CTT':'L', 'CCA':'P', 'CCC':'P', 'CCG':'P', 'CCT':'P',
    'CAC':'H', 'CAT':'H', 'CAA':'Q', 'CAG':'Q', 'CGA':'R', 'CGC':'R', 'CGG':'R', 'CGT':'R',
    'GTA':'V', 'GTC':'V', 'GTG':'V', 'GTT':'V', 'GCA':'A', 'GCC':'A', 'GCG':'A', 'GCT':'A',
    'GAC':'D', 'GAT':'D', 'GAA':'E', 'GAG':'E', 'GGA':'G', 'GGC':'G', 'GGG':'G', 'GGT':'G',
    'TCA':'S', 'TCC':'S', 'TCG':'S', 'TCT':'S', 'TTC':'F', 'TTT':'F', 'TTA':'L', 'TTG':'L',
    'TAC':'Y', 'TAT':'Y', 'TAA':'*', 'TAG':'*', 'TGC':'C', 'TGT':'C', 'TGA':'*', 'TGG':'W',
}

# ============================================================
# 1. Fetch Clean UniProt Reference Framework
# ============================================================
print("Downloading UniProt P0DTC2 Reference...")
fasta_response = requests.get(UNIPROT_URL).text
uniprot_seq = "".join(fasta_response.split("\n")[1:]).strip().upper()
print(f"UniProt Reference Length: {len(uniprot_seq)} AA\n")

# Initialize Global Aligner Configuration
aligner = PairwiseAligner()
aligner.mode = 'global'

report_data = []

# ============================================================
# 2. Iterate and Process JSON Files
# ============================================================
if not os.path.exists(FOLDER_PATH):
    print(f"Error: The directory path '{FOLDER_PATH}' does not exist.")
    exit()

files_to_process = [f for f in os.listdir(FOLDER_PATH) if f.lower().endswith('.json')]
print(f"Found {len(files_to_process)} JSON files to process.\n")

for file_name in files_to_process:
    file_path = os.path.join(FOLDER_PATH, file_name)
    print(f"Mapping sequence for: {file_name} ... ", end="", flush=True)
    
    try:
        with open(file_path, "r") as f:
            gisaid_data = json.load(f)
            
        strain = gisaid_data.get("strain", file_name)
        raw_sequence_string = gisaid_data.get("sequence", "").strip().upper()
        
        # Parse for space-delimited formats
        tokens = [t.upper() for t in re.split(r"[\s,;\-]+", raw_sequence_string) if t.strip()]
        
        if tokens and any(t in AA3_TO_AA1 for t in tokens):
            gisaid_protein_seq = "".join(AA3_TO_AA1.get(t, "X") for t in tokens)
        else:
            gisaid_protein_seq = raw_sequence_string

        # DYNAMIC EXTRACTION: Detect raw nucleotide genomes and apply Adaptive Reading Frame Scan
        if len(gisaid_protein_seq) > 20000 and set(gisaid_protein_seq).issubset(set("ACGTUNX")):
            gisaid_protein_seq = gisaid_protein_seq.upper()
            
            start_anchor = "ATGTTTGTTTTT"
            start_idx = gisaid_protein_seq.find(start_anchor)
            
            if start_idx == -1:
                # Scan a coordinate neighborhood around the standard expected gene opening locus
                for window_offset in range(21500, 21650):
                    if gisaid_protein_seq[window_offset:window_offset+3] == "ATG":
                        test_idx = window_offset
                        has_early_stop = False
                        for check_step in range(0, 3600, 3):
                            if (test_idx + check_step + 3) > len(gisaid_protein_seq):
                                break
                            current_codon = gisaid_protein_seq[test_idx + check_step : test_idx + check_step + 3]
                            if current_codon in ["TAA", "TAG", "TGA"]:
                                has_early_stop = True
                                break
                        if not has_early_stop:
                            start_idx = window_offset
                            break
                if start_idx == -1:
                    start_idx = 21562  
            
            protein_translation = []
            max_search_length = 3900 
            
            for i in range(start_idx, min(start_idx + max_search_length, len(gisaid_protein_seq) - 2), 3):
                codon = gisaid_protein_seq[i:i+3]
                if len(codon) < 3:
                    break
                amino_acid = CODON_TABLE.get(codon, 'X')
                if amino_acid == '*':
                    break
                protein_translation.append(amino_acid)
                
            gisaid_protein_seq = "".join(protein_translation)

        if not gisaid_protein_seq:
            print("SKIPPED (Null)")
            continue

        # Run Global Sequence Alignment
        alignments = aligner.align(uniprot_seq, gisaid_protein_seq)
        best_alignment = alignments[0]
        
        target_indices, query_indices = best_alignment.indices
        matches = 0
        for t_idx, q_idx in zip(target_indices, query_indices):
            if t_idx != -1 and q_idx != -1:
                if uniprot_seq[t_idx] == gisaid_protein_seq[q_idx]:
                    matches += 1

        identity = (matches / len(uniprot_seq) * 100)
        mismatches = len(uniprot_seq) - matches
        passed = "YES" if identity >= 90.0 else "NO"  
        
        # Inject entries including the raw full amino acid sequence text string directly into row map
        report_data.append({
            "File Name": file_name,
            "Strain": strain,
            "GISAID Len (AA)": len(gisaid_protein_seq),
            "Identity (%)": round(identity, 2),
            "Matches": matches,
            "Mismatches": mismatches,
            "Passed (>=90%)": passed,
            "Spike_Sequence_AA": gisaid_protein_seq  # NEW COLUMN FOR THE COMPLETE MAPPED AMINO ACID STRING
        })
        print(f"Done (Len: {len(gisaid_protein_seq)} AA, Identity: {round(identity, 2)}%)")

    except Exception as e:
        print(f"FAILED (Error: {str(e)})")

# ============================================================
# 3. Output Master CSV Matrix File
# ============================================================
if report_data:
    keys = report_data[0].keys()
    with open(OUTPUT_CSV, "w", newline="") as output_file:
        dict_writer = csv.DictWriter(output_file, fieldnames=keys)
        dict_writer.writeheader()
        dict_writer.writerows(report_data)
    print(f"\n📊 Master alignment database with mapped protein sequences saved to: {OUTPUT_CSV}")