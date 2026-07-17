import os
import re
import pandas as pd

# ============================================================
# CONFIG
# ============================================================
INPUT_DATASET_CSV = "final_dataset_with_mutations.csv"
FINAL_SPIKE_ONLY_CSV = "final_spike_only_dataset.csv"

# ============================================================
# 1. Verification and Loading
# ============================================================
if not os.path.exists(INPUT_DATASET_CSV):
    print(f"❌ Error: Cannot locate '{INPUT_DATASET_CSV}'. Run the extraction script first.")
    exit()

df = pd.read_csv(INPUT_DATASET_CSV)
print(f"Loaded {len(df)} records containing global mutation features.")

# ============================================================
# 2. Extract Spike-Specific Mutations Natively
# ============================================================
print("⚙️ Filtering mutation arrays to isolate 'S:' prefix tokens...")
spike_only_records = []

for idx, row in df.iterrows():
    raw_mutations = str(row.get("AA_Substitutions", "")).strip()
    
    # Handle wild-type matches seamlessly
    if raw_mutations in ["WT_Match", "nan", ""]:
        row_dict = row.to_dict()
        row_dict["Spike_Mutations_Only"] = "WT"
        spike_only_records.append(row_dict)
        continue
        
    # Split the bundled mutations string (handles both spaces or comma delimeters)
    tokens = re.split(r'[,\s]+', raw_mutations)
    spike_tokens = []
    
    for token in tokens:
        token = token.strip()
        # Captures variations like S:T19R, Spike:T19R, or raw entries if already pre-isolated
        if token.upper().startswith("S:") or token.upper().startswith("SPIKE:"):
            # Strip the prefix to keep clean structural notations (e.g., 'T19R')
            clean_token = re.sub(r'^(S:|SPIKE:)', '', token, flags=re.IGNORECASE)
            spike_tokens.append(clean_token)
        elif not any(prefix in token.upper() for prefix in ["N:", "M:", "E:", "ORF"]):
            # Fallback protection: If no structural prefixes exist at all, keep it as a raw Spike variant
            spike_tokens.append(token)
            
    # Reassemble clean comma-separated arrays
    row_dict = row.to_dict()
    row_dict["Spike_Mutations_Only"] = ",".join(spike_tokens) if spike_tokens else "WT"
    spike_only_records.append(row_dict)

# ============================================================
# 3. Export Pristine Final Dataset
# ============================================================
out_df = pd.DataFrame(spike_only_records)

# Drop old raw mutation tracking column to keep feature arrays compact and pristine
if "AA_Substitutions" in out_df.columns:
    out_df = out_df.drop(columns=["AA_Substitutions"])

out_df.to_csv(FINAL_SPIKE_ONLY_CSV, index=False)

print("\n" + "=" * 90)
print("📊 SPIKE PROTEIN ISOLATION SUCCESSFUL!")
print(f"📁 Pristine structural training dataset saved cleanly to: {FINAL_SPIKE_ONLY_CSV}")
print(f"🔬 Preview sample line (Row 1 Spike targets): [{out_df['Spike_Mutations_Only'].iloc[0][:50]}...]")
print("=" * 90)