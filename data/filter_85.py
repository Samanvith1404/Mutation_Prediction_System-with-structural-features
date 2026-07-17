import os
import pandas as pd

# ============================================================
# CONFIG
# ============================================================
INPUT_CSV = "alignment_report_with_sequences.csv"
OUTPUT_FINAL_CSV = "final_training_set.csv"

# ============================================================
# 1. Load Data Matrix
# ============================================================
if not os.path.exists(INPUT_CSV):
    print(f"Error: {INPUT_CSV} not found. Ensure you run your sequence extraction script first.")
    exit()

df = pd.read_csv(INPUT_CSV)
total_raw_records = len(df)

print(f"Loaded {total_raw_records} raw records from master matrix sheet.")

# ============================================================
# 2. Apply the >= 85% Biological Evolutionary Filter
# ============================================================
# Filter for true full-length functional proteins matching at least 85% similarity
filtered_df = df[df['Identity (%)'] >= 85.0].copy()

# Recalculate and update the Passed validation flag to explicitly show the new 85% boundary line
filtered_df['Passed (>=85%)'] = "YES"

# Drop the old 90% column if it exists to keep our structural database pristine
if 'Passed (>=90%)' in filtered_df.columns:
    filtered_df = filtered_df.drop(columns=['Passed (>=90%)'])

# Reorder columns logically to keep the sequence array easily scannable
column_order = [
    'File Name', 'Strain', 'GISAID Len (AA)', 'Identity (%)', 
    'Matches', 'Mismatches', 'Passed (>=85%)', 'Spike_Sequence_AA'
]
filtered_df = filtered_df[column_order]

# ============================================================
# 3. Save Final Asset Data & Print Execution Summary
# ============================================================
filtered_df.to_csv(OUTPUT_FINAL_CSV, index=False)

kept_count = len(filtered_df)
dropped_count = total_raw_records - kept_count

print("\n" + "=" * 65)
print("📊 FINAL WORKSPACE DATASET FILTER COMPLETE")
print("=" * 65)
print(f"✅ Total Valid Evolutionary Strains KEPT (>=85%) : {kept_count}")
print(f"❌ Corrupted / Fragmented Strains DROPPED (<85%)  : {dropped_count}")
print(f"📁 Verified Training Dataset Manifest Exported to : {OUTPUT_FINAL_CSV}")
print("=" * 65)

# Quick terminal preview of the lowest-identity variant sequences we successfully saved
print("\n🔍 Preview of the most drifted evolutionary strains now saved in your dataset:")
print(filtered_df.sort_values(by='Identity (%)').head(5)[['File Name', 'GISAID Len (AA)', 'Identity (%)']])