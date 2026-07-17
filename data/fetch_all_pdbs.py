import requests
import pandas as pd

url = "https://search.rcsb.org/rcsbsearch/v2/query"

query = {
    "query": {
        "type": "group",
        "logical_operator": "and",
        "nodes": [
            {
                "type": "terminal",
                "service": "text",
                "parameters": {
                    "attribute": "rcsb_entity_source_organism.scientific_name",
                    "operator": "exact_match",
                    "value": "Severe acute respiratory syndrome coronavirus 2"
                }
            },
            {
                "type": "terminal",
                "service": "text",
                "parameters": {
                    "attribute": "rcsb_polymer_entity.rcsb_macromolecular_names_combined.name",
                    "operator": "contains_phrase",
                    "value": "Spike glycoprotein"
                }
            }
        ]
    },
    "return_type": "entry",
    "request_options": {
        "return_all_hits": True
    }
}

response = requests.post(url, json=query)
response.raise_for_status()

hits = response.json()["result_set"]

pdb_ids = sorted([x["identifier"] for x in hits])

df = pd.DataFrame({
    "pdb_id": pdb_ids
})

df.to_csv("spike_pdb_ids.csv", index=False)

print(f"Total PDBs : {len(df)}")
print(df.head(20))
print("\nSaved as spike_pdb_ids.csv")
