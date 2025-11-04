import pandas as pd

# --- Load and inspect ---
edges = pd.read_csv("gene_attribute_edges.txt", sep="\t", low_memory=False)
print("Columns:", edges.columns.tolist())
print(edges.head())

# --- Basic cleaning ---
edges = edges.dropna(subset=["source_id", "target_desc", "weight"])

# Convert weight to numeric safely
edges["weight"] = pd.to_numeric(edges["weight"], errors="coerce").fillna(0.0)

# Simplify dataset
df = edges[["source", "source_id", "target_desc", "weight"]].copy()
df.rename(columns={
    "source": "GeneSymbol",
    "source_id": "SNP_ID",
    "target_desc": "Disease",
    "weight": "AssociationWeight"
}, inplace=True)

# Create a binary target label: 1 if positive association, else 0
df["Target"] = (df["AssociationWeight"] > 0).astype(int)

# Remove duplicates
df = df.drop_duplicates(subset=["SNP_ID", "Disease"])

# Quick stats
print("\n✅ Summary:")
print(f"Rows: {len(df)}")
print(f"Unique SNPs: {df['SNP_ID'].nunique()}")
print(f"Unique Diseases: {df['Disease'].nunique()}")

# Save to CSV
df.to_csv("cleaned_snp_dataset.csv", index=False)
print("\n✅ cleaned_snp_dataset.csv created successfully!")
