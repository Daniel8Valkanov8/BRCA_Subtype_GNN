import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd

# Official PAM50 gene list (Parker et al., 2009, JCO)
PAM50 = {
    "ACTR3B", "ANLN", "BAG1", "BCL2", "BIRC5", "BLVRA",
    "CCNB1", "CCNE1", "CDC20", "CDC6", "CDH3", "CENPF",
    "CEP55", "CXXC5", "EGFR", "ERBB2", "ESR1", "EXO1",
    "FGFR4", "FOXA1", "FOXC1", "GPR160", "GRB7", "HMGB3",
    "KIF2C", "KNTC2", "LAMA2", "MAPT", "MDM2", "MELK",
    "MIA", "MKI67", "MLPH", "MMP11", "MYBL2", "MYC",
    "NAT1", "NDC80", "ORC6L", "PHGDH", "PTTG1", "RRM2",
    "SFRP1", "SLC39A6", "TMEM45B", "TYMS", "UBE2C", "UBE2T",
    "VEGFA", "PGR"
}
print(f"PAM50 official count: {len(PAM50)}")

# Our gene set
expr_df = pd.read_csv('data/tcga_expression_160genes.csv', index_col=0)
our_genes = set(expr_df.index)
print(f"Our gene count: {len(our_genes)}")
print()

present  = sorted(PAM50 & our_genes)
missing  = sorted(PAM50 - our_genes)
extra    = sorted(our_genes - PAM50)   # genes we have that are NOT in PAM50

print(f"PAM50 genes PRESENT in our set:  {len(present)} / {len(PAM50)}")
print(f"PAM50 genes MISSING from our set: {len(missing)} / {len(PAM50)}")
print()

print("=== PRESENT PAM50 genes ===")
for g in present:
    print(f"  {g}")

print()
print("=== MISSING PAM50 genes ===")
# Annotate each missing gene with its PAM50 subtype association
annotations = {
    "ACTR3B":  "Basal marker",
    "ANLN":    "Proliferation / Basal",
    "BAG1":    "LumA anti-apoptosis",
    "BCL2":    "LumA anti-apoptosis (ER+)",
    "BIRC5":   "Proliferation / LumB/Basal (Survivin)",
    "BLVRA":   "LumA marker",
    "CCNB1":   "Proliferation / LumB/Basal (Cyclin B1)",
    "CCNE1":   "Proliferation / Her2/LumB (Cyclin E1)",
    "CDC20":   "Proliferation",
    "CDC6":    "Proliferation",
    "CDH3":    "Basal marker (P-cadherin)",
    "CENPF":   "Proliferation",
    "CEP55":   "Proliferation / Basal",
    "CXXC5":   "LumA marker",
    "EGFR":    "Basal marker (EGFR/HER1)",
    "EXO1":    "Proliferation",
    "FGFR4":   "Luminal marker",
    "FOXC1":   "Basal marker (key Basal TF)",
    "GPR160":  "Luminal marker",
    "HMGB3":   "Proliferation",
    "KIF2C":   "Proliferation",
    "KNTC2":   "Proliferation (kinetochore)",
    "LAMA2":   "Normal-like marker",
    "MAPT":    "LumA marker (Tau) — key LumA/LumB separator",
    "MDM2":    "Her2/LumB (p53 regulator)",
    "MELK":    "Proliferation / Basal",
    "MIA":     "LumA/Normal marker",
    "MLPH":    "LumA marker (Melanophilin) — key LumA separator",
    "MMP11":   "Stroma / Her2",
    "MYBL2":   "Proliferation",
    "NAT1":    "LumA marker — key LumA separator",
    "NDC80":   "Proliferation (kinetochore)",
    "ORC6L":   "Proliferation",
    "PHGDH":   "Basal/serine synthesis",
    "PTTG1":   "Proliferation / LumB",
    "RRM2":    "Proliferation",
    "SFRP1":   "LumA/Normal anti-Wnt",
    "SLC39A6": "LumA marker (Zinc transporter)",
    "TMEM45B": "Her2/LumB marker",
    "TYMS":    "Proliferation",
    "UBE2C":   "Proliferation / LumB/Basal",
    "UBE2T":   "Proliferation / Basal",
    "VEGFA":   "Angiogenesis / Her2",
    "PGR":     "LumA marker (Progesterone Receptor) — key LumA/LumB separator",
    "ESR1":    "LumA/LumB master (Estrogen Receptor alpha)",
    "ERBB2":   "Her2 master marker",
    "MKI67":   "Proliferation index (Ki67)",
    "GRB7":    "Her2 co-amplified",
    "FOXA1":   "Luminal pioneer TF",
    "MYC":     "Proliferation oncogene",
    "CCNB1":   "Proliferation",
}
for g in missing:
    ann = annotations.get(g, "")
    print(f"  {g:<12}  {ann}")

print()
print("=== CRITICAL MISSING (LumA/LumB separators) ===")
critical = ["MAPT", "MLPH", "NAT1", "BAG1", "BCL2", "SFRP1", "SLC39A6",
            "CXXC5", "BLVRA", "MIA", "FGFR4", "GPR160"]
for g in critical:
    status = "PRESENT" if g in our_genes else "MISSING"
    print(f"  {g:<12}  {status:<8}  {annotations.get(g,'')}")

print()
print("=== CRITICAL MISSING (Basal separators) ===")
basal_markers = ["FOXC1", "CDH3", "MELK", "PHGDH", "CEP55", "ANLN",
                 "UBE2T", "UBE2C", "ACTR3B"]
for g in basal_markers:
    status = "PRESENT" if g in our_genes else "MISSING"
    print(f"  {g:<12}  {status:<8}  {annotations.get(g,'')}")

print()
print("=== PAM50 PROLIFERATION CLUSTER (LumA vs LumB key) ===")
prolif = ["BIRC5", "CCNB1", "CCNE1", "CDC20", "CDC6", "CENPF", "CEP55",
          "EXO1", "HMGB3", "KIF2C", "KNTC2", "MELK", "MYBL2", "NDC80",
          "ORC6L", "PTTG1", "RRM2", "TYMS", "UBE2C", "UBE2T"]
for g in prolif:
    status = "PRESENT ✓" if g in our_genes else "MISSING ✗"
    print(f"  {g:<12}  {status}")

print()
print("=== SUMMARY ===")
print(f"PAM50 coverage: {len(present)}/{len(PAM50)} = {100*len(present)/len(PAM50):.1f}%")
crit_missing = [g for g in missing if g in {"MAPT","MLPH","NAT1","FOXC1","CDH3",
                                              "BCL2","BAG1","SFRP1","BIRC5","CCNB1",
                                              "MELK","UBE2C","UBE2T","ACTR3B","PHGDH"}]
print(f"Critical missing genes: {len(crit_missing)} -> {crit_missing}")
