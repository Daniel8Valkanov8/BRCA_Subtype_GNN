"""
Изтегля експресия за 174 гена от TCGA BRCA и нова PPI мрежа от STRING.
"""
import requests
import pandas as pd

BASE_URL   = 'https://www.cbioportal.org/api'
PROFILE_ID = 'brca_tcga_pan_can_atlas_2018_rna_seq_v2_mrna_median_all_sample_Zscores'
SAMPLE_LIST = 'brca_tcga_pan_can_atlas_2018_rna_seq_v2_mrna'

GENES_174 = [
    'ABL1','AKT1','AKT2','AKT3','ALK','ANXA5','AR','AREG','ARID1A','ARNT',
    'ATM','BCL2','BCL2L1','BCL2L11','BECN1','BRAF','BRCA1','BRCA2','BTC',
    'CASP3','CASP8','CASP9','CBLB','CCNA1','CCNA2','CCNB1','CCND1','CCND2',
    'CCND3','CCNE1','CCNE2','CD274','CD44','CDC25A','CDC37','CDH1','CDK2',
    'CDK4','CDK6','CDKN1A','CDKN1B','CDKN1C','CDKN2A','CDKN2B','CDKN2C',
    'CDKN3','CTNNB1','CYP19A1','E2F1','E2F2','E2F3','EGF','EGFR','EIF4EBP1',
    'EP300','EPCAM','ERBB2','ERBB3','ERBB4','ESR1','ESR2','ETS1','EZH2',
    'FBXW7','FGF19','FGFR1','FGFR2','FKBP4','FOS','FOXA1','FOXM1','FOXO1',
    'FOXO3','GAB1','GATA3','GRB2','GRB7','GREB1','HGF','HIF1A','HRAS',
    'HSP90AA1','IGF1','IGF1R','IGF2','INPP4B','IRS1','JAK2','JUN','KDR',
    'KIT','KRAS','KRT14','KRT19','KRT5','KRT8','MAP2K1','MAPK14','MAPK8',
    'MCL1','MDM2','MDM4','MET','MKI67','MLH1','MTOR','MUC1','MYC','NCOA1',
    'NCOA3','NF1','NF2','NOTCH1','NRAS','NTRK1','PARP1','PBRM1','PDGFRA',
    'PDGFRB','PDK1','PGR','PIK3CA','PIK3CB','PIK3R1','PIK3R2','PIK3R3',
    'PLCG2','PTEN','PTK2','PTK6','PTPN11','RAF1','RASSF1','RB1','RBL2',
    'RET','RICTOR','SHC1','SKP2','SMAD2','SMAD3','SMAD4','SMARCA4','SNAI1',
    'SNAI2','SP1','SRC','STAT1','STAT3','STAT5A','STAT5B','STK11','TERT',
    'TFF1','TGFA','TP53','TSC2','VAV3','YAP1','ZEB1',
]

# --- СТЪПКА 1: Entrez IDs ---
print(f"Стъпка 1: Вземаме Entrez IDs за {len(GENES_174)} гена...")
r = requests.post(f'{BASE_URL}/genes/fetch?geneIdType=HUGO_GENE_SYMBOL',
                  json=GENES_174, timeout=30)
gene_info = r.json()
gene_map  = {g['hugoGeneSymbol']: g['entrezGeneId'] for g in gene_info}
found     = [g for g in GENES_174 if g in gene_map]
missing   = [g for g in GENES_174 if g not in gene_map]
print(f"  Намерени: {len(found)} | Липсващи: {missing}")

# --- СТЪПКА 2: Изтегляме mRNA експресия ---
print(f"\nСтъпка 2: Изтегляме mRNA z-scores от TCGA...")
all_records = []
for i, (symbol, entrez_id) in enumerate(gene_map.items(), 1):
    resp = requests.get(
        f'{BASE_URL}/molecular-profiles/{PROFILE_ID}/molecular-data',
        params={'sampleListId': SAMPLE_LIST, 'entrezGeneId': entrez_id},
        timeout=30
    )
    if resp.status_code == 200:
        for rec in resp.json():
            all_records.append({'gene': symbol, 'sampleId': rec['sampleId'], 'value': rec['value']})
    if i % 20 == 0:
        print(f"  {i}/{len(gene_map)} гена изтеглени...")

print(f"  Общо записи: {len(all_records)}")

df_long = pd.DataFrame(all_records)
df_expr = df_long.pivot_table(index='gene', columns='sampleId', values='value', aggfunc='first')
df_expr = df_expr.dropna(how='all')

print(f"  Матрица: {df_expr.shape} (гени x пациенти)")
df_expr.to_csv('data/tcga_expression_174genes.csv')
print("  Запазено: data/tcga_expression_174genes.csv")

# --- СТЪПКА 3: Нова PPI мрежа от STRING за 174-те гена ---
print(f"\nСтъпка 3: Изтегляме PPI мрежа от STRING за {len(found)} гена...")
params = {
    "identifiers": "%0d".join(found),
    "species":     9606,
    "required_score": 400,
    "network_flavor": "confidence",
}
r2   = requests.post("https://string-db.org/api/tsv/network", data=params, timeout=60)
lines = r2.text.strip().split('\n')
edges = []
for line in lines[1:]:
    parts = line.split('\t')
    if len(parts) >= 3:
        edges.append({'node1': parts[0], 'node2': parts[1], 'combined_score': float(parts[2])})

df_ppi = pd.DataFrame(edges)
print(f"  Взаимодействия: {len(df_ppi)}")
df_ppi.to_csv('data/string_ppi_174genes.tsv', sep='\t', index=False)
print("  Запазено: data/string_ppi_174genes.tsv")

print("\nГотово! Обобщение:")
print(f"  Гени в експресионната матрица: {df_expr.shape[0]}")
print(f"  Пациенти: {df_expr.shape[1]}")
print(f"  PPI ребра (score >= 0.4): {len(df_ppi)}")
