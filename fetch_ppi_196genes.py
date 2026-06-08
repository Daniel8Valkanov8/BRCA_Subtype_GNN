"""
Изтегля PPI мрежа от STRING API за всичките 196 гена (160 оригинални + 36 PAM50).
Запазва резултата като data/string_ppi_196genes.tsv
"""
import requests
import pandas as pd
import time
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

expr     = pd.read_csv('data/tcga_expression_198genes.csv', index_col=0)
genes    = list(expr.index)
print(f"Гени за заявка: {len(genes)}")

SPECIES    = 9606
MIN_SCORE  = 400
CHUNK_SIZE = 50

all_edges = []

for i in range(0, len(genes), CHUNK_SIZE):
    chunk = genes[i:i+CHUNK_SIZE]
    print(f"  Заявка {i//CHUNK_SIZE + 1}/{(len(genes)-1)//CHUNK_SIZE + 1} ({len(chunk)} гена)...")

    params = {
        'identifiers':     '%0d'.join(chunk),
        'species':         SPECIES,
        'required_score':  MIN_SCORE,
        'caller_identity': 'brca_gnn_thesis',
    }
    try:
        r = requests.post(
            'https://string-db.org/api/tsv-no-header/network',
            data=params, timeout=30
        )
        if r.status_code == 200 and r.text.strip():
            for line in r.text.strip().split('\n'):
                parts = line.split('\t')
                if len(parts) >= 6:
                    all_edges.append({
                        'node1':          parts[2],
                        'node2':          parts[3],
                        'combined_score': float(parts[5])
                    })
        else:
            print(f"    Грешка: {r.status_code}")
    except Exception as e:
        print(f"    Exception: {e}")

    time.sleep(1)

if not all_edges:
    print("Не са намерени взаимодействия!")
    exit(1)

ppi_df = pd.DataFrame(all_edges).drop_duplicates()

gene_set = set(genes)
ppi_df   = ppi_df[ppi_df['node1'].isin(gene_set) & ppi_df['node2'].isin(gene_set)]
ppi_df   = ppi_df[ppi_df['node1'] != ppi_df['node2']]

ppi_df.to_csv('data/string_ppi_196genes.tsv', sep='\t', index=False)

covered = set(ppi_df['node1']) | set(ppi_df['node2'])
print(f"\nРезултат:")
print(f"  Ребра:            {len(ppi_df)}")
print(f"  Покрити гени:     {len(covered)} / {len(genes)}")
print(f"  Средно ребра/ген: {len(ppi_df)*2/len(covered):.1f}")
print(f"  Запазено като:    data/string_ppi_196genes.tsv")
