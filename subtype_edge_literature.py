# -*- coding: utf-8 -*-
"""
ЕТАП 3: Литературен филтър (PubMed co-citation) върху субтип-разграничаващите ребра.

За всяко ребро (ген1, ген2) от строгия shortlist (sig_BOTH & corr_range>0.5):
  - n_cocite : брой PubMed статии, споменаващи и двата гена (Title/Abstract)
  - n_breast : същото, но в контекст 'breast cancer'

Класификация:
  ЦЕЛ 1 (НОВО)        : n_breast == 0   -> двойката НЕ е изследвана заедно в рак на гърдата
  ЦЕЛ 2 (ПОТВЪРДЕНО)  : n_breast >= 3   -> добре документирана връзка
  СИВА ЗОНА           : 1 <= n_breast <= 2

ВАЖНО: това е ПЪРВИЧЕН скрининг. Кратки символи (напр. KIT) дават шум в [tiab];
кандидатите от Цел 1 после се проверяват ръчно/в други бази (Етап 4).
"""
import sys, io, time, json, urllib.parse, urllib.request
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import pandas as pd

EMAIL = "daniel8valkanov8@gmail.com"
TOOL = "brca_gnn_thesis"
EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
SLEEP = 0.4               # < 3 заявки/сек без API ключ
CORR_THRESH = 0.5

df = pd.read_csv('subtype_edge_discovery.csv')
short = df[(df['sig_BOTH']) & (df['corr_range'] > CORR_THRESH)].copy().reset_index(drop=True)
print(f"Shortlist за литературен филтър: {len(short)} ребра")

def pubmed_count(term):
    params = {'db': 'pubmed', 'term': term, 'retmode': 'json',
              'tool': TOOL, 'email': EMAIL}
    url = EUTILS + '?' + urllib.parse.urlencode(params)
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                data = json.loads(r.read().decode())
            return int(data['esearchresult']['count'])
        except Exception as e:
            if attempt == 2:
                print(f"    [err] {term[:40]}... -> {e}")
                return -1
            time.sleep(1.5)

n_cocite, n_breast = [], []
for k, row in short.iterrows():
    g1, g2 = row['gene1'], row['gene2']
    t_all = f'{g1}[tiab] AND {g2}[tiab]'
    t_brc = f'{g1}[tiab] AND {g2}[tiab] AND breast cancer'
    c1 = pubmed_count(t_all); time.sleep(SLEEP)
    c2 = pubmed_count(t_brc); time.sleep(SLEEP)
    n_cocite.append(c1); n_breast.append(c2)
    if (k + 1) % 20 == 0:
        print(f"  {k+1}/{len(short)} обработени")

short['pubmed_cocite'] = n_cocite
short['pubmed_breast'] = n_breast

def classify(nb):
    if nb == 0:
        return 'ЦЕЛ1_НОВО'
    if nb >= 3:
        return 'ЦЕЛ2_ПОТВЪРДЕНО'
    return 'СИВА_ЗОНА'
short['literature_class'] = short['pubmed_breast'].apply(classify)

# подреждане: новите с най-голям ефект първо
short = short.sort_values(['literature_class', 'corr_range'], ascending=[True, False]).reset_index(drop=True)
cols = ['gene1', 'gene2', 'top_att_subtype', 'corr_range', 'string_score',
        'attention_kruskal_p', 'diffcoexp_p', 'pubmed_cocite', 'pubmed_breast', 'literature_class']
rcols = [c for c in short.columns if c.startswith('r_')]
short[cols + rcols].to_csv('subtype_edge_literature.csv', index=False, encoding='utf-8')

# ---- доклад ----
novel = short[short['literature_class'] == 'ЦЕЛ1_НОВО']
conf  = short[short['literature_class'] == 'ЦЕЛ2_ПОТВЪРДЕНО']
gray  = short[short['literature_class'] == 'СИВА_ЗОНА']
L = []
L.append("=" * 80)
L.append("ЕТАП 3 — ЛИТЕРАТУРЕН ФИЛТЪР (PubMed) НА СУБТИП-РАЗГРАНИЧАВАЩИТЕ РЕБРА")
L.append("=" * 80)
L.append(f"Shortlist (corr_range>{CORR_THRESH}): {len(short)} ребра")
L.append(f"  ЦЕЛ 1 (НОВО, breast=0):        {len(novel)}")
L.append(f"  ЦЕЛ 2 (ПОТВЪРДЕНО, breast>=3): {len(conf)}")
L.append(f"  Сива зона (1-2):              {len(gray)}")
L.append("")
L.append("─" * 80)
L.append("ЦЕЛ 1 — НОВИ КАНДИДАТИ (двойка не е изследвана заедно в рак на гърдата)")
L.append("─" * 80)
L.append(f"{'Gene1':<9}{'Gene2':<9}{'субтип':<13}{'corrRng':>8}{'coCite':>7}{'breast':>7}")
for _, r in novel.iterrows():
    L.append(f"{r['gene1']:<9}{r['gene2']:<9}{r['top_att_subtype'].split('_')[1]:<13}"
             f"{r['corr_range']:>8.2f}{int(r['pubmed_cocite']):>7}{int(r['pubmed_breast']):>7}")
L.append("")
L.append("─" * 80)
L.append("ЦЕЛ 2 — ПОТВЪРДЕНИ ОТ ЛИТЕРАТУРАТА (топ 25 по ефект)")
L.append("─" * 80)
L.append(f"{'Gene1':<9}{'Gene2':<9}{'субтип':<13}{'corrRng':>8}{'breast':>7}")
for _, r in conf.head(25).iterrows():
    L.append(f"{r['gene1']:<9}{r['gene2']:<9}{r['top_att_subtype'].split('_')[1]:<13}"
             f"{r['corr_range']:>8.2f}{int(r['pubmed_breast']):>7}")
L.append("")
L.append("СЛЕДВАЩА СТЪПКА (Етап 4): Enrichr/STRING/expression анализ на ЦЕЛ 1 кандидатите.")
report = "\n".join(L)
with open('subtype_edge_literature_report.txt', 'w', encoding='utf-8') as f:
    f.write(report)
print("\n" + report)
print("\n[OK] subtype_edge_literature.csv | subtype_edge_literature_report.txt")
