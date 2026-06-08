"""
Изтегля 38-те липсващи PAM50 гени от cBioPortal API
и ги добавя към tcga_expression_174genes.csv
"""
import requests
import pandas as pd
import time
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

MISSING_PAM50 = [
    'ACTR3B','ANLN','BAG1','BIRC5','BLVRA','CDC20','CDC6','CDH3',
    'CENPF','CEP55','CXXC5','EXO1','FGFR4','FOXC1','GPR160','HMGB3',
    'KIF2C','KNTC2','LAMA2','MAPT','MELK','MIA','MLPH','MMP11',
    'MYBL2','NAT1','NDC80','ORC6L','PHGDH','PTTG1','RRM2','SFRP1',
    'SLC39A6','TMEM45B','TYMS','UBE2C','UBE2T','VEGFA'
]

STUDY_ID         = 'brca_tcga_pan_can_atlas_2018'
BASE_URL         = 'https://www.cbioportal.org/api'

# ── Стъпка 1: намери правилния molecular profile за mRNA ───────────────────
print("Стъпка 1: Намиране на molecular profiles...")
r = requests.get(f'{BASE_URL}/molecular-profiles',
                 params={'studyId': STUDY_ID}, timeout=30)
profiles = r.json()
mrna_profiles = [p for p in profiles
                 if 'mrna' in p['molecularProfileId'].lower()
                 or 'rna' in p['molecularProfileId'].lower()]
print(f"  Намерени mRNA профили: {len(mrna_profiles)}")
for p in mrna_profiles:
    print(f"    {p['molecularProfileId']}  —  {p['name']}")

# Избираме z-score профила (съответства на данните в нашия CSV)
profile_id = None
for p in mrna_profiles:
    pid = p['molecularProfileId']
    if 'zscore' in pid.lower() or 'z_score' in pid.lower() or 'median' in pid.lower():
        profile_id = pid
        break
if not profile_id and mrna_profiles:
    profile_id = mrna_profiles[0]['molecularProfileId']

print(f"\n  Избран профил: {profile_id}")

# ── Стъпка 2: вземи sample IDs ─────────────────────────────────────────────
print("\nСтъпка 2: Зареждане на sample IDs от нашия CSV...")
existing = pd.read_csv('data/tcga_expression_174genes.csv', index_col=0)
sample_ids = list(existing.columns)
print(f"  Пациенти: {len(sample_ids)}")

# ── Стъпка 3: изтегли гените на групи ──────────────────────────────────────
print(f"\nСтъпка 3: Изтегляне на {len(MISSING_PAM50)} гена от cBioPortal...")
print(f"  Профил: {profile_id}")

all_data = {}
CHUNK = 10

for i in range(0, len(MISSING_PAM50), CHUNK):
    chunk_genes = MISSING_PAM50[i:i+CHUNK]
    print(f"  Обработка гени {i+1}–{min(i+CHUNK, len(MISSING_PAM50))}: {chunk_genes}")

    payload = {
        "entrezGeneIds":      [],
        "hugoGeneSymbols":    chunk_genes,
        "sampleIds":          sample_ids,
        "studyId":            STUDY_ID
    }
    try:
        r = requests.post(
            f'{BASE_URL}/molecular-profiles/{profile_id}/molecular-data/fetch',
            json=payload, timeout=60
        )
        if r.status_code == 200:
            data = r.json()
            for entry in data:
                gene   = entry['gene']['hugoGeneSymbol']
                sample = entry['sampleId']
                value  = entry['value']
                if gene not in all_data:
                    all_data[gene] = {}
                all_data[gene][sample] = value
            fetched = list({e['gene']['hugoGeneSymbol'] for e in data})
            print(f"    OK — получени {len(fetched)} гена: {fetched}")
        else:
            print(f"    ГРЕШКА: status={r.status_code}, {r.text[:200]}")
    except Exception as e:
        print(f"    Exception: {e}")

    time.sleep(1)

# ── Стъпка 4: Обединяване с existing data ──────────────────────────────────
print(f"\nСтъпка 4: Обединяване...")
if all_data:
    new_df = pd.DataFrame(all_data).T  # genes x samples
    new_df = new_df.reindex(columns=sample_ids)  # align samples
    combined = pd.concat([existing, new_df])
    combined = combined[~combined.index.duplicated(keep='first')]
    print(f"  Стари гени:    {len(existing)}")
    print(f"  Нови PAM50:    {len(new_df)}")
    print(f"  Общо след merge: {len(combined)}")
    combined.to_csv('data/tcga_expression_198genes.csv')
    print(f"  Запазено: data/tcga_expression_198genes.csv")

    actually_added = [g for g in MISSING_PAM50 if g in new_df.index]
    still_missing  = [g for g in MISSING_PAM50 if g not in new_df.index]
    print(f"\n  Успешно добавени ({len(actually_added)}): {actually_added}")
    print(f"  Все още липсват ({len(still_missing)}): {still_missing}")
else:
    print("  Няма получени данни — провери интернет връзката и profile_id!")
