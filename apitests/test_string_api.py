import requests

# Seed гени за 4-те BRCA-релевантни пътища
PATHWAY_SEEDS = {
    "PI3K/AKT":   ["PIK3CA", "PIK3R1", "AKT1", "PTEN", "MTOR", "PDK1"],
    "ER signaling": ["ESR1", "FOXA1", "GATA3", "CDH1", "TFF1", "PGR"],
    "HER2 signaling": ["ERBB2", "ERBB3", "EGFR", "GRB7", "MUC4"],
    "Cell cycle": ["TP53", "CDK4", "CDK6", "RB1", "CDKN2A", "CCND1", "MKI67"],
}

ALL_SEEDS = sorted(set(g for genes in PATHWAY_SEEDS.values() for g in genes))

def test_pathway_expansion():
    print("STRING-DB: Pathway-based gene expansion за BRCA\n")
    print(f"Seed гени ({len(ALL_SEEDS)}): {ALL_SEEDS}\n")

    # Заявка към STRING: разшири seed гените с техните съседи
    params = {
        "identifiers": "%0d".join(ALL_SEEDS),
        "species": 9606,
        "add_nodes": 150,          # добави до 150 нови гена
        "network_flavor": "confidence",
        "required_score": 400,     # минимален score 0.4
    }

    response = requests.post("https://string-db.org/api/json/network", data=params)
    if response.status_code != 200:
        print(f"Грешка: {response.status_code} — {response.text[:200]}")
        return

    edges = response.json()
    print(f"Върнати взаимодействия: {len(edges)}")

    # Събираме всички уникални гени от ребрата
    genes_found = set()
    for e in edges:
        genes_found.add(e["preferredName_A"])
        genes_found.add(e["preferredName_B"])

    genes_sorted = sorted(genes_found)
    print(f"Уникални гени в мрежата: {len(genes_sorted)}")

    # Нови гени (не са в seed листата)
    new_genes = sorted(genes_found - set(ALL_SEEDS))
    print(f"Новооткрити гени (добавени от STRING): {len(new_genes)}")

    # Разбивка по pathway seed
    print("\nПокритие по pathway:")
    for pathway, seeds in PATHWAY_SEEDS.items():
        covered = [s for s in seeds if s in genes_found]
        print(f"  {pathway}: {len(covered)}/{len(seeds)} seed гена намерени — {covered}")

    # Топ 20 гена по брой връзки (най-свързани = най-важни)
    degree = {}
    for e in edges:
        for name in [e["preferredName_A"], e["preferredName_B"]]:
            degree[name] = degree.get(name, 0) + 1
    top20 = sorted(degree.items(), key=lambda x: -x[1])[:20]
    print("\nТоп 20 най-свързани гена (hub гени):")
    for gene, deg in top20:
        tag = " <-- SEED" if gene in ALL_SEEDS else ""
        print(f"  {gene}: {deg} връзки{tag}")

    # Score разпределение
    scores = [e["score"] for e in edges]
    high   = sum(1 for s in scores if s >= 0.7)
    medium = sum(1 for s in scores if 0.4 <= s < 0.7)
    print(f"\nScore разпределение:")
    print(f"  Висок (>=0.7):   {high} взаимодействия")
    print(f"  Среден (0.4-0.7): {medium} взаимодействия")

    print(f"\nПълен списък гени ({len(genes_sorted)}):")
    print(genes_sorted)

if __name__ == "__main__":
    test_pathway_expansion()
