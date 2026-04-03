import pandas as pd

# 1. Пътища към файловете
input_file = 'data/mRNA expression z-scores relative to all samples (log RNA Seq V2 RSEM).txt'
output_file = 'data/mRNA_expression_fixed.txt'

print("⏳ Започвам поправката на файла...")

# 2. Четем файла (използваме sep='\t', тъй като е текстови файл с табулации)
df = pd.read_csv(input_file, sep='\t')

# 3. Поправяме имената на колоните
# Тъй като TP53 и AKT1 са слепени в името на колоната, я преименуваме
if 'TP53AKT1' in df.columns:
    # Тук трябва да сме внимателни: ако данните са слепени, 
    # вероятно TP53 и AKT1 споделят една колона, което е грешка в сорса.
    # Най-често обаче това е просто грешно заглавие.
    df = df.rename(columns={'TP53AKT1': 'TP53'})
    print("✅ Слепването 'TP53AKT1' е поправено на 'TP53'.")

# 4. Премахваме излишната колона 'Unnamed: 29' (и всяка друга празна)
cols_to_keep = [c for c in df.columns if 'Unnamed' not in c]
df = df[cols_to_keep]
print(f"✅ Премахнати са излишните празни колони. Остават {len(df.columns)} колони.")

# 5. Запазваме поправения файл
df.to_csv(output_file, sep='\t', index=False)
print(f"🚀 Готово! Поправеният файл е запазен като: {output_file}")