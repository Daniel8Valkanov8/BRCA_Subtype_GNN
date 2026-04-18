import requests
import pandas as pd

BASE_URL = "https://www.cbioportal.org/api"
STUDY_ID = "brca_tcga_pan_can_atlas_2018"

def fetch_actual_patients():
    # Ендпоинт за изтегляне на клиничните данни
    endpoint = f"{BASE_URL}/studies/{STUDY_ID}/clinical-data"
    print(f"🚀 Извличане на данни за пациентите от {STUDY_ID}...")
    
    try:
        # Правим заявка за всички данни
        response = requests.get(endpoint, timeout=60)
        response.raise_for_status()
        data = response.json()
        
        # Превръщаме "дългия" формат на API-то в удобна таблица
        # API-то връща: {patientId: X, clinicalAttributeId: Y, value: Z}
        raw_df = pd.DataFrame(data)
        
        # Пивотираме таблицата, за да стане като в Excel:
        # Редове = Пациенти, Колони = Атрибути
        df = raw_df.pivot(index='patientId', columns='clinicalAttributeId', values='value')
        
        print(f"✅ Успешно извлечени данни!")
        print(f"📊 Реална бройка на пациентите (редове): {len(df)}")
        print(f"🧬 Бройка на характеристиките (колони): {len(df.columns)}")
        
        if 'SUBTYPE' in df.columns:
            # Преброяваме колко имат попълнен подтип
            with_subtype = df['SUBTYPE'].dropna()
            print(f"🎯 Пациенти с наличен SUBTYPE: {len(with_subtype)}")
            
            print("\nПримерни данни (Първите 5 пациента с техните подтипове):")
            print(df[['SUBTYPE', 'AGE', 'AJCC_PATHOLOGIC_TUMOR_STAGE']].dropna().head(5))
        else:
            print("⚠️ Странно, SUBTYPE не беше открит в реалните данни.")

    except Exception as e:
        print(f"💥 Грешка: {e}")

if __name__ == "__main__":
    fetch_actual_patients()