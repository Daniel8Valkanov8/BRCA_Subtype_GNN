import requests
import pandas as pd

BASE_URL = "https://www.cbioportal.org/api"
STUDY_ID = "brca_metabric"

def explore_metabric_raw_data():
    print(f"🕵️‍♂️ Стартиране на дълбоко разузнаване за {STUDY_ID}...")
    
    # Ендпоинт за клиничните данни
    endpoint = f"{BASE_URL}/studies/{STUDY_ID}/clinical-data"
    
    try:
        # Изтегляме данните без филтри, за да видим всичко
        response = requests.get(endpoint, timeout=60)
        response.raise_for_status()
        raw_json = response.json()
        
        print(f"📦 Изтеглени са общо {len(raw_json)} записа.")
        
        # Превръщаме в DataFrame
        df_long = pd.DataFrame(raw_json)
        
        # КРИТИЧНА СТЪПКА: 
        # В cBioPortal данните могат да са или за Patient, или за Sample.
        # Обединяваме ги в една колона 'EntityID', за да не изпуснем нищо.
        df_long['EntityID'] = df_long['patientId'].fillna(df_long['sampleId'])
        
        # Пивотираме таблицата (превръщаме редовете в колони)
        # index = пациентите, columns = заглавията на показателите, values = стойностите
        df_wide = df_long.pivot(index='EntityID', columns='clinicalAttributeId', values='value')
        
        # Вземаме само първите 20 субекта
        top_20 = df_wide.head(20)
        
        print("\n✅ Таблицата е готова! Ето наличните колони (първите 50 ако са много):")
        print(list(df_wide.columns)[:50])
        
        print("\n📄 Данни за първите 20 субекта (всички колони):")
        # Използваме опция за показване на всички колони в конзолата
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        
        # Принтираме таблицата
        print(top_20)
        
        # ЗАПИСВАМЕ ВЪВ ФАЙЛ, за да можеш да го отвориш с Excel и да разгледаш на спокойствие
        top_20.to_csv("metabric_scout_20.csv")
        print(f"\n💾 Първите 20 пациента са записани в 'metabric_scout_20.csv'")
        
        # Търсене на конкретни стойности в цялата таблица
        print("\n🔍 Бърза проверка за ключови думи в целия масив:")
        keywords = ['Basal', 'LumA', 'LumB', 'Her2']
        for kw in keywords:
            found = df_long[df_long['value'].str.contains(kw, na=False, case=False)]
            if not found.empty:
                attr_name = found['clinicalAttributeId'].unique()
                print(f"  - Стойност '{kw}' е открита в колона(и): {attr_name}")

    except Exception as e:
        print(f"💥 Грешка: {e}")

if __name__ == "__main__":
    explore_metabric_raw_data()
    