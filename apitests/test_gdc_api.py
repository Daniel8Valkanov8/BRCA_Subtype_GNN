import requests
import json

def test_gdc():
    print("🔗 Свързване с GDC API (Cancer Genome Atlas)...")
    
    # URL за търсене на случаи (cases)
    base_url = "https://api.gdc.cancer.gov/cases"
    
    # Филтър: Търсим само рак на гърдата (BRCA) в проекта TCGA
    filters = {
        "op": "and",
        "content": [
            {"op": "in", "content": {"field": "project.project_id", "value": ["TCGA-BRCA"]}}
        ]
    }

    params = {
        "filters": json.dumps(filters),
        "fields": "submitter_id,project.project_id,diagnoses.vital_status,diagnoses.primary_diagnosis",
        "format": "JSON",
        "size": "5"  # Тестваме само с 5 случая
    }

    try:
        response = requests.get(base_url, params=params)
        if response.status_code == 200:
            data = response.json()
            print("✅ Връзката е успешна!")
            print(f"📊 Намерени общо случаи в GDC: {data['data']['pagination']['total']}")
            print("\nПримерен резултат (първите 5 случая):")
            for case in data['data']['hits']:
                print(f"- ID: {case['submitter_id']} | Проект: {case['project']['project_id']}")
        else:
            print(f"❌ Грешка при заявката: {response.status_code}")
    except Exception as e:
        print(f"💥 Възникна грешка: {e}")

if __name__ == "__main__":
    test_gdc()