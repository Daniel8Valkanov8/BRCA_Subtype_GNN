import requests

def test_string():
    print("🔗 Свързване със STRING-DB API...")
    
    base_url = "https://string-db.org/api/json/network"
    
    # Параметри за заявката
    params = {
        "identifiers": "TP53", # Търсим за TP53
        "species": 9606,       # Код за Homo Sapiens
        "add_nodes": 5,        # Искаме да добави 5 преки партньора
        "network_flavor": "confidence"
    }

    try:
        response = requests.post(base_url, data=params)
        if response.status_code == 200:
            data = response.json()
            print("✅ Връзката е успешна!")
            print(f"🧬 Намерени взаимодействия за TP53: {len(data)}")
            print("\nПърви 5 открити връзки:")
            for edge in data[:5]:
                print(f"- {edge['preferredName_A']} <--> {edge['preferredName_B']} (score: {edge['score']})")
        else:
            print(f"❌ Грешка при заявката: {response.status_code}")
    except Exception as e:
        print(f"💥 Възникна грешка: {e}")

if __name__ == "__main__":
    test_string()