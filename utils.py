import json

def writeJson(dati, percorso_file):
    try:
        with open(percorso_file, 'w', encoding='utf-8') as file:
            json.dump(dati, file, ensure_ascii=False, indent=4)
        #print(f"Dati scritti correttamente su {percorso_file}")
    except Exception as e:
        print(f"Errore durante la scrittura su file: {e}")