import json

def writeJson(dati, percorso_file):
    try:
        with open(percorso_file, 'w', encoding='utf-8') as file:
            json.dump(dati, file, ensure_ascii=False, indent=4)

    except Exception as e:
        print(f"Errore durante la scrittura su file: {e}")

def readJson(percorso_file):
    """
    Legge un file JSON e restituisce i dati come dizionario o lista.

    :param percorso_file: Il percorso del file JSON.
    :return: I dati contenuti nel file JSON.
    """
    try:
        with open(percorso_file, 'r', encoding='utf-8') as file:
            dati = json.load(file)
        return dati
    except FileNotFoundError:
        print("Il file non è stato trovato.")
    except json.JSONDecodeError:
        print("Errore nel decodificare il file JSON.")
