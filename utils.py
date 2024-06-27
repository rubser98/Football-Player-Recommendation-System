import json

def writeJson(dati, percorso_file):
    """
    Scrive un dizionario o una lista di dizionari su un file JSON.

    :param dati: Il dizionario o la lista di dizionari da scrivere su file.
    :param percorso_file: Il percorso del file JSON.
    """
    try:
        with open(percorso_file, 'w', encoding='utf-8') as file:
            json.dump(dati, file, ensure_ascii=False, indent=4)
        #print(f"Dati scritti correttamente su {percorso_file}")
    except Exception as e:
        print(f"Errore durante la scrittura su file: {e}")