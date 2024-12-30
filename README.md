MSc Thesis - Ruben Seror


Dataset contains:
- Italian commentary, scraped on diretta.it about last 5 seasons of: Champions League, Eredivisie, Europa League, LaLiga, Ligue1, Serie A and Premier League;
- Last 5 seasons of transfer moves about teams of latter championships, scraped on Transfermarkt
- Last 5 seasons of events occured during the match for each match (WhoScored).

player2vec.py -> legge un json con ni file per ogni player dove ni è il numero di partite giocate dal player. il codice si occupa del fine-tuning di sentence bert

chromadb_ingestion.py -> legge il dataset e usando l'embedder fine-tunato salva in un vector store (chroma db) gli embeddings per ogni giocatore-partita

chromadb_reader.py -> salva embeddings medi su un secondo chromadb store

