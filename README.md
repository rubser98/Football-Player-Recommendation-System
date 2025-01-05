MSc Thesis - Ruben Seror

# Football Player Recommendation System

## Overview

The goal of this project is to build a recommendation system that leverages large language models (LLMs) to analyze the performances and playing styles of football players and teams, providing recommendations based on these insights. The system processes match data, fine-tunes Sentence-BERT embeddings, and generates player and team descriptions, which can be used for various analyses and recommendations.

## Project Structure

The repository contains the following main scripts:

1. **whoscored_scraper.py**  
   Scrapes match events from WhoScored.com for the top 7 European leagues, the Champions League, and the Europa League. The data spans from the 2019-2020 to the 2023-2024 season.

2. **player2vec.py**  
   Takes match-player pairs as input and fine-tunes Sentence-BERT by adding custom tokens to the tokenizer, optimizing it for football-related data.

3. **chromadb_ingestion.py**  
   Stores embeddings of match-player pairs in a ChromaDB vector store for efficient retrieval.

4. **chromadb_reader.py**  
   Saves averaged embeddings for players and teams in a separate ChromaDB vector store, creating a high-level representation of players and teams based on their match performances.

5. **player_description.py**  
   Generates descriptions for players and teams by identifying actions and areas of the pitch most closely related to their average embeddings.






Dataset contains:
- Italian commentary, scraped on diretta.it about last 5 seasons of: Champions League, Eredivisie, Europa League, LaLiga, Ligue1, Serie A and Premier League;
- Last 5 seasons of transfer moves about teams of latter championships, scraped on Transfermarkt
- Last 5 seasons of events occured during the match for each match (WhoScored).

player2vec.py -> legge un json con ni file per ogni player dove ni è il numero di partite giocate dal player. il codice si occupa del fine-tuning di sentence bert

chromadb_ingestion.py -> legge il dataset e usando l'embedder fine-tunato salva in un vector store (chroma db) gli embeddings per ogni giocatore-partita

chromadb_reader.py -> salva embeddings medi su un secondo chromadb store


VectorStores:
    - VectorDB -> collection name: player_embeddings, average_player_embeddings_version2, team_embeddings, average_team_embeddings_version2