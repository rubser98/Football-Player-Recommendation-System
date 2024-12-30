import chromadb
import numpy as np
import utils
import os 
from chromadb_ingestion import PlayerEmbeddingFunction
from tqdm import tqdm

def getPlayersList(path: str) -> set:
    players_list = []
    #for file in os.listdir(path):
    for file in os.listdir(path):
        dict_file = utils.readJson(f'{path}/{file}')
        for row in dict_file.values():
            players = list(row['players'].keys())
            players_list = players_list + players
    players_list = [float(player) if '.' in player else int(player) for player in players_list]
    return set(players_list) - {0.0}

if __name__ == '__main__':
    client = chromadb.PersistentClient(path="VectorDB")

    collection_name = "player_embeddings"  
    collection = client.get_collection(name=collection_name)

    players_path = 'Dataset/Events2Text/TeamPlayerList'
    players = getPlayersList(players_path)

    avg_client = chromadb.PersistentClient(path="AVG_PLAYER_VECTORDB")
    embedding_function = PlayerEmbeddingFunction(model_path="Model_v2/Model")

    # Crea una collection usando la funzione di embedding personalizzata
    avg_collection = client.get_or_create_collection(
        name="average_player_embeddings",
        embedding_function=embedding_function
    )
    with tqdm(total=len(players), desc="Processing players") as pbar:

        for p in players:
                pbar.update(1)
                results = collection.get(where={'playerId': p}, include=['embeddings'])
                avg_player_emb = results['embeddings'].mean(axis=0)
                avg_collection.add(
                    ids = [str(p)],
                    embeddings=[avg_player_emb],
                    metadatas={'playerId': p}
                )
