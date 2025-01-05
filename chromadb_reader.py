import chromadb
import numpy as np
import utils
import os 
from chromadb_ingestion import PlayerEmbeddingFunction
from tqdm import tqdm
import argparse

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

def getPlayersDict(path: str) -> set:
    players_list = {}
    #for file in os.listdir(path):
    for file in os.listdir(path):
        nazione = file.split('-')[0]
        dict_file = utils.readJson(f'{path}/{file}')
        for row in dict_file.values():
            if nazione != 'Europa':
                players_list = players_list | row['players']
            #players_list = players_list + players
    #players_list = [float(player) if '.' in player else int(player) for player in players_list]
    players_list.pop('0.0')

    return players_list

def getTeamDict(path: str) -> set:
    team_list = {}
    #for file in os.listdir(path):
    for file in os.listdir(path):
        nazione = file.split('-')[0]
        dict_file = utils.readJson(f'{path}/{file}')
        for k, v in dict_file.items():
            if nazione != 'Europa':
                new_row = {k: v['name']}
                team_list = team_list | new_row
            #players_list = players_list + players
    #players_list = [float(player) if '.' in player else int(player) for player in players_list]
    return team_list

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Store player embeddings into chroma db")
    parser.add_argument("--vector_store_dir", type=str, required=True, help="Path to the directory containing the dataset file.")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Path to the directory containing the dataset file.")
    parser.add_argument("--model_dir", type=str, required=True, help="Path to the directory where the model is stored.")
    args = parser.parse_args()

    client = chromadb.PersistentClient(path=args.vector_store_dir)

    collection_name = "team_embeddings"  
    collection = client.get_collection(name=collection_name)

    #players_path = 'Dataset/Events2Text/TeamPlayerList'
    #players = getPlayersDict(args.dataset_dir)
    players = getTeamDict(args.dataset_dir)
    seasons = ['2019-2020', '2020-2021', '2021-2022', '2022-2023', '2023-2024']
    #avg_client = chromadb.PersistentClient(path=args.output_dir)
    embedding_function = PlayerEmbeddingFunction(model_path=args.model_dir)

    # Crea una collection usando la funzione di embedding personalizzata
    avg_collection = client.get_or_create_collection(
        name="average_team_embeddings_version2",
        embedding_function=embedding_function
    )

    results = collection.get(include=["metadatas"], limit=1)
    print(results["metadatas"])

    with tqdm(total=len(players.keys()), desc="Processing players") as pbar:

        for p in players.keys():
                pbar.update(1)
                for s in seasons:
                    try:
                        results = collection.get(where={'$and': [{'teamId': int(p)}, {'season': s}]}, include=['embeddings'])
                    
                        avg_player_emb = results['embeddings'].mean(axis=0)
                        avg_collection.upsert(
                            ids = [f'{p}-{s}'],
                            embeddings=[avg_player_emb],
                            metadatas={'teamId': p, 'teamName': players[p], 'season': s}
                        )
                    
                    except:
                        print(players[p],s)

