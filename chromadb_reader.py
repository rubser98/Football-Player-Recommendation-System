import chromadb
import numpy as np

if __name__ == '__main__':
    client = chromadb.PersistentClient(path="VectorDB")

    collection_name = "player_embeddings"  
    collection = client.get_collection(name=collection_name)

    results = collection.get(where={
            '$and': [
                {'playerId': 349207.0},
                {'season': '2021-2022'}
            ]}, include=['embeddings'])
    
    print(type(results['embeddings']))
    #print(results)
    print(results.keys())

    print(results['embeddings'].mean(axis=0).shape)