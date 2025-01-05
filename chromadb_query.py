import chromadb
from player_description import getPlayersDict
from chromadb_ingestion import PlayerEmbeddingFunction

#db_folder = 'AVG_PLAYER_VECTORDB'
db_folder = 'VectorDB'
client = chromadb.PersistentClient(path=db_folder)

collection_name = "average_player_embeddings_version2"  
collection = client.get_collection(name=collection_name)

players_dict = getPlayersDict('Dataset/Events2Text/TeamPlayerList')

model_dir = 'Model_v2/Model'
embedding_function = PlayerEmbeddingFunction(model_path=model_dir)


#query_text = 'Cross di destro'
query_text = 'Recupero palla trequarti offensiva'
#query_text='portiere che effettua interventi fuori area'
query = embedding_function(query_text)
k=10
results = collection.query(query_embeddings=query, include=['metadatas'], n_results=k)

print(query_text+':')
for kp in results['metadatas'][0]:
    print(players_dict[str(kp['playerId'])])



