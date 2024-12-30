import chromadb
import os
import utils
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

def getPlayersDict(path: str) -> set:
    players_list = {}
    #for file in os.listdir(path):
    for file in os.listdir(path):
        dict_file = utils.readJson(f'{path}/{file}')
        for row in dict_file.values():
            players_list = players_list | row['players']
            #players_list = players_list + players
    #players_list = [float(player) if '.' in player else int(player) for player in players_list]
    return players_list

if __name__ == '__main__':

    #vector store che contiene documenti ed embeddings di tutti i documenti (i.e. player-match)
    client = chromadb.PersistentClient(path="VectorDB")
    collection = client.get_collection(name="player_embeddings" )

    avg_client = chromadb.PersistentClient(path="AVG_PLAYER_VECTORDB")
    avg_collection = client.get_collection(name="average_player_embeddings")

    p = 349207.0 #Rafa Leao
    #p = 303115.0 #Theo
    #p = 300713.0 #Mbappe
    #p = 11119.0 #Messi
    #players_path = 'Dataset/Events2Text/TeamPlayerList'
    #players_dict = getPlayersDict(players_path)
    #player_name = players_dict[str(p)]
    player_name = 'Rafael Leao'

    results = avg_collection.get(where={'playerId': p}, include=['embeddings'])['embeddings']

    k = 10
    top_k_docs = collection.query(query_embeddings= results, n_results=k)
    doc = top_k_docs["documents"]

    '''
    top_k_players = avg_collection.query(query_embeddings = results, n_results=k)
    for kp in top_k_players['metadatas'][0]:
        #print(kp, kp['playerId'])
        print(players_dict[str(kp['playerId'])])
    '''

    prompt = f"""[INST]
    Sei un osservatore in ambito calcistico. Ho bisogno che mi crei un report per {player_name} evidenziando caratteristiche tecniche e tattiche, punti di forza e debolezze.
    Ecco una lista di documenti che descrivono le azioni fatte durante le partite: 
        {doc}

    Restituisci il report nel seguente formato:

    Giocatore : {player_name}
    Caratteristiche:
    Punti di forza:
    Debolezze:
    Zone del campo predilette:
    [/INST]
    """

    model_name='rstless-research/DanteLLM-7B-Instruct-Italian-v0.1'
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", load_in_8bit=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    input_ids = tokenizer(prompt, return_tensors="pt", truncation=True).input_ids.cuda()
    outputs = model.generate(input_ids=input_ids, max_new_tokens=200)
    print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0].split("[/INST]")[1])

