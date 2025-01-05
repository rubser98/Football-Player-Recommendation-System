import chromadb
import os
import utils
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from chromadb_reader import getTeamDict
from chromadb_ingestion import PlayerEmbeddingFunction

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


def getTotalVocab():
    dict_vocab = utils.readJson('action_translations.json')
    vocab = []
    for d in dict_vocab.values():
        vocab = vocab + list(d.values())
    vocab = set(vocab) - {''}
    x_text = {1: 'difesa', 2: 'trequarti difensiva', 3: 'centrocampo', 4: 'trequarti offensiva', 5: 'attacco'}
    y_text = {1: 'fascia destra', 2: 'centro destra', 3: 'centrale', 4: 'centro sinistra', 5: 'fascia sinistra'}
    field_vocab = [f'{x}, {y}' for x in x_text.values() for y in y_text.values()]
    total_vocab = [f'{x} {y}' for x in vocab for y in field_vocab]
    return total_vocab


if __name__ == '__main__':

    #vector store che contiene documenti ed embeddings di tutti i documenti (i.e. player-match)
    client = chromadb.PersistentClient(path="VectorDB")
    collection = client.get_collection(name="player_embeddings")
    #"average_player_embeddings_version2"
    avg_collection = client.get_collection(name="average_team_embeddings_version2")

    #p = 349207.0 #Rafa Leao
    #p = 303115.0 #Theo
    #p = 300713.0 #Mbappe
    p = 11119.0 #Messi
    p=31772.0

 
    players_path = 'Dataset/Events2Text/TeamPlayerList'
    players_dict = getPlayersDict(players_path)
    player_name = players_dict[str(p)]


    p='75'
    players_dict = getTeamDict(players_path)
    player_name =  players_dict[str(p)]
    print(player_name)
    results = avg_collection.get(include=["metadatas"], limit=1)
    print(results["metadatas"])
    

    #results = avg_collection.get(where={'playerId': str(p)}, include=['embeddings'])['embeddings']
    results = avg_collection.get(where={'$and': [{'teamId': p}, {'season': '2022-2023'}]}, include=['embeddings'])['embeddings']
    
    print(results.shape)
    k = 10
    top_k_docs = collection.query(query_embeddings= results, n_results=k)
    doc = top_k_docs["documents"]
    top_k_players = avg_collection.query(query_embeddings = results, n_results=k+1)
    
    '''
    i=0
    print(f'Giocatori simili a {player_name}:')
    for kp in top_k_players['metadatas'][0]:
        #print(kp, kp['playerId'])
        if i > 0:
            print(f"{i}. {players_dict[str(kp['teamId'])]}")
        i+=1
    '''
    j=0
    model = PlayerEmbeddingFunction('Model_v2/Model')
    vocab_collection = client.get_collection(
            name="vocab",
            embedding_function=model
        )  
    if j > 0:
        total_vocab = getTotalVocab()
        for i in range(len(total_vocab)):
            vocab_collection.add(ids=[str(i)], documents=[total_vocab[i]])

    
    vocab_results = vocab_collection.query(query_embeddings=results, n_results = 20, include=['documents'])
    #print(vocab_results['documents'])


    prompt = f"""
    Sei un osservatore in ambito calcistico. Ho bisogno che mi crei un report per {player_name} evidenziando caratteristiche tecniche e tattiche, punti di forza e debolezze.
    Ecco una lista di documenti che descrivono le azioni fatte durante le partite: 
        {vocab_results['documents']}

    Restituisci il report nel seguente formato:

    Caratteristiche:
    Punti di forza:
    Debolezze:
    Zone del campo predilette:
    """

    prompt_team = f"""
    Sei un football match analyst. Ho bisogno che mi crei un report per {player_name} evidenziando caratteristiche tecniche e tattiche, punti di forza e debolezze.
    Ecco una lista di documenti che descrivono le azioni fatte dalla squadra durante le partite: 
        {vocab_results['documents']}

    Restituisci il report nel seguente formato:

    Caratteristiche:
    Punti di forza:
    Debolezze:
    Zone del campo predilette:
    """
    print(prompt_team)

    prompt = f"""
    Utilizzando i seguenti dati sulle azioni del giocatore, genera una descrizione schematica e dettagliata del suo stile di gioco. Organizza la risposta in tre sezioni principali:

    1. **Punti di forza**: descrivi le aree in cui il giocatore eccelle, basandoti sui dati forniti.
    2. **Debolezze**: individua i limiti o le difficoltà del giocatore, considerando le informazioni disponibili.
    3. **Caratteristiche principali**: riassumi lo stile di gioco generale del giocatore, indicando ruolo, posizione preferita e peculiarità tecnico-tattiche.

    ### Input dati del giocatore:

    Nome giocatore: {player_name}
    Top 5 documenti più simili all'embedding medio:
  
    {doc}

    ### Formato di output richiesto:

    #### Nome giocatore: {player_name}

    1. **Punti di forza:**
    - [Descrizione sintetica del primo punto di forza, es. precisione nei passaggi]
    - [Descrizione sintetica del secondo punto di forza]

    2. **Debolezze:**
    - [Descrizione sintetica della prima debolezza, es. difficoltà nei contrasti]
    - [Descrizione sintetica della seconda debolezza]

    3. **Caratteristiche principali:**
    - Ruolo e posizione preferita: [Es. "Centrocampista difensivo"]
    - Stile di gioco: [Es. "Giocatore creativo, con visione di gioco e abilità nei passaggi lunghi"]
    - Peculiarità tecniche: [Es. "Ambidestro, ottimo nei tiri da fuori area"]
    """
    '''
    #model_name='rstless-research/DanteLLM-7B-Instruct-Italian-v0.1'
    #model_name = 'meta-llama/Meta-Llama-3.1-8B-Instruct'
    model_name = "galatolo/cerbero-7b"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto", 
        load_in_8bit=True, 
        llm_int8_enable_fp32_cpu_offload=True,
        offload_folder='offload_weights')
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    input_ids = tokenizer(prompt, return_tensors="pt", truncation=True).input_ids.cuda()
    with torch.no_grad():
        outputs = model.generate(input_ids=input_ids, max_new_tokens=200)
    
    
    #print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0].split("[/INST]")[1])
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(generated_text)
    '''
