import chromadb
import os
import utils
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from chromadb_ingestion import PlayerEmbeddingFunction
import argparse
from chromadb_reader import getTeamDict
import pandas as pd
from huggingface_hub import login


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

def getTeamDict(path: str) -> set:
    team_list = {}
    #for file in os.listdir(path):
    for file in os.listdir(path):
        nazione = file.split('-')[0]
        season = file.split('_')[1]
        if season not in team_list.keys():
            team_list[season] = []

        dict_file = utils.readJson(f'{path}/{file}')
        for k, v in dict_file.items():
            if nazione != 'Europa':
                print(v)
                break
                team_list = team_list | new_row

    return team_list

def getTeamsPlayerDict(path: str) -> set:
    team_list = {}
    for file in os.listdir(path):
        nazione = file.split('-')[0]
        season = file.split('_')[1]
        if season not in team_list.keys():
            team_list[season] = {}

        dict_file = utils.readJson(f'{path}/{file}')

        for k, v in dict_file.items():
            if nazione != 'Europa':
                team_list[season][k] = v

    return team_list


def prompt_team_description(team : dict, season:str, players_dict : dict, player_collection : chromadb.Collection, vocab_collection : chromadb.Collection, df_presenze: pd.DataFrame, k: int = 10) -> str:
    
    prompt_team = """Sei un football match analyst esperto. 
    Ho bisogno che mi crei un report per una squadra di calcio basandoti sulle azioni compiute dai giocatori che la compongono. 
    Utilizza queste informazioni per analizzare le caratteristiche tecniche e tattiche collettive della squadra, evidenziando lo stile di gioco, i punti di forza, le debolezze e le zone del campo maggiormente sfruttate.
    Per ogni giocatore è indicato il numero di presenze. Più è alto il numero delle presenze e maggiore sarà il suo contributo nello stile di gioco della squadra.
    Nel report non specificare nomi dei giocatori.
    La lista delle azioni per ciascun giocatore è fornita qui sotto:
    """

    for p in team.keys():

        player_name = players_dict[str(p)]
        try:
            results = player_collection.get(where={'$and':[{'id': p}, {'season': season}]}, include=['embeddings'])['embeddings']
            vocab_results = vocab_collection.query(query_embeddings=results, n_results = k, include=['documents'])
            actions = vocab_results['documents'][0]
            presenze = getAppearances(df_presenze, float(p), season)
            f_string = f'- {player_name}: presenze: {presenze}, caratteristiche: {actions}\n'
            prompt_team += f_string
        except:
            print(player_name)

    return prompt_team

def prompt_player_description(p: str, players_dict: dict, player_collection: chromadb.Collection, vocab_collection: chromadb.Collection, k: int = 20):

    player_name = players_dict[p]
    results = player_collection.get(where={'id': str(p)}, include=['embeddings'])['embeddings']
    vocab_results = vocab_collection.query(query_embeddings=results, n_results = k, include=['documents'])
    prompt = f"""
    Sei un osservatore in ambito calcistico. Ho bisogno che mi crei un report per {player_name} evidenziando caratteristiche tecniche e tattiche, punti di forza e debolezze.
    Ecco una lista di azioni fatte durante le partite che meglio descrivono il giocatore: 
        {vocab_results['documents']}

    Restituisci il report nel seguente formato:

    Caratteristiche:
    Punti di forza:
    Debolezze:
    Zone del campo predilette:
    """
    return prompt

def getAppearances(df_dataset: pd.DataFrame, p: str, season: str) -> int:
    df_presence = df_dataset[(df_dataset['playerId'] == p) & (df_dataset['season'] == season)]['row_count']
    return int(df_presence.loc[df_presence.index[0]])

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Generate player and team descriptions")
    parser.add_argument("--vector_store_dir", type=str, required=True, help="Path to the directory containing the dataset file.")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Path to the directory containing the dataset file.")
    args = parser.parse_args()
    #vector store che contiene documenti ed embeddings di tutti i documenti (i.e. player-match)
    client = chromadb.PersistentClient(path=args.vector_store_dir)

    vocab_collection = client.get_collection(name="vocab")
    #"average_player_embeddings_version2"
    player_collection = client.get_collection(name="average_player_embeddings_version2")
    player_season_collection = client.get_collection(name="average_player_embeddings_season")

    #records = player_collection.get(limit=1)
    #print(records['metadatas'])
    p = 349207.0 #Rafa Leao
    #p = 303115.0 #Theo
    #p = 300713.0 #Mbappe
    #p = 11119.0 #Messi

    #dir = 'Dataset/Events2Text'
    dir = 'Dataset'
    df_dataset = pd.read_json(f'{dir}/player2vec_dataset.json')
    player_season_counts = df_dataset.groupby(['playerId', 'playerName', 'season']).size().reset_index(name='row_count')

    #model = PlayerEmbeddingFunction('Model_v2/Model')
    vocab_collection = client.get_collection(name="vocab")
    #players_path = 'Dataset/Events2Text/TeamPlayerList'
    players_dict = getPlayersDict(args.dataset_dir)

    prompt = prompt_player_description(str(p), players_dict, player_collection, vocab_collection)
    #print(prompt)

    isTeam = False

    if isTeam:
        team_dict = getTeamsPlayerDict(args.dataset_dir)
        team = team_dict['2021-2022']['80']['players']
        team_prompt = prompt_team_description(team, '2021-2022', players_dict, player_season_collection, vocab_collection, player_season_counts)
    #print(team_prompt)

    '''

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
    
    #login()
    #model_name = 'meta-llama/Meta-Llama-3.1-8B-Instruct'
    model_name = 'meta-llama/Llama-3.2-3B'
    #model_name = "galatolo/cerbero-7b"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto", 
        #load_in_8bit=True, 
        #llm_int8_enable_fp32_cpu_offload=True,
        offload_folder='offload_weights')
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    input_ids = tokenizer(prompt, return_tensors="pt", truncation=True).input_ids.cuda()
    attention_mask = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).attention_mask.cuda()

    with torch.no_grad():
        outputs = model.generate(input_ids=input_ids,attention_mask=attention_mask, max_new_tokens=2000, pad_token_id=tokenizer.eos_token_id)
    
    print(len(outputs))
    
    #print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0].split("[/INST]")[1])
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    with open('prova_report_leao.txt', 'w') as f:
        f.write(generated_text)

