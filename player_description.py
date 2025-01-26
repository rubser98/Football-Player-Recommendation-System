import chromadb
import os
import utils
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from chromadb_ingestion import PlayerEmbeddingFunction
import argparse 
import pandas as pd
from chromadb_reader import getPlayersDict, getTeamDict
from tqdm import tqdm

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


def prompt_team_description(team : dict, season:str, players_dict : dict, player_collection : chromadb.Collection, vocab_collection : chromadb.Collection, df_presenze: pd.DataFrame, k: int = 10, lang: str = 'en') -> str:
    
    if lang == 'it':
        prompt_team = """Sei un football match analyst esperto. 
        Ho bisogno che mi crei un report per una squadra di calcio basandoti sulle azioni compiute dai giocatori che la compongono. 
        Utilizza queste informazioni per analizzare le caratteristiche tecniche e tattiche collettive della squadra, evidenziando lo stile di gioco, i punti di forza, le debolezze e le zone del campo maggiormente sfruttate.
        Per ogni giocatore è indicato il numero di presenze. Più è alto il numero delle presenze e maggiore sarà il suo contributo nello stile di gioco della squadra.
        Nel report non specificare nomi dei giocatori.
        La lista delle azioni per ciascun giocatore è fornita qui sotto:
        """
    elif lang == 'en':
        prompt_team = """
        """
    else:
        raise KeyError("Linguaggio non supportato")

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

def prompt_player_description(p: str, players_dict: dict, player_collection: chromadb.Collection, vocab_collection: chromadb.Collection, k: int = 20, lang: str = 'en'):

    player_name = players_dict[p]
    id = 'id' if lang == 'en' else 'playerId'
    results = player_collection.get(where={id: str(p)}, include=['embeddings'])['embeddings']
    vocab_results = vocab_collection.query(query_embeddings=results, n_results = k, include=['documents'])
    if lang == 'it':
        prompt = f"""
        Sei un osservatore calcistico professionista con esperienza nell'analisi delle caratteristiche tecniche e tattiche dei giocatori.
        Devi generare un rapporto dettagliato su un giocatore, basandoti sull'elenco delle azioni fornite che descrivono il suo stile di gioco durante le partite.
        Il tuo compito è analizzare questi dati e fornire un rapporto strutturato come segue:
        ###Input Data
            - Giocatore: {player_name}
            - Azioni: {vocab_results['documents'][0]}

        Restituisci il report nel seguente formato:

        ###Formato output
            *Giocatore*:
            {player_name}
            *Caratteristiche*:
            Evidenzia le caratteristiche tecnico tattiche che meglio rappresentano il giocatore
            *Punti di forza*:
            Evidenzia i punti di forza principali del giocatore emersi dal suo stile di gioco.
            *Debolezze*:
            Indica le aree in cui il giocatore deve migliorare.
            *Zone del campo predilette*:
            Identifica le zone del campo in cui il giocatore è più attivo o performa meglio.

        ###Note per l'analisi
            - Usa un linguaggio conciso e professionale.
            - Il report deve essere realistico e utile per scopi di osservazione calcistica.
            - Non generare codice o strutture di classe. Concentrati solo sull'analisi calcistica.
            - L'output deve essere in testo semplice, chiaramente formattato secondo la struttura sopra indicata.
            - Non scrivere il nome del giocatore.

        Genera solo il contenuto del rapporto. Non includere spiegazioni, istruzioni, o altre informazioni oltre a quelle richieste.
        ###Report generato:
        """
    elif lang == 'en':

        prompt = f""""
        You are a professional football scout with expertise in analyzing players' technical and tactical characteristics. 
        I need you to generate a detailed report for a player, based on the provided list of actions that describe their playing style during matches. 
        Your task is to analyze this data and provide a report as follows:

        ### Input Data:
            - Player: {player_name}
            - Actions: 
            {vocab_results['documents'][0]}

        ### Output Format:
        Your report should be structured in the following way:
        **Player**: {player_name}
        **Strengths**: 
        Highlight the player's key strengths evident from their playing style.
        **Weaknesses**: 
        Point out areas where the player needs improvement.
        **Preferred areas of the field**: 
        Identify the areas on the field where the player is most active or performs best.

        ### Notes for Analysis:
        - Use concise and professional language.
        - The report should be realistic for scouting purposes.
        - Do not generate code or class structures. Focus only on the football analysis.
        - The output must be in plain text, clearly formatted according to the structure above.
        - Do not write the name of the player 
        
        Provide the report below this prompt, clearly labeled as "Generated Report".
        ###Generated Report:
        """
    else:
        raise KeyError('Linguaggio non supportato')
        
    return prompt

def getAppearances(df_dataset: pd.DataFrame, p: str, season: str) -> int:
    df_presence = df_dataset[(df_dataset['playerId'] == p) & (df_dataset['season'] == season)]['row_count']
    return int(df_presence.loc[df_presence.index[0]])

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Generate player and team descriptions")
    parser.add_argument("--vector_store_dir", type=str, required=True, help="Path to the directory containing the dataset file.")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Path to the directory containing the dataset file.")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to the directory containing the output file.")
    parser.add_argument("--lang", type=str, required=True, choices=["it", "en"], help="Language option. Choose between 'it' (Italian) or 'en' (English).")
    args = parser.parse_args()
    #vector store che contiene documenti ed embeddings di tutti i documenti (i.e. player-match)
    client = chromadb.PersistentClient(path=args.vector_store_dir)
    vocab_name = "vocab" if args.lang == 'it' else "vocab_en"
    vocab_collection = client.get_collection(name=vocab_name)
    isTeam = True
    season = '2023-2024'
    #"average_player_embeddings_version2"
    if not isTeam:
        collection_name = "average_player_embeddings_version2" if args.lang == 'it' else "average_player_embeddings_en"
    else:
        collection_name = "average_player_embeddings_season"
    player_collection = client.get_collection(name= collection_name)
    print(player_collection.metadata)
    #player_season_collection = client.get_collection(name=f"average_player_embeddings_season_{args.lang}")
    collections = client.list_collections()
    # Mostra i nomi delle collezioni
    for collection in collections:
        print(f"Collection Name: {collection.name}")
    records = player_collection.get(limit=1)
    print(records['metadatas'])
    #p = 349207.0 #Rafa Leao
    #p = 303115.0 #Theo
    #p = 300713.0 #Mbappe
    #p = 11119.0 #Messi
    #p=141646.0 #maignan
    #p=480249.0 #yamal

    dir = 'Dataset/Events2Text' if args.lang == 'it' else 'Dataset/Events2TextEN'
    #dir = 'Dataset'
    filename = 'player2vec_dataset.json' if args.lang == 'it' else 'player2vec_dataset_en.json'
    df_dataset = pd.read_json(f'{dir}/{filename}')
    player_season_counts = df_dataset.groupby(['playerId', 'playerName', 'season']).size().reset_index(name='row_count')

    #players_path = 'Dataset/Events2Text/TeamPlayerList'

    players_dict = getPlayersDict(args.dataset_dir, season) 
    teams_dict = getTeamsPlayerDict(args.dataset_dir)[season]
    

    #prompt = prompt_player_description(str(p), players_dict, player_collection, vocab_collection, lang = args.lang)
    #print(prompt)
    
    

    prompt = f"""Sei un osservatore calcistico professionista con esperienza nell'analisi delle caratteristiche tecniche e tattiche dei giocatori.
    Devi generare un rapporto dettagliato su un giocatore, basandoti sull'elenco delle azioni fornite che descrivono il suo stile di gioco durante le partite.
    Il tuo compito è analizzare questi dati e fornire un rapporto strutturato come segue:
    ###Input Data
        - Azioni: ["tiro fuori bersaglio dall'interno dell'area attacco, centro sinistra", "tiro fuori bersaglio dall'interno dell'area attacco, centro destra", "tiro dall'area di rigore attacco, centro sinistra", "parata fuori dall'area di rigore attacco, centro sinistra", "tiro dall'area di rigore attacco, centro destra", "parata fuori dall'area di rigore attacco, centro destra", "tiro dall'area piccola attacco, centro sinistra", 'azione eseguita con una parte del corpo diversa attacco, centro sinistra', 'parata su tiro fuori area attacco, centro sinistra', 'azione offensiva attacco, centro sinistra', 'azione di contropiede attacco, centro sinistra', 'parata su tiro fuori area attacco, centro destra', 'tiro bloccato attacco, centro sinistra', 'azione offensiva attacco, centro destra', 'azione eseguita con i piedi attacco, centro sinistra', "tiro dall'area piccola attacco, centro destra", "parata nell'area di rigore attacco, centro sinistra", 'azione bloccata attacco, centro sinistra', 'azione di contropiede attacco, centro destra', "tiro fuori bersaglio dall'interno dell'area attacco, centrale"]

    Restituisci il report nel seguente formato:

    ###Formato output
        *Caratteristiche*:
        Evidenzia le caratteristiche tecnico tattiche che meglio rappresentano il giocatore
        *Punti di forza*:
        Evidenzia i punti di forza principali del giocatore emersi dal suo stile di gioco.
        *Debolezze*:
        Indica le aree in cui il giocatore deve migliorare.
        *Zone del campo predilette*:
        Identifica le zone del campo in cui il giocatore è più attivo o performa meglio.

    ###Note per l'analisi
        - Usa un linguaggio conciso e professionale.
        - Il report deve essere realistico e utile per scopi di osservazione calcistica.
        - Non generare codice o strutture di classe. Concentrati solo sull'analisi calcistica.
        - L'output deve essere in testo semplice, chiaramente formattato secondo la struttura sopra indicata.
        - Non scrivere il nome del giocatore.

    ###Report generato:
    """
    
    iteration_dict = teams_dict if isTeam else players_dict


    description_dataset = {}
    with tqdm(total=len(iteration_dict.keys()), desc="Processing players") as pbar:

        for p in iteration_dict.keys():
            pbar.update(1)
            #p = "349207.0"
            #p="255777.0"

            new_record = {}

            if not isTeam:
                prompt = prompt_player_description(str(p), players_dict, player_collection, vocab_collection, lang = args.lang)
                new_record['name'] = players_dict[p] 
            else:
                team = iteration_dict[p]['players']
                prompt = prompt_team_description(team, season, players_dict, player_collection, vocab_collection, player_season_counts,lang=args.lang)
                new_record['name'] = iteration_dict[p]['name'] 

            
            
            new_record['prompt'] = prompt
            #new_record['description'] = generated_text
            
            description_dataset[p] = new_record
            #print(description_dataset)
            

    if not isTeam:
        utils.writeJson(description_dataset, f'{args.output_dir}/players_description_{args.lang}_v3.json')
    else:
        utils.writeJson(description_dataset, f'{args.output_dir}/team_description_{args.lang}.json')

    #with open('prova_report_leao_qnt_it.txt', 'w') as f:
    #    f.write(generated_text)

