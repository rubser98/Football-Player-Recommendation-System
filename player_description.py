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
    results = player_collection.get(where={'id': str(p)}, include=['embeddings'])['embeddings']
    vocab_results = vocab_collection.query(query_embeddings=results, n_results = k, include=['documents'])
    if lang == 'it':
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
    elif lang == 'en':
        prompt = f"""
        You are a football scout. I need you to create a report for player {player_name}, highlighting their technical and tactical characteristics, strengths, and weaknesses. 
        Here is a list of action describing the playing style performed during the matches:
        {vocab_results['documents']}
        Provide the report in the following format:
            Characteristics:
            Strengths:
            Weaknesses:
            Preferred areas of the field: 
        ###
        """

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

        ### Generated Report:
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
    #"average_player_embeddings_version2"
    player_collection = client.get_collection(name= f"average_player_embeddings_{args.lang}")
    #player_season_collection = client.get_collection(name=f"average_player_embeddings_season_{args.lang}")

    #records = player_collection.get(limit=1)
    #print(records['metadatas'])
    #p = 349207.0 #Rafa Leao
    #p = 303115.0 #Theo
    #p = 300713.0 #Mbappe
    #p = 11119.0 #Messi
    #p=141646.0 #maignan
    #p=480249.0 #yamal

    dir = 'Dataset/Events2Text' if args.lang == 'it' else 'Dataset/Events2TextEN'
    dir = 'Dataset'
    filename = 'player2vec_dataset.json' if args.lang == 'it' else 'player2vec_dataset_en.json'
    df_dataset = pd.read_json(f'{dir}/{filename}')
    player_season_counts = df_dataset.groupby(['playerId', 'playerName', 'season']).size().reset_index(name='row_count')

    #players_path = 'Dataset/Events2Text/TeamPlayerList'
    players_dict = getPlayersDict(args.dataset_dir)

    #prompt = prompt_player_description(str(p), players_dict, player_collection, vocab_collection, lang = args.lang)
    #print(prompt)
    
    prompt= """
    You are a football scout. I need you to create a report for player Rafael Leao, highlighting their technical and tactical characteristics, strengths, and weaknesses. 
    Here is a list of action describing the playing style performed during the matches:
    ["shot off target from inside the attacking area, left center", "shot off target from inside the attacking area, right center", "shot from the penalty area, left center", "save outside the penalty area, left center", "shot from the penalty area, right center", "save outside the penalty area, right center", "shot from the six-yard box, left center", "action performed with a body part other than the feet, left center", "save on a shot from outside the penalty area, left center", "offensive action, left center", "counterattack action, left center", "save on a shot from outside the penalty area, right center", "blocked shot, left center", "offensive action, right center", "action performed with the feet, left center", "shot from the six-yard box, right center", "save inside the penalty area, left center", "blocked action, left center", "counterattack action, right center", "shot off target from inside the attacking area, central"]]
    Provide the report in the following format:
        Characteristics:
        Strengths:
        Weaknesses:
        Preferred areas of the field: 
    
    ###
    """

    prompt = f"""Sei un osservatore calcistico professionista con esperienza nell'analisi delle caratteristiche tecniche e tattiche dei giocatori.
    Devi generare un rapporto dettagliato su un giocatore, basandoti sull'elenco delle azioni fornite che descrivono il suo stile di gioco durante le partite.
    Il tuo compito è analizzare questi dati e fornire un rapporto strutturato come segue:
    ###Input Data
        - Giocatore: Rafael Leao 
        - Azioni: ["tiro fuori bersaglio dall'interno dell'area attacco, centro sinistra", "tiro fuori bersaglio dall'interno dell'area attacco, centro destra", "tiro dall'area di rigore attacco, centro sinistra", "parata fuori dall'area di rigore attacco, centro sinistra", "tiro dall'area di rigore attacco, centro destra", "parata fuori dall'area di rigore attacco, centro destra", "tiro dall'area piccola attacco, centro sinistra", 'azione eseguita con una parte del corpo diversa attacco, centro sinistra', 'parata su tiro fuori area attacco, centro sinistra', 'azione offensiva attacco, centro sinistra', 'azione di contropiede attacco, centro sinistra', 'parata su tiro fuori area attacco, centro destra', 'tiro bloccato attacco, centro sinistra', 'azione offensiva attacco, centro destra', 'azione eseguita con i piedi attacco, centro sinistra', "tiro dall'area piccola attacco, centro destra", "parata nell'area di rigore attacco, centro sinistra", 'azione bloccata attacco, centro sinistra', 'azione di contropiede attacco, centro destra', "tiro fuori bersaglio dall'interno dell'area attacco, centrale"]

    Restituisci il report nel seguente formato:

    ###Formato output
        *
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
    """
    
    isTeam = False

    if isTeam:
        team_dict = getTeamsPlayerDict(args.dataset_dir)
        team = team_dict['2021-2022']['80']['players']
        team_prompt = prompt_team_description(team, '2021-2022', players_dict, player_season_collection, vocab_collection, player_season_counts)
    #print(team_prompt)


    #model_name='rstless-research/DanteLLM-7B-Instruct-Italian-v0.1'
    
    #login()
    model_name = 'meta-llama/Meta-Llama-3.1-8B-Instruct'
    model_name = 'meta-llama/Llama-3.1-8B'
    #model_name = 'meta-llama/Llama-3.2-3B'
    #model_name = "galatolo/cerbero-7b"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto", 
        load_in_8bit=True, 
        llm_int8_enable_fp32_cpu_offload=True,
        offload_folder='offload_weights')
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Imposta il token di padding
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token 
    
    description_dataset = {}
    with tqdm(total=len(players_dict.keys()), desc="Processing players") as pbar:

        for p in players_dict.keys():
            pbar.update(1)
            p = "349207.0"
            #prompt = prompt_player_description(str(p), players_dict, player_collection, vocab_collection, lang = args.lang)
            input_ids = tokenizer(prompt, return_tensors="pt", truncation=True).input_ids.cuda()
            attention_mask = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).attention_mask.cuda()

            with torch.no_grad():
                outputs = model.generate(input_ids=input_ids,
                                        attention_mask=attention_mask, 
                                        max_new_tokens=2000, 
                                        temperature=0.2,     # Modifica la temperatura qui
                                        top_k=20,            # Filtraggio top-k opzionale
                                        top_p=0.8,           # Nucleus sampling (top-p sampling) opzionale
                                        do_sample=True 
                                        #pad_token_id=tokenizer.eos_token_id
                                        )
            
            #print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0].split("[/INST]")[1])
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True).split("### Generated Report:")[1]

            new_record = {}
            new_record['name'] = players_dict[p]
            new_record['prompt'] = prompt
            new_record['description'] = generated_text
            
            description_dataset[p] = new_record
            print(description_dataset)
            break
    
    utils.writeJson(description_dataset, f'{args.output_dir}/players_description.json')
    #with open('prova_report_leao_qnt_it.txt', 'w') as f:
    #    f.write(generated_text)

