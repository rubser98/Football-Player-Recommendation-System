from utils import readJson, writeJson
from description_generation import cleanDesc
#from langchain.vectorstores import Chroma
from langchain_community.vectorstores import Chroma
#from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
#from langchain.chat_models import ChatOpenAI
from langchain_community.chat_models import ChatOllama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from typing import List, Dict
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import argparse
import warnings
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline
import torch
import numpy as np
warnings.filterwarnings("ignore")


class PlayerRecommendation:
    def __init__(self, dir, embedding_model_name="sentence-transformers/all-MiniLM-L6-v2", llm_model="qwen-7b-instruct"):
        """Inizializza la knowledge base e il modello di raccomandazione."""
        # Modello di embedding
        self.embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)
        self.dir = dir
        # Database vettoriale (ChromaDB)
        
        self.vector_db_players = Chroma(embedding_function=self.embedding_model, persist_directory="./chroma_players")
        self.vector_db_teams = Chroma(embedding_function=self.embedding_model, persist_directory="./chroma_teams")

        if self.vector_db_players._collection.count() == 0:
            self.initialize_players_db()
        if self.vector_db_teams._collection.count() == 0:
            self.initialize_teams_db()

        # LLM per generare raccomandazioni
        #self.llm = ChatOpenAI(model=llm_model, temperature=0.7)
        model_name = 'Qwen/Qwen2.5-7B-Instruct'
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            load_in_8bit=True, 
            llm_int8_enable_fp32_cpu_offload=True,  # Solo se è su CPU
            offload_folder='offload_weights'  # Solo se è su CPU
            )
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
            # Imposta il token di padding
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token 

        if model.config.pad_token_id is None:
            model.config.pad_token_id = tokenizer.eos_token_id
        # Carica il modello con supporto per CUDA (se disponibile)
        hf_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=500)

        retrieval_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=200)
        
        #self.llm = ChatOllama(model="qwen-7b-instruct", temperature=0.7)
        self.llm = HuggingFacePipeline(pipeline=hf_pipeline)
        self.retrieval_llm = HuggingFacePipeline(pipeline=retrieval_pipeline)

        self.team_mapping = readJson(f'{dir}/merged_teams.json')
        self.related_positions = readJson(f'{dir}/related_positions.json')

        
        # Prompt per la raccomandazione
        self.prompt_template = PromptTemplate(
            input_variables=["team_description", "player_role","role_skill", "player_list"],
            template="""
            You are a football scouting and tactical analysis expert. 
            ## Task:
            You are given a team's tactical and technical description along with a specific role they are looking to fill. Your task is to analyze the team's needs and recommend the most suitable players from a predefined pool.
            
            ## Reasoning process
            - Detect each skill required for the role
            - Evaluate available players: Compare each player's scouting report against the each detected skill.
            - Rank players based on whos are better for each skill
            - Justify selections concisely: Provide a clear and short reasoning for why each player fits the team and role.


            ## Input format:
		    - Team description: Team's tactical and technical description
		    - Target role: Position where the team is looking for a players
            - Required characteristics: A list of key attributes and skills the player should have to fit the role. 
		    - Available player pool: A list of players, where each player has the following information: id, name, description
		      The description contains a scouting report outlining the player's playing style, strengths, weaknesses, and key attributes.
		
            ##Required output:
		    Provide a ranked list of recommended players in descending order of relevance. Select 10 top recommendations.
            The output format must be as follows:
            Output format:
            - ID: <player_id>
            - Name: <player_name>
            - Justification: <reason_for_recommendation>
            Each justification should explain why the player is a good fit based on tactical compatibility, technical attributes, and adaptability to the team's playing style. If no ideal player is found, suggest the closest alternatives.
            The justifications should be very concise, at maximum 20 tokens

            ##Output Guidelines:
            - Justifications must be concise (max **20 tokens**).  
            - Add [END_REPORT] at the end of the recommendation. 

            ## Input data:
            Team Description: {team_description}
            Target Role: {player_role}
            Required characteristics: {role_skill}
            Available Players:
            {player_list}

            ##Recommendation
            """
        )

        self.prompt_per_retrieval = PromptTemplate(
            input_variables=['team_description', 'player_role'],
            template="""You are an expert football analyst. 
            ##Task
            Analyze a team's playing style and generate a concise description of the key attributes the team values for a specific player role.  
            
            ## Reasoning Process:
            - **Understand the team's tactical identity**: Analyze the provided team description to determine their style of play, formation tendencies, and strategic principles.  
            - **Identify role-specific demands**:  Extract the key responsibilities of the given player role within the team's system.  
            - **Determine essential attributes**: Define the technical, tactical, physical, and mental qualities the team prioritizes for this role.  
            - **Generate a tailored description**: Summarize the key characteristics in a structured and role-specific way, avoiding generalizations.  
            
            ### Input Format:
            - Team description: Team's tactical and technical description
		    - Target role: Position where the team is looking for a players

            ### Output Format:
            Provide a concise yet detailed paragraph (3-4 sentences) that highlights **specific attributes** the team looks for in this role. Focus on **technical, tactical, physical, and mental qualities** that align with the team's playing style. 
            Ensure that the description is precise and role-specific, avoiding general or vague statements.  

            ## Output Guidelines:
            - Add [END_REPORT] at the end of summarization.

            **Input:**
            - Team Description: {team_description}
            - Player Role: {player_role}

            ##Generated output:
            """
        )
        
        # Chain per generare raccomandazioni
        self.recommendation_chain = LLMChain(llm=self.llm, prompt=self.prompt_template)
        self.retrieval_chain = LLMChain(llm=self.retrieval_llm, prompt=self.prompt_per_retrieval)

    def get_team_mapped(self, team):

        for tm_team in self.team_mapping.keys():
            if tm_team.upper() == team.upper():
                return self.team_mapping[tm_team]
        
        raise KeyError("Team non trovato")
        
    
    def initialize_players_db(self):
        players_desc = readJson(f'{self.dir}/player_descriptions.json')
        players_desc = players_desc | readJson(f'{self.dir}/player_descriptions_mancanti.json')
        player_meta = readJson(f'{self.dir}/transfermarkt_fbref_dataset.json')
        player_meta_manc = readJson(f'{self.dir}/transfermarkt_fbref_mancanti.json')
        #merge giocatori mancanti con dataset originale
        for i in range(len(player_meta_manc)):
            for j in range(len(player_meta)):
                if player_meta_manc[i]['stats'] != {} and player_meta_manc[i]['id'] == player_meta[j]['id']:
                    player_meta[j] = player_meta_manc[i]
        docs = []
        with tqdm(total=len(players_desc.keys()), desc="Processing players") as pbar:    
            for id, desc_dict in players_desc.items():
 
                desc = ' '.join(list(desc_dict['description'].values()))
                pos_str = desc_dict['position_str']
                meta = {}
                for rec in player_meta:
                    if rec['id'] == id:
                        meta = rec
                        break
                meta.pop('stats')
                meta['position_str'] = pos_str

                docs.append((desc, meta))
                pbar.update(1)

        """Aggiunge giocatori alla knowledge base."""
        #docs = [(p["description"], {"id": p["id"], "name": p["name"]}) for p in players_desc]
        self.vector_db_players.add_texts([d[0] for d in docs], metadatas=[d[1] for d in docs])
    
    def get_player_by_id(self, id: str) -> Dict:
        results = self.vector_db_players.get(where={"id": id})
    
        if not results["documents"]:
            return {"error": "Player not found"}
        
        description = results["documents"][0]
        embedding = self.embedding_model.embed_query(description)
        
        return results['metadatas'][0] | {'description': results["documents"][0], 'embedding': embedding}
    
    def get_player_by_name(self, name: str) -> Dict:
        results = self.vector_db_players.get(where={"name": name})
    
        if not results["documents"]:
            return {"error": "Player not found"}
        
        description = results["documents"][0]
        embedding = self.embedding_model.embed_query(description)
        return results['metadatas'][0] | {'description': results["documents"][0], 'embedding': embedding}

    def get_team_by_name(self, name: str) -> Dict:

        team = self.get_team_mapped(name)
        results = self.vector_db_teams.get(where={"name": team})
    
        if not results["documents"]:
            return {"error": "Team not found"}
        
        return results['metadatas'][0] | {'description': results["documents"][0]}

    
    def initialize_teams_db(self):
        teams = readJson(f'{self.dir}/team_descriptions.json')
        teams = teams | readJson(f'{self.dir}/team_descriptions_mancanti.json')
        team_description = [t['description']['general'] for t in teams.values()]
        team_name = list(teams.keys())
        team_name = [{'name': x} for x in team_name]
        self.vector_db_teams.add_texts(team_description, metadatas=team_name)

    
    def retrieve_players_without_filter(self, team_desc: str, role: str, role_filter:str, top_k: int = 10) -> List[Dict]:
        """Recupera i giocatori più pertinenti alla descrizione della squadra e al ruolo richiesto."""
        #query = f"{team_desc}. Looking for a {role}."
        query = self.retrieval_chain.run(team_description=team_desc, player_role=role_filter).split('##Generated output:')[1]
        query = cleanDesc(query)
        query_embedding = self.embedding_model.embed_query(query)
        results = self.vector_db_players.similarity_search_by_vector(query_embedding, k=top_k)
        
        return query,[{"id": p.metadata["id"], "name": p.metadata["name"], "description": p.page_content} for p in results]
    

    def retrieve_players(self, team_desc: str, team_name: str, role: str, role_filter: str, top_k: int = 10) -> List[Dict]:
        """Recupera i giocatori più pertinenti alla descrizione della squadra e al ruolo richiesto,
        filtrando prima per il ruolo specificato e poi selezionando i top K più simili."""
        expanded_roles = self.related_positions['it'][role]
        expanded_roles.append(role)
        # Step 1: Filtro per ruolo nel database
        #filtered_results = self.vector_db_players.get(where={"tm_role": role})
        all_ids = self.vector_db_players.get()["ids"]

        # Recupera i dati completi per gli ID
        if all_ids:
            all_data = self.vector_db_players.get(all_ids)
        else:
            return []

        # Filtra i giocatori che appartengono ai ruoli specificati
        filtered_results = [
            {
                "id": all_data["metadatas"][i]["id"],
                "name": all_data["metadatas"][i]["name"],
                "description": all_data["documents"][i],
                "role": all_data["metadatas"][i]["tm_role"]
            }
            for i in range(len(all_data["documents"]))
            if all_data["metadatas"][i]["tm_role"] in expanded_roles
            and all_data["metadatas"][i]['team'] != team_name
        ]
        
        if not filtered_results:
            return []
        
        ##Generated output:
        query = self.retrieval_chain.run(team_description=team_desc, player_role=role_filter).split('##Generated output:')[1]
        query = cleanDesc(query)
        # Step 2: Creazione della query per la ricerca vettoriale
        #query = f"{team_desc}. Looking for a {role_filter}."
        query_embedding = self.embedding_model.embed_query(cleanDesc(query))
        
        # Step 3: Estrarre gli embeddings dei risultati filtrati
        '''
        filtered_embeddings = [
            (cleanDesc(doc), meta, self.embedding_model.embed_query(cleanDesc(doc))) 
            for doc, meta in zip(filtered_results["documents"], filtered_results["metadatas"])
        ]
        '''
        filtered_embeddings = [
        (cleanDesc(player["description"]), player, self.embedding_model.embed_query(cleanDesc(player["description"])))
        for player in filtered_results]


        # Step 4: Calcolo della similarità tra query e documenti filtrati
        scored_results = sorted(
            filtered_embeddings,
            key=lambda x: cosine_similarity([query_embedding], [x[2]])[0][0], 
            reverse=True
        )

        # Step 5: Prendere i top_k più simili
        top_players = scored_results[:top_k]

        return query, [
            {"id": p[1]["id"], "name": p[1]["name"], "description": p[0]} 
            for p in top_players
        ]


    def recommend_players(self, team_desc: str, team_name: str, role: str, role_filter: str, top_k: int = 10) -> str:
        """Genera la classifica dei migliori giocatori per la squadra."""
        retr_desc,retrieved_players = self.retrieve_players(team_desc, team_name, role, role_filter, top_k=top_k)
        #retr_desc,retrieved_players = self.retrieve_players_without_filter(team_desc, role, role_filter, top_k=top_k)
        
        player_list = "\n".join([
            f"- ID: {p['id']}, Name: {p['name']}, Description: {p['description']}"
            for p in retrieved_players
        ])
        
        len(player_list)
        response = self.recommendation_chain.run(
            team_description=team_desc,
            player_role=role_filter,
            role_skill=retr_desc,
            player_list=player_list
        )
        
        return retr_desc, response
    
    def get_ids_recommendation(self, rec):
    
        ids_rec = []
        for id_str in rec['recommendation'].split('- ID: '):
            id = id_str[:8]
            ids_rec.append(id)
        
        ids_ret = []
        for id_str in rec['prompt'].split('- ID: '):
            id = id_str[:8]
            ids_ret.append(id)

        return ids_rec[1:], ids_ret[2:]
    
    def verify_similarity_in_rec(self, id, ids_rec, threshold=0.7):
        # Ottieni l'embedding del giocatore acquistato usando il filtro per ID
        player_data = self.get_player_by_id(id)
        if not player_data or not player_data['embedding']:
            raise ValueError(f"Embedding non trovato per il giocatore con ID {id}")
        
        player_embedding = np.array(player_data['embedding']).reshape(1, -1)
        # Normalizzazione dell'embedding del giocatore acquistato
        player_embedding = player_embedding / np.linalg.norm(player_embedding)

        results = {}
        for rec_id in ids_rec:
            rec_data = self.get_player_by_id(rec_id)
        
            if not player_data or not player_data['embedding']:
                raise ValueError(f"Embedding non trovato per il giocatore con ID {id}")
                results[rec_id] = 0  # Se non troviamo l'embedding, lo consideriamo non simile
                continue
            
            rec_embedding = np.array(rec_data['embedding']).reshape(1, -1)
            # Normalizzazione dell'embedding del giocatore raccomandato
            rec_embedding = rec_embedding / np.linalg.norm(rec_embedding)
            
            # Calcola la similarità coseno
            similarity = cosine_similarity(player_embedding,rec_embedding)[0][0]
            print(f"ID {rec_id}: Similarità = {similarity}")
            # Assegna 1 se supera la soglia, altrimenti 0
            results[rec_id] = 1 if similarity >= threshold else 0
    
        return results
    
    def evaluate_recommendations(self, recommendations):
        n = len(recommendations)
        hit_rec = 0
        hit_ret = 0
        similarity = {}
        for rec in recommendations:
            id = rec['id']
            ids_rec, ids_ret = self.get_ids_recommendation(rec)
            if id in ids_rec:
                print('rec:', rec['id'], rec['player_name'])
                hit_rec+=1
            if id in ids_ret:
                print('ret:', rec['id'], rec['player_name'])
                hit_ret+=1
            
            similarity_rec = self.verify_similarity_in_rec(id, ids_rec, threshold=0.2)
            similarity_ret = self.verify_similarity_in_rec(id, ids_ret, threshold=0.2)
            similarity[id] = {'recommendation': similarity_rec, 'retrieval': similarity_ret}
        
        eval = {}
        eval['hit_rec'] = hit_rec
        eval['hit_ret'] = hit_ret
        eval['similarity'] = similarity

        return eval
    
    def print_player_attributes(self):
        """Stampa tutti gli attributi presenti nel database dei giocatori."""
        all_data = self.vector_db_players.get()
        
        if not all_data:
            print("Il database dei giocatori è vuoto o non accessibile.")
            return
        
        # Prendi il primo record per mostrare gli attributi disponibili
        sample_record = all_data.get('metadatas', [{}])[0]

        if not sample_record:
            print("Nessun metadato trovato nei dati.")
            return

        print("Attributi disponibili nel database dei giocatori:")
        for key in sample_record.keys():
            print(f"- {key}")
        
        print(sample_record)

    def main_recommendation(self, transfers_file):

        recommendations = []
        transfers = readJson(f'{self.dir}/{transfers_file}')#[:20]
        '''
        with tqdm(total=len(transfers), desc="Processing recommendations") as pbar:   
            for t in transfers:
                team_desc = self.get_team_by_name(t['team'])
                retr_desc,response = self.recommend_players(team_desc['description'], team_desc['name'], t['tm_role'], t['tm_role_en'], top_k=20)
                response = response.split('##Recommendation')
                prompt = response[0]
                rec = cleanDesc(response[1])
                t['recommendation'] = rec
                t['prompt'] = prompt
                t['retrieval_description'] = retr_desc
                recommendations.append(t)

                pbar.update(1)

        writeJson(recommendations, f'{self.dir}/recommendations.json')'
        '''

        recommendations = readJson(f'{self.dir}/recommendations.json')[1:]
        eval = self.evaluate_recommendations(recommendations)
        writeJson(eval, f'{self.dir}/evaluation.json')





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate player and team descriptions")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Path to the directory containing the dataset file.")

    args = parser.parse_args()

    recommender = PlayerRecommendation(args.dataset_dir)
    recommender.main_recommendation('log_trasferimenti_gt.json')
    

