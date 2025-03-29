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
        hf_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)
        
        #self.llm = ChatOllama(model="qwen-7b-instruct", temperature=0.7)
        self.llm = HuggingFacePipeline(pipeline=hf_pipeline)

        self.team_mapping = readJson(f'{dir}/merged_teams.json')

        
        # Prompt per la raccomandazione
        self.prompt_template = PromptTemplate(
            input_variables=["team_description", "player_role", "player_list"],
            template="""
            You are a football scouting and tactical analysis expert. 
            ## Task:
            You are given a team's tactical and technical description along with a specific role they are looking to fill. Your task is to analyze the team's needs and recommend the most suitable players from a predefined pool.
            
            ## Input format:
		    - Team description: Team's tactical and technical description
		    - Target role: Position where the team is looking for a players
		    - Available player pool: A list of players, where each player has the following information: id, name, description
		      The description contains a scouting report outlining the player's playing style, strengths, weaknesses, and key attributes.
		
            ##Required output:
		    Provide a ranked list of recommended players in descending order of relevance. The output format must be as follows:
            Output format:
            - ID: <player_id>
            - Name: <player_name>
            - Justification: <reason_for_recommendation>
            Each justification should explain why the player is a good fit based on tactical compatibility, technical attributes, and adaptability to the team's playing style. If no ideal player is found, suggest the closest alternatives.

            ## Input data:
            Team Description: {team_description}
            Target Role: {player_role}
            Available Players:
            {player_list}

            ##Recommendation
            """
        )
        
        # Chain per generare raccomandazioni
        self.recommendation_chain = LLMChain(llm=self.llm, prompt=self.prompt_template)

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
        
        return results['metadatas'][0] | {'description': results["documents"][0]}
    
    def get_player_by_name(self, name: str) -> Dict:
        results = self.vector_db_players.get(where={"name": name})
    
        if not results["documents"]:
            return {"error": "Player not found"}
        
        return results['metadatas'][0] | {'description': results["documents"][0]}

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

    '''
    def retrieve_players(self, team_desc: str, role: str,role_filter:str, top_k: int = 10) -> List[Dict]:
        """Recupera i giocatori più pertinenti alla descrizione della squadra e al ruolo richiesto."""
        query = f"{team_desc}. Looking for a {role}."
        query_embedding = self.embedding_model.embed_query(query)
        results = self.vector_db.similarity_search_by_vector(query_embedding, k=top_k)
        
        return [{"id": p.metadata["id"], "name": p.metadata["name"], "description": p.page_content} for p in results]
    '''

    def retrieve_players(self, team_desc: str, role: str, role_filter: str, top_k: int = 10) -> List[Dict]:
        """Recupera i giocatori più pertinenti alla descrizione della squadra e al ruolo richiesto,
        filtrando prima per il ruolo specificato e poi selezionando i top K più simili."""
        
        # Step 1: Filtro per ruolo nel database
        filtered_results = self.vector_db_players.get(where={"tm_role": role})
        
        if not filtered_results["documents"]:
            return []

        # Step 2: Creazione della query per la ricerca vettoriale
        query = f"{team_desc}. Looking for a {role_filter}."
        query_embedding = self.embedding_model.embed_query(query)
        
        # Step 3: Estrarre gli embeddings dei risultati filtrati
        filtered_embeddings = [
            (cleanDesc(doc), meta, self.embedding_model.embed_query(cleanDesc(doc))) 
            for doc, meta in zip(filtered_results["documents"], filtered_results["metadatas"])
        ]

        # Step 4: Calcolo della similarità tra query e documenti filtrati
        scored_results = sorted(
            filtered_embeddings,
            key=lambda x: cosine_similarity([query_embedding], [x[2]])[0][0], 
            reverse=True
        )

        # Step 5: Prendere i top_k più simili
        top_players = scored_results[:top_k]

        return [
            {"id": p[1]["id"], "name": p[1]["name"], "description": p[0]} 
            for p in top_players
        ]


    def recommend_players(self, team_desc: str, role: str, role_filter: str, top_k: int = 10) -> str:
        """Genera la classifica dei migliori giocatori per la squadra."""
        retrieved_players = self.retrieve_players(team_desc, role, role_filter, top_k=top_k)
        
        player_list = "\n".join([
            f"- ID: {p['id']}, Name: {p['name']}, Description: {p['description']}"
            for p in retrieved_players
        ])
        
        response = self.recommendation_chain.run(
            team_description=team_desc,
            player_role=role,
            player_list=player_list
        )
        
        return response

    def main_recommendation(self, transfers_file):

        recommendations = []
        transfers = readJson(f'{self.dir}/{transfers_file}')[:10]
        
        with tqdm(total=len(transfers), desc="Processing recommendations") as pbar:   
            for t in transfers:
                team_desc = self.get_team_by_name(t['team'])
                response = self.recommend_players(team_desc, t['tm_role'], t['tm_role_en'], top_k=5).split('##Recommendation')
                prompt = response[0]
                rec = response[1]
                t['recommendation'] = rec
                t['prompt'] = prompt
                recommendations.append(t)

                pbar.update(1)
        
        writeJson(recommendations, f'{self.dir}/recommendations.json')




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate player and team descriptions")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Path to the directory containing the dataset file.")

    args = parser.parse_args()

    recommender = PlayerRecommendation(args.dataset_dir)
    recommender.main_recommendation('log_trasferimenti_gt.json')
    

