from utils import readJson
from description_generation import cleanDesc
#from langchain.vectorstores import Chroma
from langchain_community.vectorstores import Chroma
#from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from typing import List, Dict
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
import argparse

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
        self.llm = ChatOpenAI(model=llm_model, temperature=0.7)
        
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
            """
        )

        # Chain per generare raccomandazioni
        self.recommendation_chain = LLMChain(llm=self.llm, prompt=self.prompt_template)

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
    
    def initialize_teams_db(self):
        teams = readJson(f'{self.dir}/team_descriptions.json')
        teams = teams | readJson(f'{self.dir}/team_descriptions_mancanti.json')
        team_description = [t['description']['general'] for t in teams.values()]
        team_name = list(teams.keys())
        self.vector_db_teams.add_texts(team_description, metadatas=team_name)


    def retrieve_players(self, team_desc: str, role: str, top_k: int = 5) -> List[Dict]:
        """Recupera i giocatori più pertinenti alla descrizione della squadra e al ruolo richiesto."""
        query = f"{team_desc}. Looking for a {role}."
        query_embedding = self.embedding_model.embed_query(query)
        results = self.vector_db.similarity_search_by_vector(query_embedding, k=top_k)
        
        return [{"id": p.metadata["id"], "name": p.metadata["name"], "description": p.page_content} for p in results]

    def recommend_players(self, team_desc: str, role: str) -> str:
        """Genera la classifica dei migliori giocatori per la squadra."""
        retrieved_players = self.retrieve_players(team_desc, role, top_k=5)
        
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



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate player and team descriptions")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Path to the directory containing the dataset file.")

    args = parser.parse_args()


    recommender = PlayerRecommendation(args.dataset_dir)

