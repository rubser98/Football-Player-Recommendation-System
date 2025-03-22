from utils import readJson
from description_generation import cleanDesc
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from typing import List, Dict


class PlayerRecommendation:
    def __init__(self, dir, embedding_model_name="sentence-transformers/all-MiniLM-L6-v2", llm_model="qwen-7b-instruct"):
        """Inizializza la knowledge base e il modello di raccomandazione."""
        # Modello di embedding
        self.embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)
        
        # Database vettoriale (ChromaDB)
        self.vector_db_players = Chroma(embedding_function=self.embedding_model, persist_directory="./chroma_players")
        self.vector_db_teams = Chroma(embedding_function=self.embedding_model, persist_directory="./chroma_teams")
        
        
        # LLM per generare raccomandazioni
        self.llm = ChatOpenAI(model=llm_model, temperature=0.7)
        
        # Prompt per la raccomandazione
        self.prompt_template = PromptTemplate(
            input_variables=["team_description", "player_role", "player_list"],
            template="""
            You are a football scouting expert. Given the team description and the role they are looking for, analyze the best players from the pool.

            Team Description: {team_description}
            Target Role: {player_role}

            Available Players:
            {player_list}

            Rank the players based on suitability and provide justifications.

            Output format:
            - ID: <player_id>
            - Name: <player_name>
            - Justification: <reason_for_recommendation>
            """
        )

        # Chain per generare raccomandazioni
        self.recommendation_chain = LLMChain(llm=self.llm, prompt=self.prompt_template)

    def add_players(self, players_file: str):
        players = readJson(players_file)
        """Aggiunge giocatori alla knowledge base."""
        docs = [(p["description"], {"id": p["id"], "name": p["name"]}) for p in players]
        self.vector_db.add_texts([d[0] for d in docs], metadatas=[d[1] for d in docs])
    
    def add_teams(self, teams_file: str):
        teams = readJson(teams_file)
        
        self.vector_db_teams.add_texts([team_description], metadatas=[{"id": team_id}])


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




