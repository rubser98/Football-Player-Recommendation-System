import os
import logging
from neo4j import GraphDatabase
import networkx as nx
from networkx.algorithms import community as nx_comm_bipartite
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from collections import defaultdict
from utils import writeJson, readJson

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TeamProfiler:
    """
    Classe per analizzare una squadra da un KG Neo4j, eseguire community detection
    bipartita (Player-Skill) e generare un profilo tramite LLM.
    """
    def __init__(self, uri, user, password):
        """Inizializza la connessione a Neo4j e il client LLM."""
        self.uri = uri
        self.user = user
        self.password = password

        try:
            self._driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            self._driver.verify_connectivity()
            logging.info("Connessione a Neo4j stabilita con successo.")
        except Exception as e:
            logging.error(f"Errore durante la connessione a Neo4j: {e}")
            raise
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        load_in_8bit = True if device == "cuda" else False

        model_name='Qwen/Qwen2.5-7B-Instruct'
        model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",  # Distribuisce automaticamente su GPU/CPU
                #load_in_8bit=load_in_8bit,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32, # float16 su GPU se possibile
                # Aggiungi i parametri di offload se necessari e decommentati sopra:
                # llm_int8_enable_fp32_cpu_offload=llm_int8_enable_fp32_cpu_offload,
                # offload_folder=offload_folder,
                trust_remote_code=True # Necessario per alcuni modelli come Qwen
            )
        
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        logging.info(f"Modello {model_name} caricato con successo.")

        # --- Creazione Pipeline Transformers e Integrazione LangChain ---
        # Aumenta max_new_tokens se i profili generati sono troncati
        hf_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=2000, # Massimo numero di token da generare
            # Parametri aggiuntivi per la generazione (opzionali):
            temperature=0.7,
            # top_p=0.9,
            repetition_penalty=1.1
        )

        # Wrapper LangChain
        self._llm = HuggingFacePipeline(pipeline=hf_pipeline)
        

        # Mappatura livelli skill a pesi numerici (opzionale, ma utile)
        self.skill_level_weights = {
            "WEAK_IN": 1,
            "AVERAGE_IN": 2,
            "GOOD_IN": 3,
            "VERY_GOOD_IN": 4,
            "EXCELLENT_IN": 5
        }

    def close(self):
        """Chiude la connessione al driver Neo4j."""
        if self._driver:
            self._driver.close()
            logging.info("Connessione a Neo4j chiusa.")

    def _extract_team_subgraph_data(self, team_name: str) -> list:
        """
        Estrae i giocatori della squadra, le loro skill e i livelli dal KG.
        Restituisce una lista di dizionari con i dati delle relazioni Player-Skill.
        """
        query = """
        MATCH (t:team {name: $team_name})<-[:PLAY_FOR]-(p:player)
        OPTIONAL MATCH (p)-[r]->(s:skill)
        WHERE type(r) IN $skill_levels // Filtra solo per relazioni di skill
        RETURN p.name AS player_name,
               p.preferred_position AS player_position,
               s.name AS skill_name,
               s.description AS skill_description,
               type(r) AS skill_level,
               id(p) as player_id, // ID univoco per networkx
               id(s) as skill_id   // ID univoco per networkx
        """
        params = {
            "team_name": team_name,
            "skill_levels": list(self.skill_level_weights.keys())
        }
        data = []
        try:
            with self._driver.session() as session:
                results = session.run(query, params)
                data = [record.data() for record in results]

            if not data:
                 return [] # Restituisci lista vuota se non ci sono dati
            
        except Exception as e:
            logging.error(f"Errore durante l'estrazione dei dati da Neo4j: {e}")
            raise
        return data

    def _build_bipartite_graph(self, subgraph_data: list):
        """
        Costruisce un grafo bipartito NetworkX (Player, Skill) dai dati estratti.
        """
        B = nx.Graph()
        player_nodes = set()
        skill_nodes = set()

        if not subgraph_data:
            logging.warning("Dati del sottografo vuoti, impossibile costruire il grafo.")
            return B, player_nodes, skill_nodes # Grafo vuoto

        for record in subgraph_data:
            player_id = f"p_{record['player_id']}" # Prefisso per evitare collisioni ID
            skill_id = f"s_{record['skill_id']}"   # Prefisso per evitare collisioni ID

            # Aggiungi nodi con attributo bipartite e altri dati utili
            if player_id not in player_nodes:
                B.add_node(player_id, bipartite=0, name=record['player_name'], position=record['player_position'])
                player_nodes.add(player_id)

            if record['skill_name'] and skill_id not in skill_nodes: # Assicurati che la skill esista
                 B.add_node(skill_id, bipartite=1, name=record['skill_name'], description=record["skill_description"])
                 skill_nodes.add(skill_id)

            # Aggiungi arco pesato tra player e skill (se la skill esiste)
            if record['skill_name']:
                 level = record['skill_level']
                 weight = self.skill_level_weights.get(level, 0) # Peso numerico
                 B.add_edge(player_id, skill_id, level=level, weight=weight)

        logging.info(f"Grafo bipartito costruito con {len(player_nodes)} nodi Player e {len(skill_nodes)} nodi Skill.")
        return B, player_nodes, skill_nodes

    def _run_bipartite_community_detection(self, B: nx.Graph, player_nodes: set):
        """
        Esegue l'algoritmo Louvain bipartito per trovare comunità di giocatori.
        """
        if not B or B.number_of_edges() == 0 or not player_nodes: # Aggiunto controllo edges
            logging.warning("Grafo bipartito vuoto, senza archi o senza nodi Player. Impossibile eseguire community detection.")
            return [] # Nessuna comunità

        # Filtra i player_nodes che sono effettivamente nel grafo B
        valid_player_nodes = {p_node for p_node in player_nodes if p_node in B}
        if not valid_player_nodes:
             logging.warning("Nessuno dei nodi player specificati è presente nel grafo B.")
             return []

        try:
            # Trova comunità *di giocatori* basate sulle skill condivise
            # Nota: specificare i nodi del set "top" (i giocatori in questo caso)
            # Usiamo i nodi validi presenti nel grafo
            #communities_generator = nx_comm_bipartite.louvain_communities(B, nodes=valid_player_nodes, resolution=1.0) # Prova a variare la resolution
            communities_generator = nx_comm_bipartite.louvain_communities(B,  resolution=1, threshold=1e-5)
            # Converti il generatore in una lista di set (ogni set è una comunità di player_id_nx)
            # Filtra comunità banali (singoli giocatori)
            communities = [set(comm) for comm in communities_generator if len(comm) > 1]
            logging.info(f"Trovate {len(communities)} comunità di giocatori (non banali).")

            # Log di esempio delle comunità trovate (nomi dei giocatori)
            if communities:
                logging.debug("Comunità trovate (ID NetworkX):")
                for i, comm_ids in enumerate(communities):
                     member_names = [B.nodes[p_id]['name'] for p_id in comm_ids if p_id in B.nodes]
                     logging.debug(f" Comunità {i+1}: {member_names}")

            return communities
        
        except Exception as e:
            # Log più specifico per errori comuni con Louvain (es. grafo disconnesso)
            logging.error(f"Errore durante la community detection bipartita (potrebbe essere dovuto a componenti disconnesse): {e}", exc_info=True)
            # Considera di analizzare i componenti connessi separatamente se necessario
            # components = list(nx.connected_components(B))
            return []

    def _get_community_details(self, B: nx.Graph, communities: list) -> list:
        """
        Estrae dettagli per ogni comunità: membri, posizioni, skill chiave.
        """
        community_details = []
        if not B or not communities:
             return community_details

        node_data = B.nodes(data=True) # Ottieni dati dei nodi una sola volta

        for i, player_ids_in_comm in enumerate(communities):
            comm_data = {
                "id": i + 1,
                "members": [],
                "positions": defaultdict(int),
                "key_skills": defaultdict(lambda: {'count': 0, 'total_weight': 0, 'levels': defaultdict(int), 'description': 'No description available.'})
            }
            player_names_in_comm = set()
            

            # Raccogli info sui membri e posizioni
            for p_id in player_ids_in_comm:
                node_info = node_data[p_id]
                name = node_info.get('name', 'N/A')
                player_pos = node_info.get('position', 'N/A')

                if node_info['bipartite'] == 0:
                    comm_data["members"].append(name)
                    player_names_in_comm.add(p_id) # Set di nomi per ricerca skill
                    comm_data["positions"][player_pos] += 1

            for p_id in player_ids_in_comm:

                node_info = node_data[p_id]
                name = node_info.get('name', 'N/A')

                if node_info['bipartite'] == 1:

                    for player in player_names_in_comm:
                        skill_desc = node_info.get('description', '')
                        edge_data = B.get_edge_data(player, p_id)
                        try:
                            level = edge_data.get('level', 'N/A')
                            weight = edge_data.get('weight', 0)


                            # Aggiorna conteggi e pesi per la skill in questa comunità
                            comm_data["key_skills"][name]['count'] += 1
                            comm_data["key_skills"][name]['total_weight'] += weight
                            comm_data["key_skills"][name]['levels'][level] += 1
                            comm_data["key_skills"][name]['description'] = skill_desc
                        except:
                            continue

            '''
            # Raccogli info sulle skill associate ai membri della comunità
            for p_id in player_ids_in_comm:
                # Trova skill collegate a questo giocatore nel grafo
                for neighbor_skill_id in B.neighbors(p_id):
                    if B.nodes[neighbor_skill_id]['bipartite'] == 1: # Assicurati sia una skill
                        skill_name = B.nodes[neighbor_skill_id]['name']
                        skill_desc = node_data[neighbor_skill_id].get('description', '')
                        print('funge:', p_id, neighbor_skill_id)
                        break
                        edge_data = B.get_edge_data(p_id, neighbor_skill_id)
                        level = edge_data.get('level', 'N/A')
                        weight = edge_data.get('weight', 0)

                        # Aggiorna conteggi e pesi per la skill in questa comunità
                        comm_data["key_skills"][skill_name]['count'] += 1
                        comm_data["key_skills"][skill_name]['total_weight'] += weight
                        comm_data["key_skills"][skill_name]['levels'][level] += 1
                        if skill_desc:
                            comm_data["key_skills"][skill_name]['description'] = skill_desc
            '''

            
            # Calcola skill più rilevanti (es. per peso medio o frequenza > soglia)
            relevant_skills = {}
            #min_players_with_skill = max(1, len(player_ids_in_comm) // 2) 
            for skill, data in comm_data["key_skills"].items():
                #print(skill, data)
                #if data['count'] >= min_players_with_skill:
                    avg_weight = data['total_weight'] / data['count']
                     # Converti livelli in stringa leggibile
                    level_str = ", ".join([f"{lvl}: {cnt}" for lvl, cnt in data['levels'].items()])
                    skill_description = data.get('description', 'No description available.')

                    relevant_skills[skill] = {"summary": f"Avg Weight: {avg_weight:.2f}, Levels count: {level_str})", 
                                              "description": skill_description}

            comm_data["key_skills"] = relevant_skills # Sovrascrivi con le skill filtrate/formattate
            comm_data["positions"] = dict(comm_data["positions"]) # Converti defaultdict a dict

            community_details.append(comm_data)

        return community_details
    
    def _format_single_community_context(self, team_name: str, comm: dict) -> str:
        context = f"Group of players from team: {team_name}\n"
        context += f"Number of players: {len(comm['members'])}\n"
        positions_str = ", ".join([f"{pos} ({cnt})" for pos, cnt in sorted(comm['positions'].items())])
        context += f"Predominant Positions: {positions_str}\n"
        context += "Key Shared Skills:\n"
        
        if comm['key_skills']:
            for skill, details in sorted(comm['key_skills'].items()):
                context += f"- {skill}: {details['summary']}"
                if details['description']:
                    context += f" (Description: {details['description']})"
                context += "\n"
        else:
            context += "- No relevant key skills detected.\n"
        
        return context


    def _format_context_for_llm(self, team_name: str, community_details: list, all_players: list) -> str:
        """Formats the community detection results into ENGLISH text for the LLM."""
        # ENGLISH context
        context = f"Team Analysis: {team_name}\n"
        context += f"Total players analyzed: {len(all_players)}\n\n"

        if not community_details:
            context += "No significant communities of players with similar skills were detected.\n"
            context += "The team might lack specialized core groups or have highly diverse player profiles.\n"
            return context

        context += "The following player communities were identified based on shared skills:\n\n"
        for comm in community_details:
            context += f"--- Community {comm['id']} ---\n"
            #context += f"Members ({len(comm['members'])}): {', '.join(sorted(comm['members']))}\n"
            positions_str = ", ".join([f"{pos} ({cnt})" for pos, cnt in sorted(comm['positions'].items())])
            context += f"Predominant Positions: {positions_str}\n"
            context += "Key Shared Skills (among these members):\n"
            if comm['key_skills']:
                for skill, details in sorted(comm['key_skills'].items()):
                    # <-- Confermato: Formattazione include descrizione
                    context += f"- {skill}: {details['summary']}"
                    if details['description']:
                        context += f" (Description: {details['description']})"
                    context += "\n"
            else:
                context += "- No relevant key skills detected based on criteria.\n"
            context += "\n"

        players_in_communities = set(p for comm in community_details for p in comm['members'])
        isolated_players = sorted([p for p in all_players if p not in players_in_communities])
        if isolated_players:
            context += f"--- Players Not Belonging to Relevant Communities ({len(isolated_players)}) ---\n"
            context += f"{', '.join(isolated_players)}\n"
            context += "These players might have unique skill profiles, be very versatile, or not fully integrate into the identified specialization clusters.\n\n"
        else:
             context += "All analyzed players belong to a relevant community.\n\n"

        return context

    def _generate_profile_with_llm(self, context: str) -> str:
        
        if not context:
            return "Cannot generate profile: no community data available."
        if not self._llm:
            return "Cannot generate profile: LLM not initialized."

        # ENGLISH system prompt
        system_prompt = """You are an expert football analyst tasked with analyzing teams based on structured data.
        You will be provided with the results of a community detection analysis performed on a bipartite Player-Skill graph for a specific team.
        Communities group players who share similar skills. The data includes members, predominant positions, and key shared skills for each community.
        It also lists any players not belonging to a significant community.

        Your task is to generate a narrative team profile highlighting:
        1.  **Strengths:** Describe the team's specializations based on the most cohesive communities and their key skills (especially those with high levels like GOOD_IN, VERY_GOOD_IN, EXCELLENT_IN). Indicate which departments or types of play seem well-covered.
        2.  **Potential Weaknesses or Areas for Improvement:** Based on the community analysis (lack of certain key skills, low average levels even if shared), the potential absence of communities for crucial roles (e.g., if there's no strong community of finishing strikers), or the presence of many isolated players (which might indicate heterogeneity or lack of cohesion).
        3.  **Implied Playing Style:** If possible, infer a likely playing style based on the identified strengths (e.g., defensively solid team, technical midfield, fast wing play).
        
        ## Output Guidelines:
        Use clear, analytical language typical of a scouting report. Do not invent information not present in the provided context. Focus on analyzing the presented data.
        Mandatory: Do not use player names in the scouting report.
        Mandatory: Report must start with [START_REPORT] and [END_REPORT]
        Mandatory: Do not use word "community" in the report. 
        Mandatory: Do not use statistics in the report, use their description to generate discursive report. 
        Mandatory: Do not use words like EXCELLENT_IN
        """

        system_prompt = """You are an expert football analyst tasked with analyzing teams based on structured data.
        You will be provided with the results of a community detection analysis performed on a bipartite Player-Skill graph for a specific team.
        Communities group players who share similar skills. The data includes members, predominant positions, and key shared skills for each community.
        It also lists any players not belonging to a significant community.

        Your task is to generate a narrative team profile highlighting:
        1.  **Strengths:** Describe the team's specializations based on the most cohesive communities and their key skills (especially those with high levels like GOOD_IN, VERY_GOOD_IN, EXCELLENT_IN). Indicate which departments or types of play seem well-covered.
        2.  **Potential Weaknesses or Areas for Improvement:** Based on the community analysis (lack of certain key skills, low average levels even if shared), the potential absence of communities for crucial roles (e.g., if there's no strong group of finishing strikers), or the presence of many isolated players (which might indicate heterogeneity or lack of cohesion).
        3.  **Implied Playing Style:** If possible, infer a likely playing style based on the identified strengths (e.g., defensively solid team, technical midfield, fast wing play).
        4.  **Recruitment Needs by Role:** Based on gaps or underrepresented skill sets, suggest what the team might be looking for in each main role (defender, midfielder, forward). For each role, infer what kind of player profile would strengthen the team (e.g., "a forward with strong finishing and off-the-ball movement" or "a midfielder who can progress the ball under pressure").

        ## Output Guidelines:
        Use clear, analytical language typical of a scouting report. Do not invent information not present in the provided context. Focus on analyzing the presented data.
        Mandatory: Do not use player names in the scouting report.
        Mandatory: Report must start with [START_REPORT] and end with [END_REPORT]
        Mandatory: Do not use the word "community" in the report. 
        Mandatory: Do not use statistics in the report, use their description to generate discursive report. 
        Mandatory: Do not use words like EXCELLENT_IN
        """

        system_prompt = """You are an expert football analyst tasked with translating skill-based analysis into actionable recruitment insights.

        You will be provided with the results of a community detection analysis performed on a bipartite Player-Skill graph for a specific team.
        Players are grouped based on shared skillsets. The analysis lists which players belong to each cluster, their predominant positions, and the most common skills shared among them.
        Some players are also listed as isolated (i.e., not part of any meaningful group).

        Your task is to infer and write the **player requirements** for the team in the following roles:
        - Defenders
        - Full-Backs
        - Midfielders
        - Wingers
        - Forwards

        These requirements must reflect:
        - Gaps or weaknesses observed in the available skillsets
        - Underrepresented or missing qualities by role
        - Reinforcement of strengths through complementary profiles
        - Check if you put only trivial skills (e.g. Expected goals for forwards)

        Use **only the skills mentioned in the data** to infer the types of players that would fit the team's needs. Avoid speculation or general football clichés.

        Each player requirement must:
        - Specify the role and suggested profile 
        - Include key skills the player should possess
        - Briefly explain how the profile fits the team context

        **Do not mention the word “community.”**  
        **Do not use player names.**  
        **Do not use statistics.**  
        **Do not invent skills or traits not found in the data.**  
        **Output must begin with [PLAYER_REQUIREMENTS] and end with [END_PLAYER_REQUIREMENTS]**
        """


        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=system_prompt),
            HumanMessage(content=context) # The actual data context
        ])

        chain = prompt | self._llm

        logging.info("Sending request to local LLM via LangChain...")
        try:
            response = chain.invoke({})
            profile = response.strip()
            logging.info("Team profile generated successfully by local LLM.")
            return profile

        except Exception as e:
            logging.error(f"Error during LangChain LLM chain invocation: {e}", exc_info=True)
            return f"Error generating LLM profile: {e}"
    
    
        
    def _generate_community_profile_with_llm(self, context: str) -> str:
        if not context:
            return "Cannot generate profile: no community data available."
        if not self._llm:
            return "Cannot generate profile: LLM not initialized."

        system_prompt = """You are an expert football analyst tasked with analyzing groups of players based on shared skills.
        You will be provided with a structured summary of a group of players who share similar skill profiles.
        The group includes player count, predominant positions, and shared key skills.

        Your task is to generate a brief scouting-style narrative that:
        1. Describes the specialization of this group (e.g., creative midfielders, defensive fullbacks).
        2. Identifies any interesting patterns (e.g., all fast players, mostly defensive skills).
        3. Suggests what kind of tactical role such a group might serve in a team (e.g., high pressing, counter-attacks).

        ## Output Guidelines:
        - Use analytical, clear language.
        - Do not mention player names.
        - Do not use the word "community".
        - Do not list statistics or skill levels explicitly.
        - Keep it concise and focused (max 4-6 sentences).
        """

        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=system_prompt),
            HumanMessage(content=context)
        ])

        chain = prompt | self._llm

        try:
            response = chain.invoke({})
            return response.strip()
        except Exception as e:
            logging.error(f"Error generating community description: {e}", exc_info=True)
            return f"Error: {e}"
    
    def generate_profile_single_community(self, team_name: str):
        """
        Main method to generate the complete profile for a team with single community profile generation.
        """
        logging.info(f"Starting profile generation for team: {team_name}")

        # 1. Extract raw data
        subgraph_data = self._extract_team_subgraph_data(team_name)
        if not subgraph_data:
            return f"Cannot generate profile: No data found for team '{team_name}' in the database or no associated players/skills."

        all_players_in_team = list(set(record['player_name'] for record in subgraph_data if record.get('player_name')))
        if not all_players_in_team:
             logging.warning(f"No valid player names found for team '{team_name}'.")

        # 2. Build bipartite graph
        B, player_nodes, skill_nodes = self._build_bipartite_graph(subgraph_data)
        if not B or B.number_of_nodes() == 0:
             return f"Cannot generate profile: Bipartite graph not created or empty for '{team_name}' (check extracted data)."
        if B.number_of_edges() == 0:
             logging.warning(f"Bipartite graph for '{team_name}' contains no Player-Skill edges. Profile might be limited.")

        # 3. Run community detection
        valid_player_nodes_for_comm = {p_node for p_node in player_nodes if p_node in B}
        communities = self._run_bipartite_community_detection(B, valid_player_nodes_for_comm)

        # 4. Extract community details
        community_details = self._get_community_details(B, communities)

        final_profile = {}

        for comm in community_details:
            comm_context = self._format_single_community_context(team_name, comm)
            comm_description = self._generate_community_profile_with_llm(comm_context)
            final_profile[comm['id']] = comm_description


        # 5. Format context for LLM (Now in English)
        #llm_context = self._format_context_for_llm(team_name, community_details, all_players_in_team)

        # 6. Generate profile with LLM (Using English prompt)
        #final_profile = self._generate_profile_with_llm(llm_context)

        logging.info(f"Profile for '{team_name}' completed.")
        return final_profile


    def generate_team_profile(self, team_name: str) -> str:
        """
        Main method to generate the complete profile for a team.
        """
        logging.info(f"Starting profile generation for team: {team_name}")

        # 1. Extract raw data
        subgraph_data = self._extract_team_subgraph_data(team_name)
        if not subgraph_data:
            return f"Cannot generate profile: No data found for team '{team_name}' in the database or no associated players/skills."

        all_players_in_team = list(set(record['player_name'] for record in subgraph_data if record.get('player_name')))
        if not all_players_in_team:
             logging.warning(f"No valid player names found for team '{team_name}'.")

        # 2. Build bipartite graph
        B, player_nodes, skill_nodes = self._build_bipartite_graph(subgraph_data)
        if not B or B.number_of_nodes() == 0:
             return f"Cannot generate profile: Bipartite graph not created or empty for '{team_name}' (check extracted data)."
        if B.number_of_edges() == 0:
             logging.warning(f"Bipartite graph for '{team_name}' contains no Player-Skill edges. Profile might be limited.")

        # 3. Run community detection
        valid_player_nodes_for_comm = {p_node for p_node in player_nodes if p_node in B}
        communities = self._run_bipartite_community_detection(B, valid_player_nodes_for_comm)

        # 4. Extract community details
        community_details = self._get_community_details(B, communities)

        # 5. Format context for LLM (Now in English)
        llm_context = self._format_context_for_llm(team_name, community_details, all_players_in_team)

        # 6. Generate profile with LLM (Using English prompt)
        final_profile = self._generate_profile_with_llm(llm_context)

        logging.info(f"Profile for '{team_name}' completed.")
        return final_profile
    
    def _save_team_profile_to_kg(self, team_name: str, profile_text: str):
        """Saves the generated team profile text to the Team node in Neo4j."""
        if not profile_text:
             logging.warning(f"Skipping saving empty profile for team '{team_name}'.")
             return

        query = """
        MATCH (t:team {name: $team_name})
        SET t.profile_analysis_generated = $profile_text,
            t.profile_last_updated = timestamp() // Aggiungi timestamp
        """
        params = {"team_name": team_name, "profile_text": profile_text}
        try:
            with self._driver.session() as session:
                session.run(query, params)
            logging.info(f"Saved generated profile to Neo4j for team '{team_name}'.")
        except Exception as e:
            logging.error(f"Error saving team profile to Neo4j for '{team_name}': {e}")




if __name__ == "__main__":
    profiler = None # Inizializza a None

    neo4j_cred = readJson('neo4j_cred.json')

    uri = neo4j_cred['uri']
    user = neo4j_cred['username']
    password = neo4j_cred['password']

    profiler = TeamProfiler(uri, user, password)
    out_dict = {}
    for team_to_analyze in ['Milan', 'Inter']:
        profile = profiler.generate_team_profile(team_to_analyze)
        #profile = profiler.generate_profile_single_community(team_to_analyze)
        print("-" * 80)
        print(f"Profilo Generato per: {team_to_analyze}")
        print("-" * 80)
        out_dict[team_to_analyze] = profile
    writeJson(out_dict, 'Descriptions/graph_team_profile.json')


    profiler.close() # Assicurati di chiudere la connessione

