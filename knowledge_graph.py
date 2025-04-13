from neo4j import GraphDatabase
import os
from utils import readJson

class KnowledgeGraphBuilder:
    def __init__(self, dir, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.dir = dir
        
        self.initialize_nodes()
        self.create_relationships()

    def close(self):
        self.driver.close()

    def initialize_nodes(self):
        entity_files = [x for x in os.listdir(self.dir) if x.endswith('entities.json')]
        for file in entity_files:

            nodes = readJson(f'{self.dir}/{file}')['nodes']
            self.create_nodes(nodes)


    def create_nodes(self, nodes):
       
        with self.driver.session() as session:
            for node in nodes:
                label = node.pop('type')  # use 'type' as label
                query_template = f"MERGE (n:{label} {{id: $id}})   SET n += $props"
                node_id = node['id']
                session.run(query_template, id=node_id, props=node)
    
    def create_relationships(self):
        relationships = readJson(f'{self.dir}/relationships.json')['relationships']
        query_template = """
        MATCH (a {id: $from_id})
        MATCH (b {id: $to_id})
        MERGE (a)-[r:%s]->(b)
        """  # We insert rel type later with %s
        with self.driver.session() as session:
            for rel in relationships:
                rel_type = rel["type"].upper()  # Cypher relation type must be uppercase with underscores
                from_id = rel["from"]
                to_id = rel["to"]
                session.run(query_template % rel_type, from_id=from_id, to_id=to_id)



if __name__ == '__main__':

    neo4j_cred = readJson('neo4j_cred.json')

    uri = neo4j_cred['uri']
    user = neo4j_cred['username']
    password = neo4j_cred['password']
    kg_dir = 'Dataset/Knowledge_graph'

    kgb = KnowledgeGraphBuilder(kg_dir, uri, user, password)
