import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from datasets import load_dataset

class PlayerEmbeddingFunction(EmbeddingFunction):
    def __init__(self, model_path: str):
        # Inizializza il modello fine-tunato
        self.model = SentenceTransformer(model_path)

    def __call__(self, input: Documents) -> Embeddings:
        # Calcola gli embeddings usando il modello
        embeddings = self.model.encode(input).tolist()  # Converte gli embeddings in formato lista
        return embeddings

if __name__ == '__main__':

    embedding_function = PlayerEmbeddingFunction(model_path="Model_v2/Model")

    client = chromadb.PersistentClient(path="VectorDB")

    # Crea una collection usando la funzione di embedding personalizzata
    collection = client.get_or_create_collection(
        name="player_embeddings",
        embedding_function=embedding_function
    )

    dataset = load_dataset('json', data_files='Dataset/Events2Text/player2vec_dataset.json')
    columns = list(dataset.column_names.values())[0]
    columns.remove('text')
    texts = dataset['train']['text']
    metadata = [{key: row[key] for key in columns} for row in dataset['train']]

    #fill missing data with their id
    for x in metadata:
        for c in columns:
            if x[c] == None:
                if c == 'playerName':
                    x[c] = x['playerId']
                if c == 'teamName':
                    x[c] = x['teamId']

    ids = [f'id{i+1}' for i in range(len(texts))]
    batch_size = 5000
    for i in range(0, len(texts), batch_size):
        print(f'records {i}-{ i + batch_size}/{len(texts)}')
        batch = texts[i:i + batch_size]
        metadata_batch = metadata[i: i + batch_size]
        ids_batch = ids[i:i + batch_size]
        collection.add(
            documents=batch,
            metadatas=metadata_batch,
            ids=ids_batch
        )

    print('Caricamento Vector DB effettuato con successo!')