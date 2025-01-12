import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
import argparse
import torch

class PlayerEmbeddingFunction(EmbeddingFunction):
    def __init__(self, model_path: str, use_gpu: bool = True):
        # Inizializza il modello fine-tunato
        self.model = SentenceTransformer(model_path)
        if use_gpu and torch.cuda.is_available():
            self.model = self.model.to("cuda")
        elif use_gpu:
             print("GPU richiesta, ma non disponibile. Il modello verrà eseguito su CPU.")

    def __call__(self, input: Documents) -> Embeddings:
        # Calcola gli embeddings usando il modello
        embeddings = self.model.encode(input, device="cuda" if torch.cuda.is_available() else "cpu").tolist()  # Converte gli embeddings in formato lista
        return embeddings

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Store player embeddings into chroma db")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Path to the directory containing the dataset file.")
    parser.add_argument("--model_dir", type=str, required=True, help="Path to the directory where the model is stored.")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to the directory where the chromadb is stored.")
    args = parser.parse_args()

    #"Model_v2/Model"
    embedding_function = PlayerEmbeddingFunction(model_path=args.model_dir)

    client = chromadb.PersistentClient(path=args.output_dir)

    collection_name = "team_embeddings"
    # Crea una collection usando la funzione di embedding personalizzata
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_function
    )

    nome_file = 'team2vec_dataset.json'
    dataset = load_dataset('json', data_files=f'{args.dataset_dir}/{nome_file}')
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