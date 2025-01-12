import chromadb
import utils
from chromadb_ingestion import PlayerEmbeddingFunction
import argparse


def getTotalVocab():
    dict_vocab = utils.readJson('action_translations.json')
    vocab = []
    for d in dict_vocab.values():
        vocab = vocab + list(d.values())
    vocab = set(vocab) - {''}
    x_text = {1: 'difesa', 2: 'trequarti difensiva', 3: 'centrocampo', 4: 'trequarti offensiva', 5: 'attacco'}
    y_text = {1: 'fascia destra', 2: 'centro destra', 3: 'centrale', 4: 'centro sinistra', 5: 'fascia sinistra'}
    field_vocab = [f'{x}, {y}' for x in x_text.values() for y in y_text.values()]
    total_vocab = [f'{x} {y}' for x in vocab for y in field_vocab]
    return total_vocab


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Store vocab embeddings into chroma db")
    parser.add_argument("--vector_store_dir", type=str, required=True, help="Path to the directory containing the dataset file.")
    parser.add_argument("--model_dir", type=str, required=True, help="Path to the directory where the model is stored.")
    args = parser.parse_args()

    client = chromadb.PersistentClient(path=args.vector_store_dir)
    model = PlayerEmbeddingFunction(args.model_dir)
    vocab_collection = client.get_or_create_collection(name="vocab", embedding_function=model)
    total_vocab = getTotalVocab()

    for i in range(len(total_vocab)):
        vocab_collection.add(ids=[str(i)], documents=[total_vocab[i]])