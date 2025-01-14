import chromadb
import utils
from chromadb_ingestion import PlayerEmbeddingFunction
import argparse


def getTotalVocab(lang: str = 'en'):
    if lang not in ['it','en']:
        raise KeyError('language {lang} not available')
    dict_vocab = utils.readJson('action_translations.json')[lang]
    vocab = []
    for d in dict_vocab.values():
        vocab = vocab + list(d.values())
    vocab = set(vocab) - {''}
    if lang == 'it':
        x_text = {1: 'difesa', 2: 'trequarti difensiva', 3: 'centrocampo', 4: 'trequarti offensiva', 5: 'attacco'}
        y_text = {1: 'fascia destra', 2: 'centro destra', 3: 'centrale', 4: 'centro sinistra', 5: 'fascia sinistra'}
    else:
        x_text = {1: 'defensive', 2: 'defensive third', 3: 'midfield', 4: 'attacking third', 5: 'attack'}
        y_text = {1: 'right flank', 2: 'right center', 3: 'central', 4: 'left center', 5: 'left flank'}

    field_vocab = [f'{x}, {y}' for x in x_text.values() for y in y_text.values()]
    total_vocab = [f'{x} {y}' for x in vocab for y in field_vocab]
    return total_vocab


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Store vocab embeddings into chroma db")
    parser.add_argument("--vector_store_dir", type=str, required=True, help="Path to the directory containing the dataset file.")
    parser.add_argument("--model_dir", type=str, required=True, help="Path to the directory where the model is stored.")
    parser.add_argument("--lang", type=str, required=True, choices=["it", "en"], help="Language option. Choose between 'it' (Italian) or 'en' (English).")
    args = parser.parse_args()

    client = chromadb.PersistentClient(path=args.vector_store_dir)
    model = PlayerEmbeddingFunction(args.model_dir)
    vocab_collection = client.get_or_create_collection(name=f"vocab_{args.lang}", embedding_function=model)
    total_vocab = getTotalVocab(args.lang)

    for i in range(len(total_vocab)):
        vocab_collection.add(ids=[str(i)], documents=[total_vocab[i]])