import argparse
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, losses
import torch
from utils import readJson

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train a model using a dataset and save it to a specified directory.")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Path to the directory containing the dataset file.")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to the directory where the model will be saved.")
    parser.add_argument("--batch_size", type=int, default=30, help="Batch size for training.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs for training.")
    parser.add_argument("--lang", type=str, required=True, choices=["it", "en"], help="Language option. Choose between 'it' (Italian) or 'en' (English).")
    args = parser.parse_args()

    # Load dataset

    data_files = f"{args.dataset_dir}/player2vec_dataset_{args.lang}.json"
    dataset = load_dataset('json', data_files=data_files)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Initialize model
    model = SentenceTransformer("distiluse-base-multilingual-cased-v1", device=device)
    tokenizer = model.tokenizer
    huggingface_model = model._first_module().auto_model

    vocab_dict = readJson('action_translations.json')[args.lang]
    vocab = []
    for v in vocab_dict.values():
        for t in v.values():
            tokens = t.split()
            vocab = vocab + tokens
    vocab = list(set(vocab))
    num_added_tokens = tokenizer.add_tokens(vocab)
    huggingface_model.resize_token_embeddings(len(tokenizer))

    # Prepare training examples and DataLoader
    train_examples = dataset['train']['text']
    train_dataloader = losses.ContrastiveTensionDataLoader(train_examples, batch_size=args.batch_size, pos_neg_ratio=3)
    train_loss = losses.ContrastiveTensionLoss(model=model)

    # Train model
    model.fit(
        [(train_dataloader, train_loss)],
        epochs=args.epochs,
        output_path=args.output_dir,
        show_progress_bar=True
    )
