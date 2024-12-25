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
    args = parser.parse_args()

    # Load dataset
    data_files = f"{args.dataset_dir}/player2vec_dataset.json"
    dataset = load_dataset('json', data_files=data_files)

    # Initialize model
    model = SentenceTransformer("distiluse-base-multilingual-cased-v1", device="cuda" if torch.cuda.is_available() else "cpu")

    vocab_dict = readJson('action_translations.json')
    vocab = []
    for v in vocab_dict.values():
        for t in v.values():
            tokens = t.split()
            vocab = vocab + tokens
    vocab = list(set(vocab))
    model.tokenizer.add_tokens(vocab)
    model.resize_token_embeddings(len(model.tokenizer))

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
