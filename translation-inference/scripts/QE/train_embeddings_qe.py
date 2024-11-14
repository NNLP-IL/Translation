import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn import metrics
import numpy as np
from tqdm import tqdm
import datetime
import optuna

from model import FCNN
from scripts.utils.print_colors import *
from scripts.utils.general_utils import load_data, Write2Streams

import openai
from sentence_transformers import SentenceTransformer
local_embeddings_model = SentenceTransformer('intfloat/multilingual-e5-large')


def get_openai_embedding(openai_client: openai.OpenAI, text: str, model: str = "text-embedding-ada-002"):
   text = text.replace("\n", " ")
   return openai_client.embeddings.create(input=[text], model=model).data[0].embedding


def get_embedding_local(text: str):
   text = text.replace("\n", " ")
   input_text = f"query: {text}"
   return local_embeddings_model.encode([input_text], normalize_embeddings=True)


def generate_run_id():
    now = datetime.datetime.now()
    run_id = now.strftime("%Y%m%d_%H%M%S")
    return run_id


class EmbeddingsDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]


def split_data(embeddings: torch.tensor, labels: torch.tensor, validation_split: float = 0.2, batch_size: int = 100):
    # Create custom dataset
    dataset = EmbeddingsDataset(embeddings, labels)

    # shuffle and choose indices
    shuffle_dataset = True
    dataset_size = len(embeddings)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset:
        # np.random.seed(42)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    from torch.utils.data.sampler import SubsetRandomSampler
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    # Create dataloaders from the datasets
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    val_loader = DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler)

    print('Number of training samples: {}'.format(len(train_indices)))
    print('Number of validation samples: {}'.format(len(val_indices)))

    return train_loader, val_loader


def save_model(model, opt, epoch_idx: int, output_path: str, model_name: str):
    os.makedirs(output_path, exist_ok=True)
    # Save the trained model
    MODEL_PATH = os.path.join(output_path, model_name)
    torch.save({
        'epoch': epoch_idx,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': opt.state_dict()
    }, MODEL_PATH)
    print(f'Model saved to {MODEL_PATH}')


def train_objective(trial, features: torch.tensor, labels: torch.tensor, save_path: str,
                    num_epochs: int = 50, saving_interval: int = 25, classification_threshold: float = 0.5,
                    val_split: float = 0.2):

    # Hyperparameter search space
    hidden_dims = [
        trial.suggest_int('hidden_dim1', 512, 2048),
        trial.suggest_int('hidden_dim2', 128, 512),
        trial.suggest_int('hidden_dim3', 32, 128)
    ]
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])

    print(f"{PRINT_START}{RED}Hidden_dims: {hidden_dims}{PRINT_STOP}")
    print(f"{PRINT_START}{RED}Learning rate: {lr}{PRINT_STOP}")
    print(f"{PRINT_START}{RED}Batch size: {batch_size}{PRINT_STOP}")
    tmp_save_path = os.path.join(save_path, f"hidden_dims={','.join([str(dim) for dim in hidden_dims])}"
                                            f"@lr={lr}@batch_size={batch_size}")

    # Create data loaders
    train_loader, val_loader = split_data(features, labels, validation_split=val_split, batch_size=batch_size)

    # Initialize the model, loss function, and optimizer
    input_dim = features.shape[1]
    output_dim = 1
    model = FCNN(input_dim, hidden_dims, output_dim)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)

        train_loss = train_loss / len(train_loader.sampler)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {train_loss:.4f}')

        if ((epoch + 1) % saving_interval == 0) or (epoch + 1 == num_epochs):
            # Save the trained model
            save_model(model=model, opt=optimizer, epoch_idx= epoch + 1,
                       output_path=tmp_save_path, model_name=f'translation_quality_model@epoch{epoch+1}.pt')

    # Evaluate the model
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, targets in val_loader:
            outputs = model(inputs).squeeze()
            preds = (outputs >= classification_threshold).float()
            all_preds.extend(preds.numpy())
            all_labels.extend(targets.numpy())

    accuracy = metrics.accuracy_score(all_labels, all_preds)
    balanced_accuracy = metrics.balanced_accuracy_score(all_labels, all_preds)
    precision = metrics.precision_score(all_labels, all_preds)
    recall = metrics.recall_score(all_labels, all_preds)
    print(f'{PRINT_START}{BLUE}Validation Accuracy: {accuracy * 100:.2f}%{PRINT_STOP}')
    print(f'{PRINT_START}{BLUE}Validation Balanced Accuracy: {balanced_accuracy * 100:.2f}%{PRINT_STOP}')
    print(f'{PRINT_START}{BLUE}Validation Precision: {precision * 100:.2f}%{PRINT_STOP}')
    print(f'{PRINT_START}{BLUE}Validation Recall: {recall * 100:.2f}%{PRINT_STOP}')

    # return accuracy, precision, recall
    return balanced_accuracy, precision, recall


def main():

    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Train a QE Classifier")
    parser.add_argument('xsts_data_file', type=str, help="csv file containing all the necessary data -"
                                                    "src sentences, translations (Google), ranks from human annotator")
    parser.add_argument('--output_dir', type=str, help="The output dir to save the current run")
    parser.add_argument('--epochs', type=int, default=50, help="Number of epochs per training")
    parser.add_argument('--save_interval', type=int, default=25, help="saving interval")
    parser.add_argument('--val_split', type=float, default=0.2, help="The portion of the data that is"
                                                                     " used for validation")

    # Parse the arguments
    args = parser.parse_args()

    # Hyperparameters
    NUM_EPOCHS = args.epochs
    SAVE_INTERVAL = args.save_interval
    THRESHOLD = 0.5  # Threshold for positive prediction

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using {device}")

    project_dir = args.get('output_dir', 'runs')
    run_id = generate_run_id()
    save_path = os.path.join(project_dir, run_id)
    os.makedirs(save_path, exist_ok=True)

    # Save print statements to a log file
    log_file = open(os.path.join(save_path, "output.log"), "w")
    sys.stdout = Write2Streams(sys.stdout, log_file)

    # Extract the data
    xsts_df = load_data(args.xsts_data_file)
    src_sentences = xsts_df["src"]
    translations = xsts_df["google"]
    labels = xsts_df["google_rank1"]

    if os.path.exists('test_embeddings/src_local_embeddings.npy'):
        src_embeddings = np.load('test_embeddings/src_local_embeddings.npy')
        print("existing src embeddings has been loaded")
    else:
        # Generate embeddings
        src_embeddings = np.array([get_embedding_local(sentence) for sentence in tqdm(src_sentences)])
        np.save('test_embeddings/src_local_embeddings.npy', src_embeddings)
    if os.path.exists('test_embeddings/translation_local_embeddings.npy'):
        translation_embeddings = np.load('test_embeddings/translation_local_embeddings.npy')
        print("existing trans embeddings has been loaded")
    else:
        # Generate embeddings
        translation_embeddings = np.array([get_embedding_local(sentence) for sentence in tqdm(translations)])
        np.save('test_embeddings/translation_local_embeddings.npy', translation_embeddings)

    # Combine src and translation embeddings
    src_embeddings = src_embeddings.squeeze(1)
    translation_embeddings = translation_embeddings.squeeze(1)
    embeddings = np.concatenate((src_embeddings, translation_embeddings), axis=1)

    # Convert to PyTorch tensors
    embeddings = torch.tensor(embeddings).to(torch.float32)
    labels = torch.tensor(labels).to(torch.float32)

    # Create a study object and optimize the objective function
    study = optuna.create_study(directions=['maximize', 'maximize', 'maximize'])
    study.optimize(lambda trial: train_objective(trial, embeddings, labels, save_path,
                                                 num_epochs=NUM_EPOCHS, saving_interval=SAVE_INTERVAL,
                                                 classification_threshold=THRESHOLD, val_split=args.val_split),
                   n_trials=10)

    # Print the best hyperparameters
    print(f'{PRINT_START}{GREEN}Best trial:')
    for i, trial in enumerate(study.best_trials):
        print(f'Trial {i + 1}:')
        print('  Value:', trial.values)
        print('  Params:')
        for key, value in trial.params.items():
            print('    {}: {}'.format(key, value))

    print(PRINT_STOP)

    log_file.close()

if __name__ == "__main__":
    main()
