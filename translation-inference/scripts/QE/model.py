import numpy as np
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from comet import load_from_checkpoint

from scripts.utils.general_utils import load_config, load_pkl_model
from scripts.utils.print_colors import *
from scripts.utils.eval_utils import get_comet_qe_scores


class FCNN(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(FCNN, self).__init__()
        self.layers = nn.ModuleList()

        # Input layer
        self.layers.append(nn.Linear(input_dim, hidden_dims[0]))
        self.layers.append(nn.ReLU())

        # Hidden layers
        for in_dim, out_dim in zip(hidden_dims[:-1], hidden_dims[1:]):
            self.layers.append(nn.Linear(in_dim, out_dim))
            self.layers.append(nn.ReLU())

        # Output layer
        self.layers.append(nn.Linear(hidden_dims[-1], output_dim))
        self.layers.append(nn.Sigmoid())

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class TranslationQualityClassifier:

    QE_TYPE = 'TranslationQualityClassifier'

    def __init__(self, config="Configs/qe_config.yaml"):
        super(TranslationQualityClassifier, self).__init__()
        self.config = load_config(config)[self.QE_TYPE]
        self.local_embeddings_model = SentenceTransformer(self.config.get("embeddings_model", "intfloat/multilingual-e5-large"))
        self.input_dim = self.config.get("INPUT_DIM", 2048)
        self.hidden_dims = self.config["HIDDEN_DIMS"] # necessary in config file
        self.output_dim = self.config.get("OUTPUT_DIM", 1)
        self.threshold = self.config.get("THRESHOLD", 0.5)
        self.qe_model = self.load_qe_model()
        self.qe_model.eval()

    def get(self, attr, default=None):
        return getattr(self, attr, default)

    def load_qe_model(self):
        # load model
        model = FCNN(self.input_dim, self.hidden_dims, self.output_dim)
        qe_model_path = self.config.get("qe_model_path", "best_models/local_embeddings/hidden_dims=794,177,80@lr=0.0028575617445571523@batch_size=32/translation_quality_model@epoch50.pt")
        model.load_state_dict(torch.load(qe_model_path)['model_state_dict'])
        print(f"{PRINT_START}{BLUE}QE model is loaded{PRINT_STOP}")
        return model

    def get_embedding_local(self, text_batch: list):
       texts = [text.replace("\n", " ") for text in text_batch]
       input_texts = [f"query: {text}" for text in texts]
       return self.local_embeddings_model.encode(input_texts, normalize_embeddings=True)

    def get_quality_scores(self, src_batch: list, trans_batch: list, return_raw_output: bool = False):

        # send to the embedder
        src_embeddings = self.get_embedding_local(src_batch)
        translation_embeddings = self.get_embedding_local(trans_batch)

        # Combine src and translation embeddings
        embeddings = np.concatenate((src_embeddings, translation_embeddings), axis=1)

        # model prediction
        embeddings = torch.from_numpy(embeddings).to(torch.float32)
        model_outputs = self.qe_model(embeddings)
        model_outputs = model_outputs.detach().numpy()
        preds = (model_outputs > self.threshold).astype(np.int64)

        if return_raw_output:
            return model_outputs, preds
        return preds


class CometQE:

    QE_TYPE = 'CometQE'

    def __init__(self, config="Configs/qe_config.yaml"):
        self.config = load_config(config)[self.QE_TYPE]
        self.comet_qe_model = load_from_checkpoint(self.config.get("comet_qe_model_path"))
        self.model = load_pkl_model(self.config.get("qe_model_path"))

    def get(self, attr, default=None):
        return getattr(self, attr, default)

    def get_quality_scores(self, src_batch: list, trans_batch: list):
        comet_qe_scores = get_comet_qe_scores(comet_qe_model=self.comet_qe_model,
                                              src_sentences=src_batch, trans_sentences=trans_batch)
        preds = self.model.predict(np.array(comet_qe_scores))
        return preds

