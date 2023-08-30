from typing import Any
from transformers import AutoTokenizer, AutoModel
import numpy as np
import torch
import nltk

class Retriever:
    def __init__(self, model_name="BAAI/bge-large-en", device="cuda"):
        self.device = device
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.embeddings = None
        self.chunks = None

        self.load_model()
        self.load_embeddings()

    def load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device)

    def load_embeddings(self):
        data = np.load(f"retrieval/wiki_embeddings_{self.model_name.split('/')[-1]}.npy", allow_pickle=True).item()
        self.chunks = data["texts"]
        self.embeddings = data["embeddings"]

    def embed(self, texts):
        """Embeds a list of texts using the model."""
        encoded_chunks = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            model_output = self.model(**encoded_chunks.to(self.device))
            # Perform pooling. In this case, cls pooling.
            embeddings = model_output[0][:, 0]
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings.cpu().numpy()
    
    def __call__(self, texts: str, k: int = 1, threshold: float = 0.8) -> list:
        """Retrieves k texts from the dataset that are most similar to the input text.
        Args:
            texts (str): Text to retrieve similar texts for.
            k (int, optional): Maximum number of texts to retrieve.
            threshold (float, optional): Threshold for the cosine similarity.
        Returns:
            List of texts that are most similar to the input text, sorted by similarity.
        """
        # TODO: use faiss for faster retrieval
        if type(texts) == str:
            texts = [texts]
        embeddings = self.embed(texts)
        scores = np.dot(self.embeddings, embeddings.T)
        best_indices = np.argpartition(scores, -k, axis=0)[-k:]
        best_scores = scores[best_indices, np.arange(scores.shape[1])]
        best_indices = best_indices[best_scores > threshold]
        return self.chunks[best_indices]
    
    @staticmethod
    def trim_chunk(chunk):
        """Trims away sentence fragments from the beginning and end of a chunk."""
        chunk = chunk.strip()
        sents = nltk.sent_tokenize(chunk)

        punc_chars = ".!?"
        if sents[-1][-1] not in punc_chars:
            try:
                chunk = chunk.removesuffix(sents[-1])
            except ValueError:
                pass
        try:
            chunk = chunk.removeprefix(sents[0])
        except ValueError:
            pass
        return chunk.strip()
