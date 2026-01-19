import numpy as np
import requests
import torch

TEI_HOST = "http://localhost:8264"


class EmbeddingUtils:
    @staticmethod
    def get_embeddings_from_host(text_list, tei_host=TEI_HOST):
        url = f"{tei_host}/embed"
        headers = {"Content-Type": "application/json"}
        data = {"inputs": text_list}
        try:
            response = requests.post(url, headers=headers, json=data, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"Request error: {e}")
            return []

    @staticmethod
    def get_most_similar_from_host(text, list_refers, top_k=1, tei_host=TEI_HOST):
        text_embedding = np.array(
            EmbeddingUtils.get_embeddings_from_host([text], tei_host=tei_host)[0]
        )
        refer_embedding = np.array(
            EmbeddingUtils.get_embeddings_from_host(list_refers, tei_host=tei_host)
        )
        similarity = np.dot(refer_embedding, text_embedding)
        return [
            {"text": list_refers[idx], "score": round(similarity[idx].item(), 4)}
            for idx in np.argsort(similarity)[::-1][:top_k]
        ]

    @staticmethod
    def get_embeddings_from_model(model, tokenizer, text_list):
        inputs = tokenizer(
            text_list, padding=True, truncation=True, return_tensors="pt"
        ).to(model.device)
        with torch.no_grad():
            outputs = model(**inputs)
            sentence_embeddings = outputs[0][:, 0]
            sentence_embeddings = torch.nn.functional.normalize(
                sentence_embeddings, p=2, dim=1
            )
        return sentence_embeddings

    @staticmethod
    def get_most_similar_from_model(model, tokenizer, text, list_refers):
        text_embedding = EmbeddingUtils.get_embeddings_from_model(
            model, tokenizer, [text]
        )
        refer_embedding = EmbeddingUtils.get_embeddings_from_model(
            model, tokenizer, list_refers
        )
        similarity = torch.nn.functional.cosine_similarity(
            text_embedding, refer_embedding
        )
        sorted_similarity, sorted_indices = torch.sort(similarity, descending=True)
        sorted_scores = sorted_similarity.tolist()
        sorted_indices = sorted_indices.tolist()
        sorted_sentences = [list_refers[idx] for idx in sorted_indices]
        return [
            {"sentence": sentence, "score": score, "index": idx}
            for sentence, score, idx in zip(
                sorted_sentences, sorted_scores, sorted_indices
            )
        ]
