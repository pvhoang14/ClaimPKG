import logging
from collections import defaultdict

import torch
from fuzzywuzzy import fuzz
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

logger = logging.getLogger(__name__)

SCORING_METHODS = ["embedding", "rerank", "fuzzy", "exact"]


class ScoringUtils:
    def __init__(self, embedding_model_path, reranker_model_path, scoring_method):
        self.reranker_model, self.embedding_model = None, None
        self.reranker_tokenizer, self.embedding_tokenizer = None, None
        if scoring_method not in SCORING_METHODS:
            raise ValueError(f"Scoring method must be one of {SCORING_METHODS}")
        self.scoring_method = scoring_method

        if scoring_method == "embedding":
            if embedding_model_path is None:
                raise ValueError("Embedding model path must be provided.")
            self.embedding_model = AutoModel.from_pretrained(
                embedding_model_path, device_map={"": "cuda:0"}
            )
            self.embedding_model.eval()
            self.embedding_tokenizer = AutoTokenizer.from_pretrained(
                embedding_model_path
            )
        elif scoring_method == "rerank":
            if reranker_model_path is None:
                raise ValueError("Reranker model path must be provided.")
            self.reranker_model = AutoModelForSequenceClassification.from_pretrained(
                reranker_model_path, device_map={"": "cuda:0"}
            )
            self.reranker_model.eval()
            self.reranker_tokenizer = AutoTokenizer.from_pretrained(reranker_model_path)

    def get_most_similar_by_embeddings(self, query, sentences):
        def get_embeddings_from_model(texts):
            inputs = self.embedding_tokenizer(
                texts, padding=True, truncation=True, return_tensors="pt"
            ).to(self.embedding_model.device)
            with torch.no_grad():
                outputs = self.embedding_model(**inputs)
                sentence_embeddings = outputs[0][:, 0]
                sentence_embeddings = torch.nn.functional.normalize(
                    sentence_embeddings, p=2, dim=1
                )
            return sentence_embeddings

        query_embedding = get_embeddings_from_model([query])
        sentence_embeddings = get_embeddings_from_model(sentences)
        similarity = torch.nn.functional.cosine_similarity(
            query_embedding, sentence_embeddings
        )
        sorted_similarity, sorted_indices = similarity.sort(descending=True)
        sorted_scores = sorted_similarity.tolist()
        sorted_indices = sorted_indices.tolist()
        sorted_sentences = [sentences[i] for i in sorted_indices]
        return [
            {"sentence": sentence, "score": score, "index": i}
            for (sentence, score, i) in zip(
                sorted_sentences, sorted_scores, sorted_indices
            )
        ]

    def get_most_similar_by_rerank(self, query, sentences):
        pairs = [(query, sentence) for sentence in sentences]
        inputs = self.reranker_tokenizer(
            pairs, padding=True, truncation=True, return_tensors="pt", max_length=32
        ).to(self.reranker_model.device)
        with torch.no_grad():
            outputs = self.reranker_model(**inputs)
            logits = outputs.logits
            logits = torch.nn.functional.sigmoid(logits)

        num_labels = self.reranker_model.config.num_labels
        if num_labels == 1:
            scores = logits.squeeze(-1).tolist()
        else:
            scores = logits[:, 1].tolist()

        scored_sentences = [
            {"sentence": sentences[i], "score": score, "index": i}
            for i, score in enumerate(scores)
        ]
        scored_sentences.sort(key=lambda x: (-x["score"], x["index"]))
        return scored_sentences

    def get_most_similar_by_fuzzy(self, query, sentences):
        scores = []
        for idx, sentence in enumerate(sentences):
            score = fuzz.token_set_ratio(query, sentence) / 100.0
            scores.append((sentence, score, idx))

        scores.sort(key=lambda x: (-x[1], x[2]))
        return [{"sentence": s, "score": sc, "index": idx} for s, sc, idx in scores]

    def get_most_similar_by_exact(self, query, sentences):
        scored_sentences = []
        for idx, sentence in enumerate(sentences):
            score = 1.0 if sentence == query else 0.0
            scored_sentences.append(
                {"sentence": sentence, "score": score, "index": idx}
            )

        scored_sentences.sort(key=lambda x: (-x["score"], x["index"]))
        return scored_sentences

    def get_most_similar(self, query, sentences):
        if self.scoring_method == "embedding":
            return self.get_most_similar_by_embeddings(query, sentences)
        elif self.scoring_method == "rerank":
            return self.get_most_similar_by_rerank(query, sentences)
        elif self.scoring_method == "fuzzy":
            return self.get_most_similar_by_fuzzy(query, sentences)
        elif self.scoring_method == "exact":
            return self.get_most_similar_by_exact(query, sentences)
        else:
            raise ValueError(f"Unsupported scoring method: {self.scoring_method}")


class GraphUtils:
    @staticmethod
    def deduplicate_triplets(triplets):
        taken = set()
        returned_triplets = []
        for triplet in triplets:
            head, relation, tail = triplet
            if "~" in relation:
                key = (tail, head)
            else:
                key = (head, tail)
            if key not in taken:
                returned_triplets.append(triplet)
                taken.add(key)
        return returned_triplets

    @staticmethod
    def parse_subgraph(
        subgraph,
        start_entity_token="<entity>",
        end_entity_token="</entity>",
        separator="||",
        unknown_prefix="unknown",
    ):
        triplets = [
            tuple(
                entity.replace(start_entity_token, "")
                .replace(end_entity_token, "")
                .split(separator)
            )
            for entity in subgraph.strip().split("\n")
        ]
        triplets = [triplet for triplet in triplets if len(triplet) == 3]
        triplets = [
            triplet
            for triplet in triplets
            if not all(unknown_prefix in node for node in [triplet[0], triplet[2]])
        ]
        return GraphUtils.deduplicate_triplets(triplets)

    @staticmethod
    def get_relation_by_source(nx_graph, source):
        relation_dict = defaultdict(list)
        for target in nx_graph[source]:
            for relation in nx_graph[source][target]["relations"]:
                r_source, r_target = relation["source"], relation["target"]
                relation = {
                    (r_source, r_target): relation["relation"],
                    (r_target, r_source): (
                        relation["relation"].replace("~", "")
                        if "~" in relation["relation"]
                        else "~" + relation["relation"]
                    ),
                }[(source, target)]
                relation_dict[relation].append(target)
        return relation_dict


class GraphRetrieval:
    def __init__(
        self,
        nx_graph,
        embedding_model_path=None,
        reranker_model_path=None,
        unknown_prefix="unknown",
        top_k_unknown_relations=1,
        top_k_unknown_each_connected_node=1,
        top_k_complete_relations=1,
        scoring_method="embedding",
    ):
        self.nx_graph = nx_graph
        logger.info("Loading retrieval model...")
        self.scoring_utils = ScoringUtils(
            embedding_model_path, reranker_model_path, scoring_method
        )

        self.unknown_prefix = unknown_prefix
        self.top_k_unknown_relations = top_k_unknown_relations
        self.top_k_unknown_each_connected_node = top_k_unknown_each_connected_node
        self.top_k_complete_relations = top_k_complete_relations
        self.scoring_method = scoring_method

    def _retrieve_for_complete_triplets(self, triplets):
        for triplet in triplets:
            if any(self.unknown_prefix in node for node in triplet):
                print(f"A triplet contains unknown prefix.\n{triplet}")
                return []
        relevant_triplets = []
        for triplet in triplets:
            head, query_relation, tail = triplet
            if head not in self.nx_graph or tail not in self.nx_graph:
                continue
            if self.nx_graph.has_edge(head, tail):
                best_relations = set()
                for relation in self.nx_graph[head][tail]["relations"]:
                    is_source = (
                        relation["source"] == head and relation["target"] == tail
                    )
                    rel = relation["relation"]
                    best_relations.add(
                        rel
                        if is_source
                        else (rel.replace("~", "") if "~" in rel else f"~{rel}")
                    )
                scores = self.scoring_utils.get_most_similar(
                    query_relation, list(best_relations)
                )
                best_relations = {s["sentence"]: s["score"] for s in scores}
                best_relations = sorted(
                    best_relations.items(), key=lambda x: x[1], reverse=True
                )[: self.top_k_complete_relations]
                for best_relation, _ in best_relations:
                    relevant_triplets.append((head, best_relation, tail))

        return relevant_triplets

    def retrieve(self, subgraphs):
        subgraph_triplets = [
            GraphUtils.parse_subgraph(subgraph) for subgraph in subgraphs
        ]
        all_relevant_triplets = []
        for triplets in subgraph_triplets:
            complete_triplets = []
            for triplet in triplets:
                if all(
                    self.unknown_prefix not in node for node in [triplet[0], triplet[2]]
                ):
                    complete_triplets.append(triplet)
            if complete_triplets:
                all_relevant_triplets.extend(
                    self._retrieve_for_complete_triplets(complete_triplets)
                )

        return GraphUtils.deduplicate_triplets(all_relevant_triplets)
