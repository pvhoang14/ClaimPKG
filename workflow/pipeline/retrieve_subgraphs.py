import json
import logging
import os
import pickle
from argparse import ArgumentParser
from time import perf_counter

from tqdm import tqdm

from src.retrieval_algorithm import GraphRetrieval

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = ArgumentParser()
    # algorithm arguments
    parser.add_argument("--algorithm-top-k-unknown-relations", type=int, default=1)
    parser.add_argument(
        "--algorithm-top-k-unknown-each-connected-node", type=int, default=1
    )
    parser.add_argument("--algorithm-top-k-complete-relations", type=int, default=1)
    parser.add_argument(
        "--scoring-method",
        type=str,
        default="embedding",
        choices=["embedding", "rerank", "fuzzy", "exact"],
    )

    # input arguments
    parser.add_argument("--input-file-path", type=str, required=True)
    parser.add_argument("--output-folder", type=str, required=True)
    parser.add_argument("--nx-graph-path", type=str, required=True)
    parser.add_argument("--retrieval-model-path", type=str, required=False)
    parser.add_argument("--rerank-model-path", type=str, default=None, required=False)
    parser.add_argument("--retrieval-model-gpu-id", type=int, default=0)
    parser.add_argument("--save-step", type=int, default=5)
    parser.add_argument("--force-retrieve", action="store_true")
    return parser.parse_args()


def retrieval_task(sample, retrieve_function):
    if "retrieved_triplets" in sample and sample["retrieved_triplets"] is not None:
        return sample["retrieved_triplets"]
    retrieved_triplets = None
    try:
        retrieved_triplets = retrieve_function(sample["pseudo_subgraphs"])
    except Exception as e:
        logger.error(f"Error: {e}")
    return retrieved_triplets


if __name__ == "__main__":
    args = parse_args()
    logger.info("Arguments:")
    logger.info(json.dumps(vars(args), indent=2))

    save_file_name = args.input_file_path.split("/")[-1].replace(
        ".json",
        f"_retrieved_{args.scoring_method}_{hash(frozenset(sorted(vars(args).items())))}.json",
    )
    save_file_path = os.path.join(args.output_folder, save_file_name)

    logger.info("Loading data...")
    if os.path.exists(save_file_path) and not args.force_retrieve:
        logger.info(f"File {save_file_path} already exists. Loading file...")
        with open(save_file_path, "r") as f:
            data = json.load(f)
            config = data["args"]
            data = data["data"]
    else:
        with open(args.input_file_path, "r") as f:
            data = json.load(f)
            config = data["args"]
            data = data["data"]

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    logger.info("Loading nx graphs...")
    with open(args.nx_graph_path, "rb") as f:
        nx_graph = pickle.load(f)

    retriever = GraphRetrieval(
        nx_graph=nx_graph,
        embedding_model_path=args.retrieval_model_path,
        reranker_model_path=args.rerank_model_path,
        top_k_unknown_relations=args.algorithm_top_k_unknown_relations,
        top_k_unknown_each_connected_node=args.algorithm_top_k_unknown_each_connected_node,
        top_k_complete_relations=args.algorithm_top_k_complete_relations,
        scoring_method=args.scoring_method,
    )
    logger.info("Retrieving subgraphs...")
    for key, samples in data.items():
        logger.info(f"Processing partition: {key}")
        for i, sample in enumerate(tqdm(samples)):
            try:
                if "pseudo_subgraphs" not in sample:
                    logger.error(
                        f"Sample {i} in partition {key} does not have pseudo_subgraphs"
                    )
                    continue

                if (
                    "retrieved_triplets" not in sample
                    or sample["retrieved_triplets"] is None
                ):
                    start_time = perf_counter()
                    sample["retrieved_triplets"] = retriever.retrieve(
                        sample["pseudo_subgraphs"]
                    )
                    end_time = perf_counter()
                    sample["runtime"] = {
                        "start_time": start_time,
                        "end_time": end_time,
                    }
            except Exception as e:
                logger.error(f"Error: {e}")
            if (i + 1) % args.save_step == 0:
                with open(save_file_path, "w") as f:
                    data_to_save = {
                        "pseudo_subgraphs_args": config,
                        "retrieve_subgraphs_args": vars(args),
                        "data": data,
                    }
                    json.dump(data_to_save, f, indent=2)
    logger.info(f"Saved file: {save_file_path}")
