import pickle
import sys
from argparse import ArgumentParser

import networkx as nx
from tqdm.auto import tqdm


def convert_kg_to_nx_kg(kg_path):
    print("Loading kg...")
    with open(kg_path, "rb") as f:
        kg = pickle.load(f)

    G = nx.Graph()
    for source, relations in tqdm(kg.items(), desc="Converting KG to NX KG"):
        if not G.has_node(source):
            G.add_node(source)
        for relation, target_entities in relations.items():
            reversed_relation = (
                ("~" + relation.strip())
                if "~" not in relation
                else relation.replace("~", "").strip()
            )
            for target in target_entities:
                if not G.has_node(target):
                    G.add_node(target)
                if not G.has_edge(source, target):
                    G.add_edge(source, target, relations=[])
                relation_list = [a["relation"] for a in G[source][target]["relations"]]
                can_add = True
                for r in [relation, reversed_relation]:
                    if r in relation_list:
                        can_add = False
                        break
                if not can_add:
                    continue
                G[source][target]["relations"].append(
                    {"source": source, "target": target, "relation": relation}
                )
    return G


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--kg-path", type=str, required=True, help="Path to the knowledge graph"
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Path to save the converted knowledge graph",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    G = convert_kg_to_nx_kg(args.kg_path)
    original_limit = sys.getrecursionlimit()
    try:
        sys.setrecursionlimit(100000)
        with open(args.output_path, "wb") as f:
            pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)
    finally:
        sys.setrecursionlimit(original_limit)
