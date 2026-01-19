import json
import logging
import os
from argparse import ArgumentParser
from functools import partial

from src.openai_utils import get_completion as get_completion_openai
from src.utils import DataUtils, get_completion_vllm, multi_thread_task_dict

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
httpx_logger = logging.getLogger("httpx")
httpx_logger.setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


VERIFY_PROMPT = """
### Task:
Verify whether the fact in the given sentence is true or false based on the provided graph triplets. Use only the information in the triplets for verification.

- The triplets provided represent all relevant knowledge that can be retrieved.
- Interpret the "~" symbol in triplets as indicating a reverse relationship. For example:
  - "A ~loves B" means "B loves A".
  - "A ~south of B" means "B is north of A".
- The unit is not important. (e.g. "98400" is also same as 98.4kg)

### Response Format:
Provide your response in the following JSON format without any additional explanations:
{
    "rationale": "A concise explanation for your decision",
    "verdict": "true/false as the JSON value"
}

### Triplets:
{{triplets}}

### Claim:
{{claim}}
""".strip()


def task(sample, get_completion):
    llm_call = None
    try:
        triplet_text = "\n".join(
            str(tuple(triplet)) for triplet in sample["retrieved_triplets"]
        ).strip()
        if triplet_text == "":
            triplet_text = "There is no triplet evidence."
        prompt = VERIFY_PROMPT.replace("{{triplets}}", triplet_text).replace(
            "{{claim}}", sample["claim"]
        )
        llm_call = get_completion(prompt)
    except Exception as e:
        logger.error("Error in task: %s", e)
    finally:
        return llm_call

class Utils:
    @staticmethod
    def load_retrieved_data(path):
        with open(path, "r") as f:
            retrieved_data = json.load(f)
        return {
            "gen_subgraph_args": retrieved_data["pseudo_subgraphs_args"],
            "retrieved_subgraph_agrs": retrieved_data["retrieve_subgraphs_args"],
            "data": retrieved_data["data"],
        }

    @staticmethod
    def get_config(retrieved_data):
        retrieved_subgraph_agrs = retrieved_data["retrieved_subgraph_agrs"]
        gen_subgraph_args = retrieved_data["gen_subgraph_args"]

        k_unknown_relations = retrieved_subgraph_agrs[
            "algorithm_top_k_unknown_relations"
        ]
        k_unknown_each_connected_node = retrieved_subgraph_agrs[
            "algorithm_top_k_unknown_each_connected_node"
        ]
        k_complete_relations = retrieved_subgraph_agrs[
            "algorithm_top_k_complete_relations"
        ]
        scoring_method = retrieved_subgraph_agrs["scoring_method"]
        llm_size = None
        for size in ["llm_8b", "llm_7b", "llm_3b", "llm_1.5b", "llm_1b"]:
            if size in gen_subgraph_args["specialized_model_path"]:
                llm_size = size.split("_")[1]
                break

        num_samples = None
        for size in ["10000", "5000", "2000", "500", "100"]:
            if size in gen_subgraph_args["specialized_model_path"]:
                num_samples = int(size)
                break

        num_beams = None
        for size in ["beams_10", "beams_3", "beams_5", "beams_1"]:
            if size in retrieved_subgraph_agrs["input_file_path"]:
                num_beams = int(size.split("_")[1])
                break

        constraint = True
        if "without_constraint" in retrieved_subgraph_agrs["input_file_path"]:
            constraint = False

        return {
            "llm": llm_size,
            "training": num_samples,
            "beams": num_beams,
            "constraint": constraint,
            "unknown_relations": k_unknown_relations,
            "unknown_each_connected_node": k_unknown_each_connected_node,
            "complete_relations": k_complete_relations,
            "scoring_method": scoring_method,
        }

    @staticmethod
    def get_suffix_of_saved_file(config):
        suffix = (
            f"llm_{config['llm']}_training_{config['training']}_beams_{config['beams']}"
        )
        if config["constraint"]:
            suffix += "_with_constraint"
        else:
            suffix += "_without_constraint"
        suffix += f"_unknown_relations_{config['unknown_relations']}_unknown_each_connected_node_{config['unknown_each_connected_node']}_complete_relations_{config['complete_relations']}_scoring_method_{config['scoring_method']}"
        return suffix

    @staticmethod
    def get_completion_function(model_name, vllm_server_host):
        if "llama" in model_name.lower() or "qwen" in model_name.lower():
            return partial(
                get_completion_vllm,
                server_host=vllm_server_host,
                model=model_name,
            )
        elif "gpt" in model_name.lower():
            return get_completion_openai
        else:
            raise ValueError(f"Invalid model name: {model_name}")


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--input-file-path", type=str, required=True)
    parser.add_argument("--output-folder", type=str, required=True)
    parser.add_argument("--num-workers", type=int, required=True, default=10)
    parser.add_argument("--vllm-server-host", type=str, required=True)
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        choices=[
            "Llama-3.3-70B-Instruct",
            "gpt-4o-mini",
            "Llama-3.1-8B-Instruct",
            "Qwen2.5-7B-Instruct",
            "Qwen2.5-72B-Instruct",
        ],
    )
    parser.add_argument("--force-rerun", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    logger.info("Arguments: %s", json.dumps(vars(args), indent=2))
    retrieved_data = Utils.load_retrieved_data(args.input_file_path)
    config = Utils.get_config(retrieved_data)
    logger.info("Config: %s", json.dumps(config, indent=2))
    logger.info("Loading data...")
    data = retrieved_data["data"]
    completion_function = Utils.get_completion_function(
        args.model_name, args.vllm_server_host
    )

    # check if the output file already exists
    save_file_name = args.input_file_path.split("/")[-1].replace(".json", "")
    save_file_name += f"_{Utils.get_suffix_of_saved_file(config)}.json"
    save_path = os.path.join(args.output_folder, save_file_name)
    if os.path.exists(save_path):
        logger.warning("Output file already exists: %s", save_path)
        if args.force_rerun:
            logger.warning(
                "####################### Overwriting the file... ########################"
            )
        else:
            exit()

    # request completions for each sample
    for partition, samples in data.items():
        print(f"Processing partition: {partition}")
        task_dict = {}
        for sample in samples:
            task_dict.update(
                {
                    sample[
                        "id"
                    ]: lambda sample=sample, get_completion=completion_function: task(
                        sample, get_completion
                    )
                }
            )
        results = multi_thread_task_dict(task_dict, args.num_workers)
        for sample in samples:
            sample["llm_reasoning"] = results[sample["id"]]

    data = {"args": vars(args), "config": config, "data": data}
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)
    with open(save_path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info("Saved results to %s", save_path)
