import json
import logging
import os
import random
from argparse import ArgumentParser
from time import perf_counter

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.constrained_decoding import Trie, constrained_decoding
from src.prompts import GEN_PSEUDO_GRAPH_PROMPT
from src.utils import DataUtils, llm_generate

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(message)s", datefmt="%d-%b-%y %H:%M:%S"
)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--specialized-model-path", type=str, required=True)
    parser.add_argument("--processed-data-path", type=str, required=True)
    parser.add_argument("--processed-trie-path", type=str, required=True)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=50)

    # Generation configs
    parser.add_argument("--generation-max-new-tokens", type=int, default=128)
    parser.add_argument("--generation-num-beams", type=int, default=5)
    parser.add_argument("--generation-num-beam-groups", type=int, default=5)
    parser.add_argument("--generation-diversity-penalty", type=float, default=1.0)
    parser.add_argument("--generation-early-stopping", action="store_true")
    parser.add_argument("--generation-num-return-sequences", type=int, default=5)

    # Constrained decoding
    parser.add_argument("--use-constrained-decoding", action="store_true")
    parser.add_argument("--start-entity-token", type=str, default="<entity>")
    parser.add_argument("--end-entity-token", type=str, default="</entity>")

    parser.add_argument("--output-folder-path", type=str, required=True)
    parser.add_argument("--description", type=str, default="")
    parser.add_argument("--save-steps", type=int, default=5)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    logger.info("Arguments\n%s", json.dumps(vars(args), indent=2))

    assert (
        args.generation_num_return_sequences
        == args.generation_num_beams
        == args.generation_num_beam_groups
    ), "Number of return sequences, beams, and beam groups should be the same"

    model_name = args.specialized_model_path.split("/")[-2]
    checkpoint_name = args.specialized_model_path.split("/")[-1]
    save_folder = os.path.join(
        args.output_folder_path, f"{model_name}_{checkpoint_name}"
    )
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    save_file_name = (
        f"{model_name}_{checkpoint_name}_num_beams_{args.generation_num_beams}"
        + ("" if args.use_constrained_decoding else "_without_constraint")
        + ".json"
    )
    save_file_path = os.path.join(save_folder, save_file_name)

    logger.info("Loading evaluation data")
    if os.path.exists(save_file_path):
        all_data = DataUtils.load_data(save_file_path)["data"]
        logger.info(f"Loaded cached data from {save_file_path}.")
    else:
        all_data = DataUtils.load_data(args.processed_data_path)

    logger.info("Loading specialized model")
    specialized_model = AutoModelForCausalLM.from_pretrained(
        args.specialized_model_path,
        device_map={"": f"cuda:{args.gpu_id}"},
        torch_dtype=(
            torch.bfloat16 if args.generation_num_beams == 1 else torch.float32
        ),
    )
    specialized_model.eval()
    specialized_tokenizer = AutoTokenizer.from_pretrained(args.specialized_model_path)

    generation_config = {
        "do_sample": False,
        "temperature": None,
        "top_k": None,
        "top_p": None,
        "max_new_tokens": args.generation_max_new_tokens,
        "num_beams": args.generation_num_beams,
        "num_beam_groups": args.generation_num_beam_groups,
        "diversity_penalty": args.generation_diversity_penalty,
        "early_stopping": args.generation_early_stopping,
        "num_return_sequences": args.generation_num_return_sequences,
    }

    if args.generation_num_return_sequences == 1:
        del generation_config["num_return_sequences"]
        del generation_config["num_beam_groups"]
        del generation_config["diversity_penalty"]

    prefix_allowed_tokens_fn = None
    if args.use_constrained_decoding:
        logger.info("Loading processed trie")
        trie = Trie.load(args.processed_trie_path)
        prefix_allowed_tokens_fn = constrained_decoding(
            specialized_tokenizer,
            trie,
            start_entity_token=args.start_entity_token,
            end_entity_token=args.end_entity_token,
        )

    for partition in all_data:
        random.seed(42)
        all_data[partition] = random.sample(
            all_data[partition], min(args.num_samples, len(all_data[partition]))
        )

    logger.info("Generating pseudo subgraphs")
    for partition, samples in all_data.items():
        for i, sample in tqdm(
            enumerate(samples),
            desc=f"Generating pseudo subgraphs for {partition}",
            total=len(samples),
        ):
            if "pseudo_subgraphs" in sample:
                continue
            try:
                prompt = GEN_PSEUDO_GRAPH_PROMPT.replace("{{claim}}", sample["claim"])
                start_time = perf_counter()
                pseudo_subgraphs = llm_generate(
                    prompt,
                    specialized_model,
                    specialized_tokenizer,
                    generation_config=generation_config,
                    prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                )
                sample["pseudo_subgraphs"] = pseudo_subgraphs
                sample["runtime"] = {
                    "start_time": start_time,
                    "end_time": perf_counter(),
                }
            except Exception as e:
                logger.error(f"Error generating pseudo subgraphs for {sample}")
                logger.error(e)
                logger.error("Skipping sample")

            if (i + 1) % args.save_steps == 0:
                with open(save_file_path, "w") as f:
                    data_to_save = {
                        "args": vars(args),
                        "data": all_data,
                    }
                    json.dump(data_to_save, f, ensure_ascii=False, indent=2)

    # Save data
    with open(save_file_path, "w") as f:
        data_to_save = {
            "args": vars(args),
            "data": all_data,
        }
        json.dump(data_to_save, f, ensure_ascii=False, indent=2)
