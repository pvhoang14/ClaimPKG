#!/bin/bash
export PYTHONPATH="$(pwd)":$PYTHONPATH
poetry run python workflow/pipeline/generate_pseudo_subgraphs.py \
  --specialized-model-path /raid/hoangpv4/models/specialized_llm_3b_base_5000/checkpoint-157 \
  --processed-data-path /home/namb/hoangpv4/kg_fact_checking/data/processed_factkg/partitioned_500/factkg_test.json \
  --data-partition 'num1' \
  --processed-trie-path /home/namb/hoangpv4/kg_fact_checking/data/indexed_trie/entity_trie.pkl \
  --gpu-id 0 \
  --generation-max-new-tokens 128 \
  --generation-num-beams 5 \
  --generation-num-beam-groups 5 \
  --generation-diversity-penalty 1.0 \
  --generation-num-return-sequences 5 \
  --output-folder-path /home/namb/hoangpv4/kg_fact_checking/data/output/pseudo_subgraphs \
  --generation-early-stopping  \
  --use-constrained-decoding  \