export PYTHONPATH="$(pwd)":$PYTHONPATH
python workflow/pipeline/retrieve_subgraphs_without_disambiguation.py \
  --algorithm-top-k-unknown-relations 3 \
  --algorithm-top-k-unknown-each-connected-node 3 \
  --algorithm-top-k-complete-relations 1 \
  --input-file-path /home/namb/hoangpv4/kg_fact_checking/data/output/pseudo_subgraphs/specialized_llm_3b_base_5000_checkpoint-157/specialized_llm_3b_base_5000_checkpoint-157_num_beams_5.json \
  --output-folder /home/namb/hoangpv4/kg_fact_checking/data/output/retrieved_subgraphs_without_disambiguation/specialized_llm_3b_base_5000_checkpoint-157 \
  --nx-graph-path /home/namb/hoangpv4/kg_fact_checking/data/processed_factkg/dbpedia_2015_undirected_light.gpkl \
  --retrieval-model-path /raid/HUB_LLM/bge-large-en-v1.5\