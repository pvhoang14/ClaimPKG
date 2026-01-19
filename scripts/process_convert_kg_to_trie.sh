# #!/bin/bash
# export PYTHONPATH="$(pwd)":$PYTHONPATH
# poetry run python workflow/data_process/process_convert_kg_to_trie.py \
#     --tokenizer-path /raid/hoangpv4/models/specialized_llm_3b_base_5000/checkpoint-157 \
#     --kg-path /home/namb/hoangpv4/kg_fact_checking/data/processed_factkg/dbpedia_2015_undirected_light.pkl \
#     --output-dir /home/namb/hoangpv4/kg_fact_checking/data/indexed_trie \
#     --end-entity-token "</entity>" \
#!/bin/bash
# export PYTHONPATH="$(pwd)":$PYTHONPATH
# poetry run python workflow/data_process/process_convert_kg_to_trie.py \
#     --tokenizer-path /raid/hoangpv4/models/specialized_llm_3b_qwen_base_5000/checkpoint-156 \
#     --kg-path /home/namb/hoangpv4/kg_fact_checking/data/processed_factkg/dbpedia_2015_undirected_light.pkl \
#     --output-dir /home/namb/hoangpv4/kg_fact_checking/data/indexed_trie_qwen \
#     --end-entity-token "</entity>" \

# export PYTHONPATH="$(pwd)":$PYTHONPATH
# poetry run python workflow/data_process/process_convert_kg_to_trie.py \
#     --tokenizer-path /raid/hoangpv4/models/specialized_llm_3b_base_5000/checkpoint-157 \
#     --kg-path /home/namb/hoangpv4/kg_fact_checking/data/processed_factkg/dbpedia_2015_undirected.pkl \
#     --output-dir /home/namb/hoangpv4/kg_fact_checking/data/indexed_trie_full_llama \
#     --end-entity-token "</entity>" \

export PYTHONPATH="$(pwd)":$PYTHONPATH
poetry run python workflow/data_process/process_convert_kg_to_trie.py \
    --tokenizer-path /raid/hoangpv4/models/specialized_llm_3b_qwen_base_5000/checkpoint-156 \
    --kg-path /home/namb/hoangpv4/kg_fact_checking/data/processed_factkg/dbpedia_2015_undirected.pkl \
    --output-dir /home/namb/hoangpv4/kg_fact_checking/data/indexed_trie_full_qwen \
    --end-entity-token "</entity>" \