export PYTHONPATH="$(pwd)":$PYTHONPATH
python workflow/pipeline/llm_reasoning.py     \
  --input-file-path /home/namb/hoangpv4/kg_fact_checking/data/output/retrieved_subgraphs_without_disambiguation/specialized_llm_3b_base_5000_checkpoint-157/specialized_llm_3b_base_5000_checkpoint-157_num_beams_5_retrieved_embedding_-6099473178803679162.json     \
  --output-folder /home/namb/hoangpv4/kg_fact_checking/data/output/reasoning_results_without_disambiguation     \
  --num-workers 30     \
  --vllm-server-host http://localhost:8264     \
  --model-name Llama-3.3-70B-Instruct\
