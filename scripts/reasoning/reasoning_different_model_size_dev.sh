export PYTHONPATH="$(pwd)":$PYTHONPATH
python workflow/pipeline/llm_reasoning.py     --input-file-path /home/namb/hoangpv4/kg_fact_checking/data/output/retrieved_subgraphs_dev/specialized_llm_8b_base_5000_checkpoint-313/specialized_llm_8b_base_5000_checkpoint-313_num_beams_5_retrieved_-5442313132024497241.json     --output-folder /home/namb/hoangpv4/kg_fact_checking/data/output/reasoning_results_dev     --num-workers 20     --vllm-server-host http://localhost:8264     --model-name Llama-3.3-70B-Instruct

export PYTHONPATH="$(pwd)":$PYTHONPATH
python workflow/pipeline/llm_reasoning.py     --input-file-path /home/namb/hoangpv4/kg_fact_checking/data/output/retrieved_subgraphs_dev/specialized_llm_8b_base_2000_checkpoint-250/specialized_llm_8b_base_2000_checkpoint-250_num_beams_5_retrieved_-1652659487431273820.json     --output-folder /home/namb/hoangpv4/kg_fact_checking/data/output/reasoning_results_dev     --num-workers 20     --vllm-server-host http://localhost:8264     --model-name Llama-3.3-70B-Instruct

export PYTHONPATH="$(pwd)":$PYTHONPATH
python workflow/pipeline/llm_reasoning.py     --input-file-path /home/namb/hoangpv4/kg_fact_checking/data/output/retrieved_subgraphs_dev/specialized_llm_1b_base_2000_checkpoint-125/specialized_llm_1b_base_2000_checkpoint-125_num_beams_5_retrieved_-4332489110700970659.json     --output-folder /home/namb/hoangpv4/kg_fact_checking/data/output/reasoning_results_dev     --num-workers 20     --vllm-server-host http://localhost:8264     --model-name Llama-3.3-70B-Instruct

export PYTHONPATH="$(pwd)":$PYTHONPATH
python workflow/pipeline/llm_reasoning.py     --input-file-path /home/namb/hoangpv4/kg_fact_checking/data/output/retrieved_subgraphs_dev/specialized_llm_3b_base_2000_checkpoint-125/specialized_llm_3b_base_2000_checkpoint-125_num_beams_5_retrieved_5830945564406735299.json     --output-folder /home/namb/hoangpv4/kg_fact_checking/data/output/reasoning_results_dev     --num-workers 20     --vllm-server-host http://localhost:8264     --model-name Llama-3.3-70B-Instruct

export PYTHONPATH="$(pwd)":$PYTHONPATH
python workflow/pipeline/llm_reasoning.py     --input-file-path /home/namb/hoangpv4/kg_fact_checking/data/output/retrieved_subgraphs_dev/specialized_llm_3b_base_5000_checkpoint-157/specialized_llm_3b_base_5000_checkpoint-157_num_beams_5_retrieved_-3500189888326455151.json     --output-folder /home/namb/hoangpv4/kg_fact_checking/data/output/reasoning_results_dev     --num-workers 20     --vllm-server-host http://localhost:8264     --model-name Llama-3.3-70B-Instruct

export PYTHONPATH="$(pwd)":$PYTHONPATH
python workflow/pipeline/llm_reasoning.py     --input-file-path /home/namb/hoangpv4/kg_fact_checking/data/output/retrieved_subgraphs_dev/specialized_llm_1b_base_5000_checkpoint-157/specialized_llm_1b_base_5000_checkpoint-157_num_beams_5_retrieved_-6387170392399290855.json     --output-folder /home/namb/hoangpv4/kg_fact_checking/data/output/reasoning_results_dev     --num-workers 20     --vllm-server-host http://localhost:8264     --model-name Llama-3.3-70B-Instruct

export PYTHONPATH="$(pwd)":$PYTHONPATH
python workflow/pipeline/llm_reasoning.py     --input-file-path /home/namb/hoangpv4/kg_fact_checking/data/output/retrieved_subgraphs_dev/specialized_llm_1b_base_5000_checkpoint-157/specialized_llm_1b_base_5000_checkpoint-157_num_beams_5_retrieved_-6127153373052931643.json     --output-folder /home/namb/hoangpv4/kg_fact_checking/data/output/reasoning_results_dev     --num-workers 20     --vllm-server-host http://localhost:8264     --model-name Llama-3.3-70B-Instruct