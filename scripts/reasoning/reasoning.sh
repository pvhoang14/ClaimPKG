export PYTHONPATH="$(pwd)":$PYTHONPATH
python workflow/pipeline/llm_reasoning.py     --input-file-path /home/namb/hoangpv4/kg_fact_checking/data/output/retrieved_subgraphs/specialized_llm_3b_base_500_checkpoint-125/specialized_llm_3b_base_500_checkpoint-125_num_beams_5_retrieved_8396630994749157428.json     --output-folder /home/namb/hoangpv4/kg_fact_checking/data/output/reasoning_results     --num-workers 20     --vllm-server-host http://localhost:8264     --model-name Llama-3.3-70B-Instruct

export PYTHONPATH="$(pwd)":$PYTHONPATH
python workflow/pipeline/llm_reasoning.py     --input-file-path /home/namb/hoangpv4/kg_fact_checking/data/output/retrieved_subgraphs/specialized_llm_3b_base_2000_checkpoint-125/specialized_llm_3b_base_2000_checkpoint-125_num_beams_5_retrieved_-375077302557354718.json     --output-folder /home/namb/hoangpv4/kg_fact_checking/data/output/reasoning_results     --num-workers 20     --vllm-server-host http://localhost:8264     --model-name Llama-3.3-70B-Instruct

export PYTHONPATH="$(pwd)":$PYTHONPATH
python workflow/pipeline/llm_reasoning.py     --input-file-path /home/namb/hoangpv4/kg_fact_checking/data/output/retrieved_subgraphs/specialized_llm_3b_base_5000_checkpoint-157/specialized_llm_3b_base_5000_checkpoint-157_num_beams_5_retrieved_1631798178181607376.json     --output-folder /home/namb/hoangpv4/kg_fact_checking/data/output/reasoning_results     --num-workers 20     --vllm-server-host http://localhost:8264     --model-name Llama-3.3-70B-Instruct

export PYTHONPATH="$(pwd)":$PYTHONPATH
python workflow/pipeline/llm_reasoning.py     --input-file-path /home/namb/hoangpv4/kg_fact_checking/data/output/retrieved_subgraphs/specialized_llm_3b_base_100_checkpoint-100/specialized_llm_3b_base_100_checkpoint-100_num_beams_5_retrieved_-3794514166055747240.json     --output-folder /home/namb/hoangpv4/kg_fact_checking/data/output/reasoning_results     --num-workers 20     --vllm-server-host http://localhost:8264     --model-name Llama-3.3-70B-Instruct

export PYTHONPATH="$(pwd)":$PYTHONPATH
python workflow/pipeline/llm_reasoning.py     --input-file-path /home/namb/hoangpv4/kg_fact_checking/data/output/retrieved_subgraphs/specialized_llm_3b_base_10000_checkpoint-313/specialized_llm_3b_base_10000_checkpoint-313_num_beams_5_retrieved_7188999006684253892.json     --output-folder /home/namb/hoangpv4/kg_fact_checking/data/output/reasoning_results     --num-workers 20     --vllm-server-host http://localhost:8264     --model-name Llama-3.3-70B-Instruct