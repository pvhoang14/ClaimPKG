#!/bin/bash

export PYTHONPATH="$(pwd)":$PYTHONPATH
poetry run python workflow/data_process/process_convert_kg_to_nx_kg.py \
    --kg-path /home/namb/hoangpv4/kg_fact_checking/data/processed_factkg/dbpedia_2015_undirected.pkl \
    --output-path /home/namb/hoangpv4/kg_fact_checking/data/processed_factkg/dbpedia_2015_undirected.gpkl \
