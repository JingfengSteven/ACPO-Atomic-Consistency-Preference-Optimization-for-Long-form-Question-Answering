import os
import hashlib
import torch
from tqdm import tqdm
from transformers import pipeline
import json
import tqdm
import random
import pandas as pd
import shutil
import nltk
from nltk.tokenize import sent_tokenize
from transformers import AutoModel, AutoTokenizer
import torch
from sklearn.cluster import AgglomerativeClustering
from collections import defaultdict
import argparse
from transformers import BartForConditionalGeneration, BartTokenizer
from Step1_generator import TextGenerator
from Step2_atomic_fact import AtomicResponseProcessor

def combine_summarized_files(input_dir, output_file, num_files):
    combined_summaries = {}
    for i in range(1, num_files + 1):
        summarized_file = os.path.join(input_dir, f'final_summarized_responses_{i}.json')
    
        if os.path.exists(summarized_file):
            print(f"Processing {summarized_file}...")
            with open(summarized_file, 'r') as f:
                summarized_data = json.load(f)
            for question, summary in summarized_data.items():
                if question not in combined_summaries:
                    combined_summaries[question] = []
                    combined_summaries[question].append(summary[-1]['content'])
                else:
                    combined_summaries[question].append(summary[-1]['content'])
        else:
            print(f"File {summarized_file} does not exist. Skipping.")
            
    with open(output_file, 'w') as f:
        json.dump(combined_summaries, f, indent=4)

    print(f"Combined summaries saved to '{output_file}'.")


def process_responses_P_N(input_file1, output_file, total_num):
    with open(input_file1, 'r') as file:
        data = json.load(file)
    
    all_results = {}
    
    for question, responses in data.items():
        sorted_responses = sorted(responses, key=lambda x: (x[1], len(x[0])), reverse=False)
        preferred = []
        non_preferred = []

        for idx, (response, score) in enumerate(sorted_responses):
            if idx ==0:
                non_preferred.append([response, score])
            if  idx ==30:
                preferred.append([response, score])
        all_results[question] = {
            "preferred": preferred,
            "non_preferred": non_preferred
        }
    with open(output_file, 'w') as outfile:
        json.dump(all_results, outfile, indent=4)

    print(f"All preferred and non-preferred responses have been saved to {output_file}.")



def correct_dpo_format(file_name):
    with open(file_name, 'r') as file:
        data = json.load(file)

    formatted_data = []
    cnt = 0
    for prompt, response_clusters in tqdm.tqdm(data.items(), desc="Formatting DPO data"):
        num = 0

        preferred_responses = [response for response, _ in response_clusters.get('preferred', [])]
        non_preferred_responses = [response for response, _ in response_clusters.get('non_preferred', [])]
    
        if not preferred_responses or not non_preferred_responses:
            continue
    
        for preferred_response in preferred_responses:
            for non_preferred_response in non_preferred_responses:
                num += 1
                cnt += 1
                formatted_data.append({
                    "prompt": prompt,
                    "preferred": preferred_response.lstrip(),
                    "non_preferred": non_preferred_response.lstrip()
                })

        print(num)
    
    print(f"Total formatted pairs: {cnt}")
    return formatted_data
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate answers and manage cache with options.")
    parser.add_argument('--base_dir', type=str, help="Base directory path")
    parser.add_argument('--model_id', type=str, help="Model ID for text generation")
    
    parser.add_argument('--system_prompt_1', type=str, default="You are an intelligent assistant who answers questions accurately", help="System prompt for the model")


    
    parser.add_argument('--question_file', type=str, help="Path to the questions file")
    parser.add_argument('--num_answers_per_question', type=int, default=30, help="Number of answers to generate for each temperature")
    parser.add_argument('--max_new_tokens', type=int, default=1024, help="Max new tokens for the generation")
    parser.add_argument('--temperatures_1', nargs='+', type=float, default=[0.6], help="List of temperatures to use for generation")
    parser.add_argument('--clear_cache_1', action='store_true', help="Clear the cache for initial generation before generating answers")


    parser.add_argument('--DPO_data_type',type=str, default='P_N')
    parser.add_argument('--device', type=int, default=0,help="Device ID for GPU (-1 for CPU; 0 for first GPU, etc.)")

    args = parser.parse_args()
    
    generator = TextGenerator(
        cache_dir=args.base_dir+'/cache_initial_generation',
        model_id=args.model_id,
        system_prompt=args.system_prompt_1,
        max_new_tokens=args.max_new_tokens,  
        clear_cache=args.clear_cache_1
    )
    
    
    generator.process_questions(args.question_file, args.base_dir+'/generated_for_llama.json', num_answers=args.num_answers_per_question, temperatures=args.temperatures_1)
    
    processor = AtomicResponseProcessor()
    processor.process_responses(
        responses_file=args.base_dir+'/generated_for_llama.json',
        clustering_results_file=args.base_dir+'/clustering_results.json',
        response_contributions_file=args.base_dir+'/response_contributions_scores.json',
    )
 
    
    if(args.DPO_data_type=='P_N'):
        process_responses_P_N(args.base_dir+'/response_contributions_scores.json', args.base_dir+'/all_preferred_non_preferred_P_N.json',args.num_summary_generation)
        formatted_pairs=correct_dpo_format(args.base_dir+'/all_preferred_non_preferred_P_N.json')
        with open(args.base_dir+'/dpo_pairs_P_N.json', 'w') as outfile:
            json.dump(formatted_pairs, outfile, indent=4)
        print("Formatted DPO pairs have been saved to 'dpo_pairs.json'.")
   
