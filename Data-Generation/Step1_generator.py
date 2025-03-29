import os
import hashlib
import torch
from tqdm import tqdm
from transformers import pipeline
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import json
import pandas as pd
import shutil
class TextGenerator:
    def __init__(self, cache_dir, model_id, system_prompt, temperature=1.0, max_new_tokens=150, clear_cache=False,device=0):
        self.cache_dir = cache_dir
        self.model_id = model_id
        self.system_prompt = system_prompt
        self.temperature = temperature  
        self.max_new_tokens = max_new_tokens  
        self.device=device
        if clear_cache: 
            self.clear_cache()

        os.makedirs(self.cache_dir, exist_ok=True)
        self.generator = self.get_model_generator()

    def clear_cache(self):
        """Clears the cache directory."""
        if os.path.exists(self.cache_dir):
            shutil.rmtree(self.cache_dir)
        os.makedirs(self.cache_dir, exist_ok=True)
        print(f"Cache cleared at {self.cache_dir}")

    def get_model_generator(self):
        return pipeline(
            "text-generation",
            model=self.model_id,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
        )
        

    def compute_hash(self, question, temperature):
        key = f"{question}_{temperature}"
        return hashlib.sha256(key.encode('utf-8')).hexdigest()

    def load_from_cache(self, question, temperature):
        question_hash = self.compute_hash(question, temperature)
        cache_path = os.path.join(self.cache_dir, f"{question_hash}.json")
        if os.path.exists(cache_path):
            with open(cache_path, 'r') as f:
                return json.load(f)
        return None

    def save_to_cache(self, question, result, temperature):
        question_hash = self.compute_hash(question, temperature)
        cache_path = os.path.join(self.cache_dir, f"{question_hash}.json")
        with open(cache_path, 'w') as f:
            json.dump(result, f)

    def get_content(self, messages, num_return_sequences):
        answers = self.generator(
            messages,
            max_new_tokens=self.max_new_tokens,
            num_return_sequences=num_return_sequences,  
            do_sample=True,
            temperature=self.temperature,
        )
        last_characters = [answer['generated_text'][-1] for answer in answers]
        return last_characters
    


    def generate_answers_for_question(self, question, num_answers, temperature):
        cached_result = self.load_from_cache(question, temperature)
        if cached_result is not None:
            return cached_result
        
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Question: {question}"}
        ]


        self.temperature = temperature  
        last_chars = self.get_content(messages, num_answers)
        self.save_to_cache(question, last_chars, temperature)

        return last_chars

    def process_questions(self, questions_file, output_file, num_answers=10, temperatures=[0.6]):
        ds = pd.read_json(questions_file, lines=True)
        questions = ds['input']
        all_last_chars = {}
        for question in tqdm(questions, desc="Generation Process"):
            all_last_chars[question] = []
            for temp in temperatures:
                print(f"Generating answers for question '{question}' with temperature {temp}")
                last_chars = self.generate_answers_for_question(question, num_answers, temp)
                all_last_chars[question].extend(last_chars)

        with open(output_file, 'w') as f:
            json.dump(all_last_chars, f, indent=4)

        print(f"Results saved to {output_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate answers and manage cache with options.")
    parser.add_argument('--cache_dir', type=str, help="Cache directory path")
    parser.add_argument('--model_id', type=str, help="Model ID for text generation")
    parser.add_argument('--system_prompt', type=str, default="You are an intelligent assistant who answers questions accurately.", help="System prompt for the model")
    parser.add_argument('--questions_file', type=str, help="Path to the questions file")
    parser.add_argument('--output_file', type=str, help="Output file for the generated results")
    parser.add_argument('--num_answers', type=int, default=30, help="Number of answers to generate for each temperature")
    parser.add_argument('--max_new_tokens', type=int, default=1024, help="Max new tokens for the generation")
    parser.add_argument('--temperatures', nargs='+', type=float, default=[0.6], help="List of temperatures to use for generation")
    parser.add_argument('--clear_cache', action='store_true', help="Clear the cache before generating answers")
    parser.add_argument('--device', type=int, default=2,help="Device ID for GPU (-1 for CPU; 0 for first GPU, etc.)")
    args = parser.parse_args()
    
    generator = TextGenerator(
        cache_dir=args.cache_dir,
        model_id=args.model_id,
        system_prompt=args.system_prompt,
        max_new_tokens=args.max_new_tokens,  
        clear_cache=args.clear_cache,
        device=args.device
    )
    generator.process_questions(args.questions_file, args.output_file, num_answers=args.num_answers, temperatures=args.temperatures)
