import nltk
from nltk.tokenize import sent_tokenize
from transformers import AutoModel, AutoTokenizer
import torch
from simcse import SimCSE
from sklearn.cluster import AgglomerativeClustering
import json
import numpy as np
from collections import defaultdict
import argparse
SEModel = SimCSE("princeton-nlp/sup-simcse-roberta-large")
class AtomicResponseProcessor:
    def __init__(self):
        pass

    def process_question_responses(self, question, generated_responses):
        all_sentences = []
        sentence_to_response_map = []  
        for idx, response in enumerate(generated_responses):
            content = response["content"]
            sentences = sent_tokenize(content)
            all_sentences.extend(sentences)
            sentence_to_response_map.extend([idx] * len(sentences))

        print(f"Processing '{question}': Extracted {len(all_sentences)} sentences from {len(generated_responses)} responses.")
        if len(all_sentences) == 0:
            return None, None

        embeddings = SEModel.encode(all_sentences)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        clustering_model = AgglomerativeClustering(n_clusters=None, metric='cosine', linkage='average', distance_threshold=0.15)
        clusters = clustering_model.fit_predict(embeddings)
        print("#####")
        clustered_sentences = defaultdict(list)
        response_contributions = defaultdict(set)
        for sentence, cluster_id, response_id in zip(all_sentences, clusters, sentence_to_response_map):
            clustered_sentences[int(cluster_id)].append((sentence, response_id))
            response_contributions[int(response_id)].add(int(cluster_id))
        print("#####")
        return clustered_sentences, response_contributions
    
    def filter_and_calculate_scores(self, clustered_sentences, response_contributions, generated_responses):
        response_contribution_count = {}
        for response_id, clusters in response_contributions.items():
            response_contribution_count[response_id] = 0
            for cluster_id in clusters:
                if len(clustered_sentences[cluster_id])>1: 
                    response_contribution_count[response_id] += 1
                if len(clustered_sentences[cluster_id])<=1:
                    response_contribution_count[response_id] -= 1
                
        response_contribution_scores = [
            [response["content"], response_contribution_count[idx]]
            for idx, response in enumerate(generated_responses)
        ]
        return response_contribution_scores


    def process_responses(self, responses_file, clustering_results_file, response_contributions_file):
        with open(responses_file, "r") as f:
            responses = json.load(f)

        clustering_results = {}
        response_contributions_results = {}

        for question, generated_responses in responses.items():
            clustered_sentences, response_contributions = self.process_question_responses(question, generated_responses)
            clustering_results[question] = clustered_sentences
            if clustered_sentences:
                response_scores = self.filter_and_calculate_scores(clustered_sentences, response_contributions, generated_responses)
                response_contributions_results[question] = response_scores
        print("------")
        
        
        with open(clustering_results_file, "w") as f:
            json.dump(clustering_results, f, indent=4)
        with open(response_contributions_file, "w") as f:
            json.dump(response_contributions_results, f, indent=4)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process question responses and perform clustering.")

    parser.add_argument('--responses_file', type=str,
                        help="Path to the JSON file containing generated responses")
    parser.add_argument('--clustering_results_file', type=str,
                        help="Output file for clustering results")
    parser.add_argument('--response_contributions_file', type=str,
                        help="Output file for response contribution scores")

    args = parser.parse_args()
    
    processor = AtomicResponseProcessor()
    processor.process_responses(
        responses_file=args.responses_file,
        clustering_results_file=args.clustering_results_file,
        response_contributions_file=args.response_contributions_file
    )
