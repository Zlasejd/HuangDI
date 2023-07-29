from transformers import AutoTokenizer
from typing import List
import torch

class MedicalEntityRecognizer:
    def __init__(self, knowledge_base_path: str, tokenizer: AutoTokenizer):
        self.tokenizer = tokenizer
        self.knowledge_base_set = self._preprocess_knowledge_base(knowledge_base_path)

    def _preprocess_knowledge_base(self, knowledge_base_path: str) -> set:
        # Initialize an empty set to store the input_ids of the entities
        knowledge_base_set = set()

        # Open the knowledge base file and go through each line
        with open(knowledge_base_path, 'r', encoding='utf-8') as kb_file:
            for line in kb_file:
                # Strip the newline character and tokenize the line
                entity = line.strip()
                entity_tokens = self.tokenizer.encode(entity, add_special_tokens=False)

                # Convert the list of tokens to a tuple and add it to the set
                knowledge_base_set.add(tuple(entity_tokens))

        return knowledge_base_set

    def extract_entity_positions(self, input_ids: torch.Tensor) -> torch.Tensor:
        # Initialize a tensor of False values with the same size as input_ids
        entity_positions = torch.zeros(input_ids.shape, dtype=torch.bool)

        # Convert the input_ids tensor to a list
        input_list = input_ids.tolist()

        # Go through each entity in the knowledge base set
        for entity in self.knowledge_base_set:
            # Try to find the entity in the input list
            try:
                start_index = input_list.index(entity[0])
                end_index = start_index + len(entity)

                # Check if the entire entity is present in the input list
                if input_list[start_index:end_index] == list(entity):
                    # If it is, set the corresponding positions in the entity_positions tensor to True
                    entity_positions[start_index:end_index] = True
            except ValueError:
                # If the entity is not in the input list, just continue to the next entity
                continue

        return entity_positions

if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,