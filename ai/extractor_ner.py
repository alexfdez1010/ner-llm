from langchain.prompts import PromptTemplate
from ai.llm import LLM
from model.category import Category
from model.entity import Entity
from ai.prompts import INITIAL_TEMPLATE


class ExtractorNER:
    def __init__(self, llm: LLM, language: str, example_prompt: str | None = None):
        """Initialize the ExtractorNER with a LLM instance.

        Args:
            llm: An instance of LLM class for text generation
            language: Language of the dataset
            example_prompt: Example prompt for the dataset
        """
        self.llm = llm

        prompt= f"""
        {INITIAL_TEMPLATE[language]}

        {example_prompt}
        """ if example_prompt else f"""
        {INITIAL_TEMPLATE[language]}
        """

        self.prompt = PromptTemplate(template=prompt, input_variables=["categories"])

    def extract_entities(
        self, categories: list[Category], text: str, sentences_per_call: int = 0, examples: str = ""
    ) -> list[Entity]:
        """Extract entities from the given text according to the specified categories.

        Args:
            categories: List of Category objects containing name and description
            text: The text to extract entities from
            sentences_per_call: Number of sentences to process per model call. If 0, process all text at once.
            examples: Optional example text to guide the model

        Returns:
            List of Entity objects containing category, entity and span
        """
        categories_text = "\n".join(
            [f"{cat.name}: {cat.description}" for cat in categories]
        )

        system_prompt = self.prompt.format(categories=categories_text)
        if examples:
            system_prompt += f"\n\nExamples:\n{examples}"

        sentences = []
        if sentences_per_call > 0:
            current_pos = 0
            for i, char in enumerate(text):
                if char == "\n":
                    sentences.append((text[current_pos : i + 1].strip(), current_pos))
                    current_pos = i + 1
            if current_pos < len(text):
                sentences.append((text[current_pos:].strip(), current_pos))
        else:
            sentences = [(text.strip(), 0)]

        # First, collect all unique entity-category pairs from LLM outputs
        entity_category_pairs = set()
        
        for i in range(0, len(sentences), max(1, sentences_per_call)):
            batch = sentences[i:i+sentences_per_call] if sentences_per_call > 0 else sentences
            batch_text = "\n".join([sentence for sentence, _ in batch])
            
            if sentences_per_call > 0:
                print(f"Processing batch {i//sentences_per_call + 1} of {len(sentences)//sentences_per_call + 1}")

            raw_output = self.llm.generate_completion(system_prompt, batch_text)

            # Process each entity line
            for line in raw_output.strip().split("\n"):
                if not line or ":" not in line:
                    continue

                category, entity = line.split(":", 1)
                category = category.strip("<>")
                entity = entity.strip()

                # Skip empty entities or categories
                if not entity or not category:
                    continue

                entity_category_pairs.add((category, entity))

        # Now find all occurrences of each entity in the original text
        entities: list[Entity] = []
        for category, entity in entity_category_pairs:
            start_pos = 0
            while True:
                start_idx = text.find(entity, start_pos)
                if start_idx == -1:
                    break
                
                end_idx = start_idx + len(entity)
                entities.append(Entity(category, entity, (start_idx, end_idx)))
                start_pos = end_idx

        categories_names = [cat.name for cat in categories]
        entities = [
            entity for entity in entities if entity.category in categories_names and entity.entity.strip()
        ]
        entities.sort(key=lambda x: x.span[0])
        return entities
