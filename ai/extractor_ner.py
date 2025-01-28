from langchain.prompts import PromptTemplate
from ai.llm import LLM
from model.category import Category
from model.entity import Entity
from ai.prompts import INITIAL_TEMPLATE


class ExtractorNER:
    def __init__(self, llm: LLM, language: str, example_prompt: str):
        """Initialize the ExtractorNER with a LLM instance.

        Args:
            llm: An instance of LLM class for text generation
        """
        self.llm = llm

        prompt= f"""
        {INITIAL_TEMPLATE[language]}

        {example_prompt}
        """

        self.prompt = PromptTemplate(template=prompt, input_variables=["categories"])

    def extract_entities(
        self, categories: list[Category], text: str, sentences_per_call: int = 0
    ) -> list[Entity]:
        """Extract entities from the given text according to the specified categories.

        Args:
            categories: List of Category objects containing name and description
            text: The text to extract entities from
            sentences_per_call: Number of sentences to process per model call. If 0, process all text at once.

        Returns:
            List of Entity objects containing category, entity and span
        """
        categories_text = "\n".join(
            [f"{cat.name}: {cat.description}" for cat in categories]
        )

        system_prompt = self.prompt.format(categories=categories_text)

        sentences = []
        if sentences_per_call > 0:
            current_pos = 0
            for i, char in enumerate(text):
                if char == "." and (i + 1 == len(text) or text[i + 1].isspace()):
                    sentences.append((text[current_pos : i + 1], current_pos))
                    current_pos = i + 1
            if current_pos < len(text):
                sentences.append((text[current_pos:], current_pos))
        else:
            sentences = [(text, 0)]

        entities: list[Entity] = []
        for i in range(0, len(sentences), max(1, sentences_per_call)):
            batch = sentences[i:i+sentences_per_call] if sentences_per_call > 0 else sentences
            batch_text = " ".join([sentence for sentence, _ in batch])
            
            if sentences_per_call > 0:
                print(f"Processing batch {i//sentences_per_call + 1} of {len(sentences)//sentences_per_call + 1}")

            raw_output = self.llm.generate_completion(system_prompt, batch_text)

            for line in raw_output.strip().split("\n"):
                if not line or ":" not in line:
                    continue

                category, entity = line.split(":", 1)
                category = category.strip().strip("<>")
                entity = entity.strip()

                if not entity or not category:
                    continue

                pos = 0
                while True:
                    pos = batch_text.lower().find(entity.lower(), pos)
                    if pos == -1:
                        break
                    absolute_pos = pos + batch[0][1]
                    entities.append(
                        Entity(
                            category=category,
                            entity=entity,
                            span=(absolute_pos, absolute_pos + len(entity)),
                        )
                    )
                    pos += 1

        categories_names = [cat.name for cat in categories]
        entities = [
            entity for entity in entities if entity.category in categories_names and entity.entity.strip()
        ]
        entities.sort(key=lambda x: x.span[0])
        return entities
