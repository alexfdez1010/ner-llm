from langchain.prompts import PromptTemplate
from ai.llm import LLM
from model.category import Category
from model.entity import Entity

class ExtractorNER:
    def __init__(self, llm: LLM):
        """Initialize the ExtractorNER with a LLM instance.
        
        Args:
            llm: An instance of LLM class for text generation
        """
        self.llm = llm
        self.template = """
You are an expert in Named Entity Recognition.
Your task is to identify and extract entities from the given text according to the following categories:

{categories}

{examples_text}

Return the entities in the following format:
<category>:<entity>

Each entity should be in a new line.
Only return entities that match exactly with the text.
"""

        self.prompt = PromptTemplate(
            template=self.template,
            input_variables=["categories", "examples_text"]
        )

    def extract_entities(self, categories: list[Category], text: str, examples: list[str] | None = None) -> list[Entity]:
        """Extract entities from the given text according to the specified categories.
        
        Args:
            categories: List of Category objects containing name and description
            text: The text to extract entities from
            examples: Optional list of examples to include in the prompt
            
        Returns:
            List of Entity objects containing category, entity and span
        """
        categories_text = "\n".join([f"{cat.name}: {cat.description}" for cat in categories])
        examples_text = "Here are some examples:\n" + "\n".join(examples) if examples else ""
        
        system_prompt = self.prompt.format(
            categories=categories_text,
            examples_text=examples_text
        )
        
        raw_output = self.llm.generate_completion(system_prompt, text)
        
        entities = []
        for line in raw_output.strip().split('\n'):
            if not line or ':' not in line:
                continue
                
            category, entity = line.split(':', 1)
            category = category.strip().strip('<>')
            entity = entity.strip()
            
            pos = 0
            while True:
                pos = text.find(entity, pos)
                if pos == -1:
                    break
                    
                entities.append(Entity(
                    category=category,
                    entity=entity,
                    span=(pos, pos + len(entity))
                ))
                pos += 1
                
        entities.sort(key=lambda x: x.span[0])
        return entities
