from pathlib import Path
from typing import List

from ai.extractor_ner import ExtractorNER
from ai.llm import LLM
from ai.rules_generator import RulesGenerator
from dataset import Dataset, Instance
from pipeline import Pipeline
from model.category import Category
from model.entity import Entity


def load_brat_file(
    ann_file: Path, txt_file: Path
) -> tuple[List[str], List[Entity] | None]:
    """Load a BRAT annotation file and its corresponding text file.

    Args:
        ann_file: Path to the .ann file
        txt_file: Path to the .txt file

    Returns:
        A tuple containing the tokens and their entities
    """
    # Read the text file
    with open(txt_file, "r", encoding="utf-8") as f:
        text = f.read().strip()
    tokens = text.split()  # Simple tokenization by whitespace
    entities: list[Entity] = []

    # Read annotations if they exist
    if ann_file.exists():
        with open(ann_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("T"):  # Entity annotation
                    parts = line.strip().split("\t")
                    if len(parts) >= 3:
                        _, entity_info, entity_text = parts
                        tag, start, end = entity_info.split()
                        start, end = int(start), int(end)
                        
                        # Create entity with the exact span and text
                        entities.append(
                            Entity(
                                category=tag,
                                entity=entity_text,
                                span=(start, end)
                            )
                        )

    return tokens, entities if entities else None


def load_multicardioner_dataset(base_path: str | Path) -> Dataset:
    """Load the Multicardioner dataset from the given base path.

    Args:
        base_path: Path to the dataset directory

    Returns:
        A Dataset object containing the loaded data
    """
    base_path = Path(base_path)

    # Initialize lists for different splits
    training_instances: List[Instance] = []
    validation_instances: List[Instance] = []
    test_instances: List[Instance] = []

    # Process training data (distemist_train)
    train_dir = base_path / "distemist_train" / "brat"
    for ann_file in train_dir.glob("*.ann"):
        txt_file = ann_file.with_suffix(".txt")
        if txt_file.exists():
            tokens, entities = load_brat_file(ann_file, txt_file)
            training_instances.append(Instance(tokens=tokens, entities=entities))

    # Process validation data (cardioccc_dev)
    dev_dir = base_path / "cardioccc_dev" / "brat"
    for ann_file in dev_dir.glob("*.ann"):
        txt_file = ann_file.with_suffix(".txt")
        if txt_file.exists():
            tokens, entities = load_brat_file(ann_file, txt_file)
            validation_instances.append(Instance(tokens=tokens, entities=entities))

    # Process test data (cardioccc_test)
    test_dir = base_path / "cardioccc_test" / "brat"
    for ann_file in test_dir.glob("*.ann"):
        txt_file = ann_file.with_suffix(".txt")
        if txt_file.exists():
            tokens, entities = load_brat_file(ann_file, txt_file)
            test_instances.append(Instance(tokens=tokens, entities=entities))

    return Dataset(
        training=training_instances,
        validation=validation_instances,
        test=test_instances,
    )


def main():
    # Initialize components
    llm = LLM(model="llama3.2-vision")
    extractor = ExtractorNER(llm)
    rules_generator = RulesGenerator(llm)

    # Load dataset
    dataset_path = Path(__file__).parent.parent / "datasets" / "multicardioner-track1"
    dataset = load_multicardioner_dataset(dataset_path)

    pipeline = Pipeline(extractor, rules_generator, dataset, "es")
    pipeline.execute(
        output_file="rules_generated/rules_multicardioner.json",
        num_iterations=200,
        categories=[Category(name="ENFERMEDAD", description="Enfermedades cardíacas")],
        sample_percentage=0.005,
    )

    rules = pipeline.load_rules("rules_generated/rules_multicardioner.json")
    precision, recall, f1 = pipeline.evaluate(rules, "test")

    print(f"Precision: {precision}\nRecall: {recall}\nF1: {f1}")

if __name__ == "__main__":
    main()
