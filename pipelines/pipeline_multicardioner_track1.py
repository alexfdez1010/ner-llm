from pathlib import Path
from typing import List, Dict

from ai.extractor_ner import ExtractorNER
from ai.llm import LLM
from ai.rules_generator import RulesGenerator
from dataset import Dataset, Instance
from pipeline import Pipeline


def load_brat_file(ann_file: Path, txt_file: Path) -> tuple[List[str], List[int] | None]:
    """Load a BRAT annotation file and its corresponding text file.
    
    Args:
        ann_file: Path to the .ann file
        txt_file: Path to the .txt file
        
    Returns:
        A tuple containing the tokens and their labels
    """
    # Read the text file
    with open(txt_file, "r", encoding="utf-8") as f:
        text = f.read().strip()
    tokens = text.split()  # Simple tokenization by whitespace
    labels = [0] * len(tokens)  # 0 represents 'O' (outside) tag
    
    # Read annotations if they exist
    if ann_file.exists():
        with open(ann_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("T"):  # Entity annotation
                    parts = line.strip().split("\t")
                    if len(parts) >= 3:
                        _, entity_info, _ = parts
                        tag, start, end = entity_info.split()
                        start, end = int(start), int(end)
                        
                        # Find tokens that overlap with this span
                        current_pos = 0
                        for i, token in enumerate(tokens):
                            token_start = text.find(token, current_pos)
                            token_end = token_start + len(token)
                            
                            if token_start < end and token_end > start:
                                labels[i] = 1  # 1 represents entity
                            
                            current_pos = token_end
    
    return tokens, labels


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
            tokens, labels = load_brat_file(ann_file, txt_file)
            training_instances.append(Instance(tokens=tokens, labels=labels))
    
    # Process validation data (cardioccc_dev)
    dev_dir = base_path / "cardioccc_dev" / "brat"
    for ann_file in dev_dir.glob("*.ann"):
        txt_file = ann_file.with_suffix(".txt")
        if txt_file.exists():
            tokens, labels = load_brat_file(ann_file, txt_file)
            validation_instances.append(Instance(tokens=tokens, labels=labels))
    
    # Process test data (cardioccc_test)
    test_dir = base_path / "cardioccc_test" / "brat"
    for ann_file in test_dir.glob("*.ann"):
        txt_file = ann_file.with_suffix(".txt")
        if txt_file.exists():
            tokens, labels = load_brat_file(ann_file, txt_file)
            test_instances.append(Instance(tokens=tokens, labels=labels))
    
    # Create index to category mapping (in this case we only have one category)
    index_to_category: Dict[int, str] = {
        0: None,  # Outside tag
        1: "ENFERMEDAD"  # Disease tag
    }
    
    return Dataset(
        training=training_instances,
        validation=validation_instances,
        test=test_instances,
        index_to_category=index_to_category
    )


if __name__ == "__main__":
    # Initialize components
    llm = LLM()
    extractor = ExtractorNER(llm)
    rules_generator = RulesGenerator(llm)
    
    # Load dataset
    dataset_path = Path(__file__).parent.parent / "datasets" / "multicardioner-track1"
    dataset = load_multicardioner_dataset(dataset_path)
    
    # Create and execute pipeline
    pipeline = Pipeline(extractor, rules_generator, dataset)
    pipeline.execute(
        output_file="rules_multicardioner.json",
        num_iterations=5,
        sample_percentage=0.2
    )
