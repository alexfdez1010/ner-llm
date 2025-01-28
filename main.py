"""
Main script for executing the NER pipeline.
"""

import argparse

from ai.extractor_ner import ExtractorNER
from ai.llm import LLM, LRM
from dataset import Dataset
from pipeline import Pipeline
from model.category import Category
from datasets_info.dataset_info_interface import DatasetInfo
from utils import save_experiment_results

MODELS = ["deepseek-r1:14b", "llama3.2-vision", "llama3.3", "qwen2.5:32b", "phi4"]

DATASETS = {
    "multicardioner_track1": ("datasets_info.multicardioner_track1", "MultiCardionerTrack1"),
    "pharmaconer": ("datasets_info.pharmaconer", "PharmaCoNER"),
    "multicardioner_track2_en": ("datasets_info.multicardioner_track2_en", "MultiCardionerTrack2En"),
    "multicardioner_track2_es": ("datasets_info.multicardioner_track2_es", "MultiCardionerTrack2Es"),
    "multicardioner_track2_it": ("datasets_info.multicardioner_track2_it", "MultiCardionerTrack2It"),
}


def get_dataset_info(dataset_name: str) -> DatasetInfo:
    """Get the dataset info instance for a given dataset name.

    Args:
        dataset_name: Name of the dataset

    Returns:
        DatasetInfo instance
    """
    if dataset_name not in DATASETS:
        raise ValueError(
            f"Dataset {dataset_name} not found. Available datasets: {list(DATASETS.keys())}"
        )

    # Import the dataset info module and class
    module_path, class_name = DATASETS[dataset_name]
    module = __import__(module_path, fromlist=[class_name])
    dataset_info_class = getattr(module, class_name)

    return dataset_info_class()


def get_dataset_loader(dataset_name: str) -> Dataset:
    """Get the dataset loader function for a given dataset name.

    Args:
        dataset_name: Name of the dataset

    Returns:
        Dataset loader function
    """
    dataset_info = get_dataset_info(dataset_name)
    return dataset_info.load_dataset


def get_categories(dataset_name: str) -> list[Category]:
    """Get the categories of the dataset.

    Returns:
        A list of categories
    """
    dataset_info = get_dataset_info(dataset_name)
    return dataset_info.categories()


def get_language(dataset_name: str) -> str:
    """Get the language of the dataset.

    Returns:
        The language of the dataset
    """
    dataset_info = get_dataset_info(dataset_name)
    return dataset_info.language()


def get_example_prompt(dataset_name: str) -> str:
    """Get the example prompt of the dataset.

    Returns:
        The example prompt of the dataset
    """
    dataset_info = get_dataset_info(dataset_name)
    return dataset_info.example_prompt()


def create_pipeline(
    model_name: str, dataset: Dataset, categories: list[Category], example_prompt: str, language: str
) -> Pipeline:
    """Create a pipeline with the specified model and dataset.

    Args:
        model_name: Name of the model to use
        dataset: Dataset to use
        categories: List of categories
        example_prompt: Example prompt for the dataset
        language: Language of the dataset

    Returns:
        Pipeline instance
    """
    is_reasoning = model_name.startswith("deepseek-r1")
    llm = LLM(model=model_name) if not is_reasoning else LRM(model=model_name)

    extractor = ExtractorNER(
        llm=llm,
        example_prompt=example_prompt,
        language=language,
    )
    return Pipeline(extractor=extractor, dataset=dataset, categories=categories)


def main():
    """Main function to execute the NER pipeline."""
    parser = argparse.ArgumentParser(description="Execute NER pipeline")
    parser.add_argument(
        "--model",
        type=str,
        default="llama3.2-vision",
        choices=MODELS,
        help="Model to use for NER",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="multicardioner_track1",
        choices=list(DATASETS.keys()),
        help="Dataset to use",
    )
    parser.add_argument(
        "--sentences-per-call",
        default=0,
        type=int,
        help="Number of sentences to process per model call. If 0, process all text at once",
    )

    args = parser.parse_args()

    # Get dataset info and load dataset
    dataset_info = get_dataset_info(args.dataset)
    dataset = dataset_info.load_dataset()
    categories = dataset_info.categories()
    example_prompt = dataset_info.example_prompt()
    language = dataset_info.language()

    pipeline = create_pipeline(
        model_name=args.model,
        dataset=dataset,
        categories=categories,
        example_prompt=example_prompt,
        language=language,
    )
    print(f"\nStarting evaluation of {args.dataset} with model {args.model} and categories {', '.join([cat.name for cat in categories])}...")
    # Evaluate pipeline
    micro_metrics, macro_metrics = pipeline.evaluate(
        sentences_per_call=args.sentences_per_call
    )

    # Save results
    save_experiment_results(
        args.results_file,
        args.model,
        args.dataset,
        args.paragraphs_per_call,
        micro_metrics,
        macro_metrics,
    )


if __name__ == "__main__":
    main()
