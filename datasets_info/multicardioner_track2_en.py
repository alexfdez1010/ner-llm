from pathlib import Path

from model.category import Category
from model.entity import Entity
from datasets_info.dataset_info_interface import DatasetInfo
from datasets_info.utils.brat import load_brat_file


class MultiCardionerTrack2En(DatasetInfo):
    """MultiCardioNER Track 2 English Dataset."""

    def __init__(self) -> None:
        """Initialize the dataset."""
        self.dataset_path = Path("datasets/multicardioner-track2/en")

    def language(self) -> str:
        """Get the language of the dataset.

        Returns:
            The language of the dataset
        """
        return "en"

    def categories(self) -> list[Category]:
        """Get the categories of the dataset.

        Returns:
            A list of categories
        """
        return [
            Category(
                name="FARMACO",
                description="Drug mentions, including medications, active ingredients, and pharmaceutical substances",
            ),
        ]

    def load_file(self, ann_file: Path, txt_file: Path) -> tuple[str, list[Entity] | None]:
        """Load a file from the dataset.

        Args:
            ann_file: Path to the .ann file
            txt_file: Path to the .txt file

        Returns:
            A tuple containing the text and its entities
        """
        return load_brat_file(ann_file, txt_file)

    def example_prompt(self) -> str:
        """Get an example prompt for the dataset.

        Returns:
            An example prompt
        """
        return """Here is a medical text about a cardiology patient. Please identify all the drug mentions (FARMACO) in the text:

HISTORY, CURRENT ILLNESS AND PHYSICAL EXAMINATION

History
70-year-old woman.
No known drug allergies.
No toxic habits.
Cardiovascular risk factors: hypertension and dyslipidaemia.
Permanent atrial fibrillation (AF) anticoagulated with acenucoumarol since 2014. Surgical interventions: right knee.
Autonomous for basic activities of daily living. Active life, no previous angina or dyspnoea.
Usual treatment: digoxin 0.25 mg daily, lorazepam 1 mg at bedtime, acenocoumarol 4 mg as prescribed, amlodipine/telmisartan 80/5 mg at dinner, verapamil 240 mg at breakfast, atorvastatin 40 mg at dinner, tramadol 100 mg if required, paracetamol 1 g if required, topical calcipotriol 0.005%.

FARMACO: acenucoumarol
FARMACO: digoxin
FARMACO: lorazepam
FARMACO: acenocoumarol
FARMACO: amlodipine/telmisartan
FARMACO: verapamil
FARMACO: atorvastatin
FARMACO: tramadol
FARMACO: paracetamol
FARMACO: calcipotriol
"""
