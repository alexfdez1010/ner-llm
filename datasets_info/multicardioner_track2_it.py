from pathlib import Path

from dataset import Dataset, Instance
from model.category import Category
from model.entity import Entity
from datasets_info.dataset_info_interface import DatasetInfo
from datasets_info.utils.brat import load_brat_file


class MultiCardionerTrack2It(DatasetInfo):
    """MultiCardioNER Track 2 Italian Dataset."""

    def __init__(self) -> None:
        """Initialize the dataset."""
        self.dataset_path = Path("datasets/multicardioner-track2/it")

    def language(self) -> str:
        """Get the language of the dataset.

        Returns:
            The language of the dataset
        """
        return "it"

    def categories(self) -> list[Category]:
        """Get the categories of the dataset.

        Returns:
            A list of categories
        """
        return [
            Category(
                name="FARMACO",
                description="Menzioni di farmaci, compresi medicinali, principi attivi e sostanze farmaceutiche",
            ),
        ]

    def load_dataset(self) -> Dataset:
        """Load the dataset from the given base path.

        Returns:
            A Dataset object containing the loaded data
        """
        instances: list[Instance] = []

        # Process brat data
        brat_dir = self.dataset_path / "brat"
        for ann_file in brat_dir.glob("*.ann"):
            txt_file = ann_file.with_suffix(".txt")
            if txt_file.exists():
                text, entities = load_brat_file(ann_file, txt_file)
                instances.append(Instance(text=text, entities=entities))

        return Dataset(instances=instances)

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
        return """Ecco un testo medico su un paziente cardiologico. Si prega di identificare tutte le menzioni di farmaci (FARMACO) nel testo:

ANAMNESI, MALATTIA ATTUALE ED ESAME FISICO

Anamnesi
Donna di 70 anni.
Nessuna allergia nota ai farmaci.
Nessuna abitudine tossica.
Fattori di rischio cardiovascolare: ipertensione e dislipidemia.
Fibrillazione atriale permanente (FA) anticoagulata con acenucumarolo dal 2014. Interventi chirurgici: ginocchio destro.
Autonoma per le attività di base della vita quotidiana. Vita attiva, nessuna angina o dispnea precedente.
Trattamento abituale: digossina 0,25 mg al giorno, lorazepam 1 mg al momento di coricarsi, acenocumarolo 4 mg come prescritto, amlodipina/telmisartan 80/5 mg a cena, verapamil 240 mg a colazione, atorvastatina 40 mg a cena, tramadolo 100 mg se necessario, paracetamolo 1 g se necessario, calcipotriolo topico 0,005%.

FARMACO: acenucumarolo
FARMACO: digossina
FARMACO: lorazepam
FARMACO: acenocumarolo
FARMACO: amlodipina/telmisartan
FARMACO: verapamil
FARMACO: atorvastatina
FARMACO: tramadolo
FARMACO: paracetamolo
FARMACO: calcipotriolo
"""
