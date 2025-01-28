from pathlib import Path

from dataset import Dataset, Instance
from model.category import Category
from model.entity import Entity
from datasets_info.dataset_info_interface import DatasetInfo
from datasets_info.utils.brat import load_brat_file


class MultiCardionerTrack2Es(DatasetInfo):
    """MultiCardioNER Track 2 Spanish Dataset."""

    def __init__(self) -> None:
        """Initialize the dataset."""
        self.dataset_path = Path("datasets/multicardioner-track2/es")

    def language(self) -> str:
        """Get the language of the dataset.

        Returns:
            The language of the dataset
        """
        return "es"

    def categories(self) -> list[Category]:
        """Get the categories of the dataset.

        Returns:
            A list of categories
        """
        return [
            Category(
                name="FARMACO",
                description="Menciones de medicamentos, incluyendo medicamentos, principios activos y sustancias farmacéuticas",
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
        return """Aquí hay un texto médico sobre un paciente de cardiología. Por favor, identifica todas las menciones de medicamentos (FARMACO) en el texto:

HISTORIA, ENFERMEDAD ACTUAL Y EXPLORACIÓN FÍSICA

Historia
Mujer de 70 años.
Sin alergias medicamentosas conocidas.
Sin hábitos tóxicos.
Factores de riesgo cardiovascular: hipertensión y dislipemia.
Fibrilación auricular permanente (FA) anticoagulada con acenocumarol desde 2014. Intervenciones quirúrgicas: rodilla derecha.
Autónoma para actividades básicas de la vida diaria. Vida activa, sin angina ni disnea previas.
Tratamiento habitual: digoxina 0,25 mg diarios, lorazepam 1 mg al acostarse, acenocumarol 4 mg según pauta, amlodipino/telmisartán 80/5 mg en la cena, verapamilo 240 mg en el desayuno, atorvastatina 40 mg en la cena, tramadol 100 mg si precisa, paracetamol 1 g si precisa, calcipotriol tópico 0,005%.

FARMACO: acenocumarol
FARMACO: digoxina
FARMACO: lorazepam
FARMACO: acenocumarol
FARMACO: amlodipino/telmisartán
FARMACO: verapamilo
FARMACO: atorvastatina
FARMACO: tramadol
FARMACO: paracetamol
FARMACO: calcipotriol
"""
