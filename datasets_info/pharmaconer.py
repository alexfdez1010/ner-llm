from pathlib import Path
from typing import List

from dataset import Dataset, Instance
from model.category import Category
from datasets_info.dataset_info_interface import DatasetInfo
from datasets_info.utils.brat import load_brat_file


class PharmaCoNER(DatasetInfo):
    """Implementation of DatasetInfo for PharmaCoNER dataset.
    
    The PharmaCoNER dataset contains Spanish clinical cases annotated with pharmacological
    substances, focusing on PROTEINAS (proteins) and NORMALIZABLES (normalizable mentions).
    """

    def load_dataset(self) -> Dataset:
        """Load the test part of the PharmaCoNER dataset from the given base path.

        Returns:
            A Dataset object containing the loaded test data
        """
        base_path = Path("datasets/PharmaCoNERCorpus/test")
        instances: List[Instance] = []

        # Get all .txt files
        txt_files = list(base_path.glob("*.txt"))
        
        for txt_file in txt_files:
            # Get corresponding .ann file
            ann_file = txt_file.with_suffix(".ann")
            
            # Load the BRAT file
            text, entities = load_brat_file(ann_file, txt_file)
            
            if entities:
                instances.append(Instance(text=text, entities=entities))

        return Dataset(instances=instances)

    def categories(self) -> List[Category]:
        """Return a list of all categories in the dataset.

        Returns:
            List of Category objects representing the dataset's categories
        """
        return [
            Category(
                name="PROTEINAS",
                description="Menciones de proteínas, incluyendo enzimas, factores de transcripción y otras moléculas de proteínas",
            ),
            Category(
                name="NORMALIZABLES",
                description="Menciones de sustancias farmacológicas que pueden ser normalizadas a terminologías estándar",
            ),
        ]

    def language(self) -> str:
        """Return the language of the dataset.

        Returns:
            Language code 'es' for Spanish
        """
        return "es"

    def example_prompt(self) -> str:
        """Return an example prompt for the dataset.

        Returns:
            A string containing an example prompt
        """
        return """
Categorías:
PROTEINAS: Menciones de proteínas, incluyendo enzimas, factores de transcripción y otras moléculas proteicas
NORMALIZABLES: Menciones de sustancias farmacológicas que pueden normalizarse según terminologías estándar

Texto:
Mujer, diagnosticada de HTA a los 14 años, cuando fue admitida por primera vez en un hospital pediátrico, con presión arterial (PA) de 210/120mmHg, cefaleas e hipopotasemia. Con la sospecha de HTA secundaria, se le realizaron diversos estudios. La ecografía renal, el ecocardiograma y el cateterismo de aorta y arterias renales no evidenciaron alteraciones. Los dosajes de actividad de renina plasmática (ARP) y aldosterona plasmática (AP) resultaron elevados en dos ocasiones. Los niveles de ácido vanilmandélico en orina y los de TSH, T4 libre y T3 plasmáticos resultaron normales. Recibió el alta hospitalaria medicada con inhibidores de la enzima convertidora de angiotensina y bloqueante cálcico, pero no regresó a control.

Salida:
PROTEINAS: renina
PROTEINAS: enzima convertidora de angiotensina
NORMALIZABLES: aldosterona
NORMALIZABLES: T4 libre
NORMALIZABLES: T3
"""