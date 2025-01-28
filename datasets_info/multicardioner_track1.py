from pathlib import Path
from typing import List

from dataset import Dataset, Instance
from model.entity import Entity
from model.category import Category
from datasets_info.dataset_info_interface import DatasetInfo


def load_brat_file(ann_file: Path, txt_file: Path) -> tuple[str, list[Entity] | None]:
    """Load a BRAT annotation file and its corresponding text file.

    Args:
        ann_file: Path to the .ann file
        txt_file: Path to the .txt file

    Returns:
        A tuple containing the text and its entities
    """
    # Read the text file
    with open(txt_file, "r", encoding="utf-8") as f:
        text = f.read().strip()
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
                            Entity(category=tag, entity=entity_text, span=(start, end))
                        )

    return text, entities if entities else None


class MultiCardionerTrack1(DatasetInfo):
    """Implementation of DatasetInfo for MultiCardioner Track 1 dataset."""

    def load_dataset(self) -> Dataset:
        """Load the test part of the Multicardioner dataset from the given base path.

        Returns:
            A Dataset object containing the loaded test data
        """
        base_path = Path("datasets") / "multicardioner-track1"
        test_instances: list[Instance] = []

        # Process test data (cardioccc_test)
        test_dir = base_path / "cardioccc_test" / "brat"
        for ann_file in test_dir.glob("*.ann"):
            txt_file = ann_file.with_suffix(".txt")
            if txt_file.exists():
                text, entities = load_brat_file(ann_file, txt_file)
                test_instances.append(Instance(text=text, entities=entities))

        return Dataset(instances=test_instances)

    def categories(self) -> List[Category]:
        """Return a list of all categories in the dataset."""
        return [Category(name="ENFERMEDAD", description="Enfermedades")]

    def language(self) -> str:
        """Return the language of the dataset."""
        return "es"

    def example_prompt(self) -> str:
        """Return an example prompt for the dataset."""
        return f"""
Categorias:
{"\n".join([f'{cat.name}: {cat.description}' for cat in self.categories()])}

Texto:
ANTECEDENTES:
Mujer de 77 años de edad, alérgica a metamizol, y con antecedentes médicos de hipertensión arterial, diabetes mellitus tipo 2 (insulinodependiente), fibrilación auricular permanente y asmabronquial. En seguimiento por cardiología por una valvulopatía mitral reumática con estenosismitral severa intervenida mediante comisurotomía quirúrgica en 1991 y posteriormente recambio valvular protésico en 2007 mediante prótesis metálica Carbomedics no 27. Durante el seguimiento desarrolló también una estenosis valvular aórtica grave no sintomática.Se encontraba en tratamiento con acenocumarol, bisoprolol, furosemida, espironolactona,olmesartán, insulina y broncodilatadores a demanda.

ENFERMEDAD ACTUAL:
La paciente acudió a urgencias por presentar dolor abdominal tipo cólico y fiebre de 38,2 oC,etiquetándose de gastroenteritis aguda que se resolvió con tratamiento conservador de formaambulatoria. A los 5 días regresó a nuestro centro por un nuevo pico febril y dolor abdominalfocalizado en fosa ilíaca izquierda que requirió ingreso hospitalario por sospecha de diverticulitisaguda. Se aisló Bacteroides fragilis en dos hemocultivos y se pautó tratamiento antibiótico con buena evolución posterior. Veinte días más tarde, volvió a ingresar por nuevo pico febril con hemocultivos positivos para el mismo patógeno; se  realizó un TC abdominal que mostró diverticulosis sin signos de infección aguda, por lo que dados sus antecedentes, se decidió realizar ecocardiograma para descartar endocarditis infecciosa (EI).

EXPLORACIÓN FÍSICA
Temperatura 38,3 oC. Tensión arterial 105/70 mmHg. Frecuencia cardiaca 68 lpm. Consciente,orientada, bien perfundida. En piel no se observan nódulos de Osler ni manchas de Roth. Aumento del pulso venoso yugular con pulso venoso bajo el ángulo mandibular a 90o con ondas"v" prominentes. Auscultación cardiaca: ruidos cardiacos rítmicos a 74 lpm con soplo sistólico3/6 en foco aórtico y segundo ruido abolido, irradiado a borde esternal izquierdo y ápex. Auscultación pulmonar: hipoventilación basal izquierda. Sin edemas en extremidades inferiores.

PRUEBAS COMPLEMENTARIAS
ANALÍTICA al ingreso: urea 0,36 g/l, creatinina 0,73 mg/dl; CKD-EPI 79,6 ml/min/1,73 m2, iones y transaminasas sin alteraciones significativas. NT-proBNP 2.797 pg/ml. VSG 102 mm, proteína Creactiva 0,81 mg/dl. Hemograma: Hb 10,2 g/dl, Hto 30%, 13.900/mm3 (neutrófilos 91,1%).
RADIGRAFÍA DE TÓRAX: cardiomegalia radiológica. Suturas de esternotomía media. Pinzamientode ambos senos costofrénicos (mayor en el lado izquierdo). Hilios vasculares prominentes.
HEMOCULTIVOS (x3): positivo para Bacteroides fragilis.
ECOCARDIOGRAFÍA TRANSESOFÁGICA: aurícula izquierda gravemente dilatada. Tabique íntegro: orejuela ocupada por imagen sugestiva de trombo poco ecodenso. Prótesis mitral de doble hemidisco con buena apertura (gradientes similares a los de estudios previos); se observan varios jets de regurgitación excéntricos intraprótesis sin inversión del flujo en lasvenas pulmonares. Se aprecian dos imágenes vegetantes implantadas en el anillo protésico de12 x 3 mm y de 10 x 3 mm móviles que no interfieren el movimiento de los discos; no sedetectan abscesos perianulares. Ventrículo izquierdo no dilatado ni hipertrofiado con fracción de eyección preservada y sin alteraciones segmentarias. Válvula aórtica trivalva, engrosada, con área efectiva reducida de forma significativa; sin regurgitación. Cavidades derechas dilatadas; contractilidad ventricular derecha reducida (TAPSE 14 mm). Válvula tricúspide engrosada conbuena apertura y movilidad; regurgitación grave. Hipertensión pulmonar moderada. No seaprecia derrame pericárdico.

EVOLUCIÓN CLÍNICA
Con el diagnóstico de endocarditis infecciosa sobre válvula protésica por Bacteroides fragilis, secomenzó tratamiento con metronidazol 500 mg/8 horas y amoxicilina-clavulánico 1000 mg/200mg/8 horas intravenoso. La paciente permaneció afebril durante todo el ingreso, senegativizaron los hemocultivos de forma precoz y evolucionó de forma favorables de su ligera descompensación cardiaca con tratamiento diurético. Tras 6 semanas de tratamiento antibiótico intravenoso dirigido, estando estable hemodinámicamente y en buena clase funcional se dio de alta hospitalaria.

DIAGNÓSTICO
Endocarditis sobre prótesis metálica mitral por anaerobios (Bacteroides fragilis) sin disfunción protésica asociada. Bacteriemia por Bacteroides fragilis de posible origen gastrointestinal. Diverticulosis sin diverticulitis.

Salida:

ENFERMEDAD: Válvula tricúspide engrosada con buena apertura y movilidad; regurgitación grave
ENFERMEDAD: alérgica a metamizol
ENFERMEDAD: hipertensión arterial
ENFERMEDAD: diabetes mellitus tipo 2 (insulinodependiente)
ENFERMEDAD: fibrilación auricular permanente
ENFERMEDAD: valvulopatía mitral reumática
ENFERMEDAD: estenosismitral severa
ENFERMEDAD: estenosis valvular aórtica grave
ENFERMEDAD: gastroenteritis aguda
ENFERMEDAD: diverticulitisaguda
ENFERMEDAD: diverticulosis
ENFERMEDAD: infección aguda
ENFERMEDAD: endocarditis infecciosa
ENFERMEDAD: EI
ENFERMEDAD: cardiomegalia
ENFERMEDAD: abscesos perianulares
ENFERMEDAD: Hipertensión pulmonar moderada
ENFERMEDAD: derrame pericárdico
ENFERMEDAD: endocarditis infecciosa sobre válvula protésica por Bacteroides fragilis
ENFERMEDAD: descompensación cardiaca
ENFERMEDAD: Endocarditis sobre prótesis metálica mitral por anaerobios (Bacteroides fragilis)
ENFERMEDAD: Bacteriemia por Bacteroides fragilis
ENFERMEDAD: Diverticulosis
ENFERMEDAD: diverticulitis
ENFERMEDAD: asmabronquial
ENFERMEDAD: aurícula izquierda gravemente dilatada
ENFERMEDAD: Cavidades derechas dilatadas
ENFERMEDAD: nódulos de Osler
ENFERMEDAD: manchas de Roth
ENFERMEDAD: Válvula aórtica trivalva, engrosada
ENFERMEDAD: disfunción protésica
ENFERMEDAD: trombo
"""
