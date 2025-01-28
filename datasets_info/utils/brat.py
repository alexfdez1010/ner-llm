from pathlib import Path

from model.entity import Entity


def load_brat_file(ann_file: Path, txt_file: Path) -> tuple[str, list[Entity] | None]:
    """Load a BRAT annotation file and its corresponding text file.

    Args:
        ann_file: Path to the .ann file
        txt_file: Path to the .txt file

    Returns:
        A tuple containing the text and its entities (None if no entities found)
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
                    # Skip AnnotatorNotes
                    if "#" in line:
                        continue
                    parts = line.strip().split("\t")
                    if len(parts) == 3:
                        _, span_type, text_span = parts
                        category, start, end = span_type.split()
                        start, end = int(start), int(end)
                        entities.append(
                            Entity(
                                entity=text_span,
                                category=category,
                                span=(start, end),
                            )
                        )

    return text, entities if entities else None
