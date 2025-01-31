"""Demo NER app."""

from typing import Dict
import threading
import signal

import spacy
import streamlit as st
from dotenv import load_dotenv
from streamlit_tags import st_tags

from ai.extractor_ner import ExtractorNER
from ai.llm import LLM
from model.category import Category

MODEL_NAME = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"

LANGUAGES = {
    "English": "en",
    "Spanish": "es",
    "Italian": "it",
}

# Load environment variables
load_dotenv()


# Load spaCy model for visualization
@st.cache_resource
def load_spacy():
    """Load the spaCy model."""
    return spacy.blank("en")


@st.cache_resource
def get_llm() -> LLM:
    """Initialize the LLM with the specified model."""
    return LLM(MODEL_NAME)


def main() -> None:
    """Main function to run the NER app."""
    # Initialize the page
    st.set_page_config(
        page_title="NER Extractor Demo",
        page_icon="🎯",
        layout="wide",
    )

    # Custom CSS for better visualization
    st.markdown(
        """
        <style>
        .entity-box {
            padding: 0.2em 0.5em;
            margin: 0 0.2em;
            line-height: 2;
            border-radius: 0.35em;
        }
        .main-text {
            line-height: 2;
            font-size: 1.1em;
        }
        .example-text {
            font-family: monospace;
            white-space: pre-wrap;
            background-color: #f0f2f6;
            padding: 1em;
            border-radius: 0.5em;
        }
        </style>
    """,
        unsafe_allow_html=True,
    )
    st.title("Named Entity Recognition Demo ")
    st.markdown(
        """
    This demo allows you to extract named entities from text using LLM-based NER.
    Add categories with their descriptions and provide text to analyze.
    """
    )

    # Sidebar for categories
    with st.sidebar:
        st.header("Category Configuration")
        st.markdown("Add between 1 and 5 categories with their descriptions.")

        # Category input
        category_names = st_tags(
            label="Enter Categories:",
            text="Press enter to add more",
            value=[],
            maxtags=5,
            key="category_names",
        )

        # Description input for each category
        descriptions: Dict[str, str] = {}
        for cat in category_names:
            desc = st.text_area(
                f"Description for {cat}:",
                key=f"desc_{cat}",
                placeholder="Enter a description for this category...",
            )
            if desc:
                descriptions[cat] = desc
    st.markdown("### Model Configuration")

    language = st.selectbox("Select language:", LANGUAGES, index=0)

    # Text input
    text_input = st.text_area(
        "Enter text to analyze:",
        height=200,
        placeholder="Enter the text you want to analyze...",
    )

    # Process button
    if st.button(
        "Extract Entities",
        type="primary",
        disabled=len(category_names) < 1 or len(category_names) > 5 or not text_input,
    ):
        if not all(cat in descriptions for cat in category_names):
            st.error("Please provide descriptions for all categories!")
            return

        # Create Category objects
        categories = [
            Category(name=name, description=descriptions[name])
            for name in category_names
        ]

        with st.spinner("Extracting entities..."):
            try:
                # Initialize LLM and ExtractorNER
                llm = get_llm()
                extractor = ExtractorNER(llm=llm, language=LANGUAGES[language])

                # Extract entities
                entities = extractor.extract_entities(
                    categories=categories, text=text_input
                )

                # Visualize results
                st.markdown("### Results")

                # Create spaCy-like visualization
                nlp = load_spacy()
                doc = nlp(text_input)

                # Convert the text to HTML with entity highlighting
                html_text = text_input
                for entity in reversed(
                    entities
                ):  # Reversed to handle overlapping spans
                    start, end = entity.span
                    entity_text = text_input[start:end]
                    color = f"hsl({hash(entity.category) % 360}, 70%, 80%)"
                    html_text = (
                        html_text[:start]
                        + f'<mark class="entity-box" style="background: {color}">{entity_text}<span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">{entity.category}</span></mark>'
                        + html_text[end:]
                    )

                st.markdown(
                    f'<div class="main-text">{html_text}</div>', unsafe_allow_html=True
                )

                # Show entities in a table
                st.markdown("### Extracted Entities")
                entities_df = []
                for entity in entities:
                    entities_df.append(
                        {
                            "Category": entity.category,
                            "Entity": entity.entity,
                            "Start": entity.span[0],
                            "End": entity.span[1],
                        }
                    )

                if entities_df:
                    st.table(entities_df)
                else:
                    st.info("No entities were found in the text.")

            except Exception as e:
                print(e)
                st.error(
                    "We have received too many calls to the model. Please try again later."
                )


if __name__ == "__main__":
    main()
