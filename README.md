# Named Entity Recognition with LLMs and LRMs

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT) [![Python](https://img.shields.io/badge/Python-3.12%2B-green.svg)](https://www.python.org)

## 🎓 Master Thesis Project

This repository contains a Named Entity Recognition (NER) framework that explores the capabilities of Large Language Models (LLMs) and Large Reasoning Models (lRMs) for extracting entities from medical texts. The project features both an interactive demo using Together AI and experimental pipelines using Ollama for research purposes.

## 🚀 Features

- Interactive web demo for real-time NER extraction
- Support for multiple languages (English, Spanish, Italian, more can be added easily)
- Modular architecture for easy dataset and model integration
- Comprehensive evaluation pipeline for NER experiments
- Support for various medical datasets (MultiCardioNER, PharmaCoNER)

## 💻 Setup

### Prerequisites

- Python 3.12+
- [Together AI API key](https://www.together.ai/) (for demo only)
- [Ollama](https://ollama.ai/) (for experiments only)

### Installation

```bash
# Clone repository
git clone https://github.com/alexfdez1010/ner-llm
cd ner-llm

# Install dependencies
pip install -r requirements.txt
```

### Configuration

1. For the demo (app.py), set your Together AI API key:
   - Change the name of the file `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` file and set your API key:

   ```toml
   TOGETHER_API_KEY = "your-api-key"
   ```

2. For experiments (main.py), install Ollama:

   ```bash
   # Install Ollama (macOS/Linux)
   curl https://ollama.ai/install.sh | sh
   
   # Start Ollama service
   ollama serve
   ```

## 🎯 Usage

### Interactive Demo

The demo uses Together AI's models for real-time NER:

```bash
streamlit run app.py
```

Also, you can check the demo [here](https://ner-llm.streamlit.app).
Note: the LLM used in the demo has a rate limit of 6 requests per minute as it is a free endpoint.

### Running Experiments

The experimental pipeline uses Ollama models:

```bash
python main.py --model "deepseek-r1:7b" --dataset "multicardioner_track1"
```

Available models:

- deepseek-r1 (7B, 8B, 14B, 32B)
- phi3.5 (3.6B)
- granite3.1-dense (8B)
- falcon3 (10B)
- llama3.2-vision (11B)
- phi4 (14B)
- qwen2.5 (32B)

Supported datasets:

- MultiCardioNER Track 1
- PharmaCoNER
- MultiCardioNER Track 2 (English, Spanish, Italian)

## 📁 Project Structure

```text
ner-llm/
├── ai/                     # AI components
│   ├── extractor_ner.py   # NER extraction logic
│   ├── llm.py             # LLM integrations
│   └── prompts.py         # Prompt templates
├── datasets_info/          # Dataset definitions
├── model/                  # Data models
├── tests/                  # Test suite
├── app.py                 # Interactive demo
└── main.py               # Experimental pipeline
```

## 🔧 Extending the Framework

### Adding a New Dataset

1. Create a new dataset info class in `datasets_info/`:

```python
from datasets_info.dataset_info_interface import DatasetInfo

class NewDatasetInfo(DatasetInfo):
    def load_dataset(self) -> Dataset:
        """Load the dataset.
        
        Returns:
            Dataset: The loaded dataset
        """

    def categories(self) -> List[Category]:
        """Get the categories of the dataset.
        
        Returns:
            List[Category]: List of categories in the dataset
        """
    
    def language(self) -> str:
        """Get the language of the dataset.
        
        Returns:
            str: Language code of the dataset
        """
    
    def example_prompt(self) -> str:
        """Get an example prompt for the dataset.
        
        Returns:
            str: Example prompt for the dataset
        """
```

2. Register in `main.py`:

```python
DATASETS = {
    "new_dataset": ("datasets_info.new_dataset", "NewDataset"),
    # ...
}
```

### Adding a New Model

For experiments, add new Ollama models to `MODELS` in `main.py`:

```python
MODELS = [
    "your-new-model",
    # ...
]
```

## 📊 Results

Experiment results are saved to `results.csv` for analysis.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
