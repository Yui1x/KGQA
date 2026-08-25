# SCEPS: Slot-complete evidence path search for hazardous waste knowledge graph question answering

SCEPS is a knowledge graph question answering framework for hazardous-waste safety information. It searches a Neo4j knowledge graph, checks whether the retrieved evidence covers the required information slots, and abstains when the evidence is insufficient. With Qwen3-Max, SCEPS achieved 89.75% single-hop accuracy, 44.00% multi-hop accuracy, and a 5.33% hallucination rate in the experiments reported in the accompanying manuscript.

## Repository structure

- `Data/`: graph data before and after tail-entity clustering. `data_after_cluster.zip` is used for Neo4j import.
- `KG_Construction/`: preprocessing, triple extraction, entity clustering, and Neo4j import scripts.
- `KG_QA/`: SCEPS graph search and question-answering code.
- `Question&Answer/`: 400 single-hop, 150 multi-hop, and 150 hallucination-control questions with reference answers.
- `Result/Ablation/`: archived outputs and evaluation results for the ablation experiments.

## Requirements and installation

The project requires Python 3.12, Neo4j 5.x with the APOC plugin, and access to an OpenAI-compatible chat-completions API.

Clone the repository:

```bash
git clone https://github.com/Yui1x/KGQA.git
cd KGQA
```

Conda is optional. To use the supplied environment file:

```bash
conda env create -f environment.yml
conda activate kgqa
```

The project can also be installed in a standard Python virtual environment. Create and activate the environment first:

```bash
python -m venv .venv
```

On Windows use `.venv\Scripts\Activate.ps1`; on Linux or macOS use `source .venv/bin/activate`. Then install the dependencies:

```bash
python -m pip install numpy torch rdkit scikit-learn beautifulsoup4 neo4j openai pubchempy requests sentence-transformers urllib3
```

## Configuration

Set the following environment variables before running SCEPS:

```text
KGQA_API_BASE
KGQA_API_KEY
KGQA_MODEL
KGQA_ENTITY_MODEL        # optional; defaults to KGQA_MODEL
KGQA_NEO4J_URI
KGQA_NEO4J_USER
KGQA_NEO4J_PASSWORD
```

`.env.example` lists the same variables, but the current code does not load `.env` automatically. Do not commit API keys or database passwords.

## Usage

Extract the final graph data:

```bash
unzip Data/data_after_cluster.zip -d data_after_cluster
```

Windows users may run:

```powershell
Expand-Archive -Path Data\data_after_cluster.zip -DestinationPath data_after_cluster
```

After starting Neo4j and enabling APOC, import the graph:

```bash
python KG_Construction/import_to_neo4j.py --data-dir data_after_cluster
```

Run a single question:

```bash
python KG_QA/main.py --single_question "乙酸丙酯的中文别名有什么？" --output_dir output/test
```

Run all evaluation splits:

```bash
python KG_QA/main.py --input_csv "Question&Answer" --input_split all --samples 99999 --output_dir output/main
```

The program writes JSON and CSV results to the selected output directory. Run `python KG_QA/main.py --help` to view the search and generation parameters.

## Data and methodology

`Data/data_after_cluster.zip` contains 2,883 chemical-property records and 196,008 relationship records. The evaluation CSV files use the columns `Questions` and `Answers`.

The workflow includes data preprocessing, relation extraction, entity clustering, Neo4j import, and evidence-path search for question answering.
