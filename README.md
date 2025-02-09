# Using Local LLM for the analyis of socio-ecologic conflicts

This repository contains the following files:

- `RAG_set_up.py`: Script to set up the Retrieval-Augmented Generation (RAG) model.
- `streamlit_app.py`: Streamlit application to interact with the RAG model.
- `RAG_from_zero.ipynb`: Jupyter notebook demonstrating the RAG model from scratch.

## Context
Oftentimes in socio-ecologic conflicts there is a huge amount of legal documentation on the case that had to be made public at one point or another. This repository was developed for the analysis of >600 legal documents concerning the case of Caño Limón, an oil field with ~350 active oil wells and two production facilities in Arauca, Colombia.
As the documents contain personal data, the idea was to build a local RAG-Pipeline for the analysis of the documents. Due to the limitation of my personal Laptop a relatively small LLM was used for embedding ("all-MiniLM-L12-v2") and for retrieving ("Deepseek-R1:14b"). I also tested with "Llama3.2:latest" for retrieving but got better results using the Deepseek-Modell.
## Getting Started

### Prerequisites

- Python 3.7 or higher
- Required Python packages (listed in `requirements.txt`)

### Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/kurtenbl/LIPA.git
    ```
2. Navigate to the project directory:
    ```sh
    cd your-repo-name
    ```
3. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

### Setting Up the RAG Model

Run the `RAG_set_up.py` script to set up the RAG model:
```sh
python RAG_set_up.py
```

### Running the Streamlit App

Launch the Streamlit application to interact with the RAG model:
```sh
streamlit run streamlit_app.py
```

### Exploring the Jupyter Notebook

Open and run the `RAG_from_zero.ipynb` notebook to explore the RAG model from scratch.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

Feel free to use the code for your purposes!

