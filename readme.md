# ChatBot Project

This repository contains a chatbot application designed to process and answer user queries using advanced language models and vector-based retrieval systems. The project leverages the LangChain framework, HuggingFace models, and FAISS for efficient document retrieval and embedding storage.

## Features

- **PDF Document Processing**: Extracts and processes text from PDF files.
- **Text Chunking**: Splits large documents into manageable chunks for better embedding and retrieval.
- **Vector Embedding Storage**: Uses FAISS to store and retrieve vector embeddings.
- **Customizable LLM Integration**: Supports HuggingFace models for generating responses.
- **Streamlit UI**: Provides an interactive web-based interface for user interaction.
- **Medical Chatbot**: A specialized chatbot for answering medical queries with context-based responses.

## Project Structure

- `create_memory_for_llm.py`: Script to process PDF files, create text chunks, and store embeddings in FAISS.
- `create_memory_with_llm.py`: Script to integrate the FAISS database with a HuggingFace language model for query answering.
- `medibot.py`: Streamlit-based application for interacting with the chatbot.
- `test.py`: Script to verify HuggingFace authentication.
- `.gitignore`: Specifies files and directories to ignore in version control.
- `Pipfile`: Dependency management file for the project.

## Setup Instructions

1. Clone the repository:
    ```bash
    git clone <repository-url>
    cd ChatBot
    ```

2. Install dependencies using `pipenv`:
    ```bash
    pipenv install
    ```

3. Create a `.env` file and add your HuggingFace token:
    ```
    HF_TOKEN=<your_huggingface_token>
    ```

4. Place your PDF files in the `data/` directory.

5. Run the scripts:
    - To process documents and create embeddings:
      ```bash
      pipenv run python create_memory_for_llm.py
      ```
    - To interact with the chatbot via Streamlit:
      ```bash
      pipenv run streamlit run medibot.py
      ```

## Requirements

- Python 3.13
- HuggingFace account and API token
- Dependencies listed in the `Pipfile`

## Usage

- Use the `medibot.py` application to ask questions and receive context-aware responses.
- Ensure the FAISS database is populated with embeddings before running the chatbot.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments

- [LangChain](https://github.com/hwchase17/langchain)
- [HuggingFace](https://huggingface.co/)
- [FAISS](https://github.com/facebookresearch/faiss)
