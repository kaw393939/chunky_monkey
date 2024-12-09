# Chunk Manager - Chunky Monkey

**Chunk Manager** is a powerful command-line interface (CLI) tool designed to efficiently process, manage, and manipulate large text documents by breaking them down into manageable chunks. This tool is ideal for tasks such as natural language processing (NLP), data analysis, and preparing documents for machine learning applications.

## Table of Contents

- [Chunk Manager - Chunky Monkey](#chunk-manager---chunky-monkey)
  - [Table of Contents](#table-of-contents)
  - [Features](#features)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Configuration](#configuration)
    - [Global Arguments](#global-arguments)
    - [Example Configuration](#example-configuration)
  - [Usage](#usage)
    - [Importing Documents](#importing-documents)
    - [Retrieving Documents](#retrieving-documents)
    - [Retrieving Chunks](#retrieving-chunks)
    - [Updating Documents](#updating-documents)
    - [Deleting Documents](#deleting-documents)
    - [Updating Chunks (Coming Soon)](#updating-chunks-coming-soon)
  - [Logging](#logging)
    - [Log Levels](#log-levels)
    - [Configuring Log Level](#configuring-log-level)
    - [Log Files](#log-files)
  - [Verification](#verification)
    - [Performing Verification](#performing-verification)
  - [Project Structure](#project-structure)
    - [Key Components](#key-components)
  - [Contributing](#contributing)
    - [Reporting Issues](#reporting-issues)
  - [License](#license)
  - [Acknowledgements](#acknowledgements)

## Features

- **Import Documents:** Easily import text documents from directories or individual files.
- **Chunking:** Automatically split large documents into smaller chunks based on token limits, ensuring efficient processing.
- **Metadata Management:** Maintain comprehensive metadata for documents and chunks, including token counts and content hashes.
- **Concurrent Processing:** Utilize asynchronous processing to handle multiple files concurrently, optimizing performance.
- **Verification:** Ensure data integrity through various verification modes (`strict`, `lenient`, `token`).
- **Retrieve & Update:** Access and modify individual documents and their respective chunks.
- **Delete:** Safely remove documents and their associated chunks from the system.
- **Logging:** Detailed logging to monitor processing activities and troubleshoot issues.

## Prerequisites

Before installing and using Chunk Manager, ensure you have the following prerequisites:

- **Python 3.8+**: Ensure Python is installed on your system. You can download it from [python.org](https://www.python.org/downloads/).
- **Virtual Environment (Recommended)**: It's advisable to use a virtual environment to manage dependencies.

## Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/chunk_manager.git
   cd chunk_manager
   ```

2. **Create a Virtual Environment**

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Download SpaCy Model**

   Chunk Manager uses SpaCy for NLP tasks. Download the required model:

   ```bash
   python -m spacy download en_core_web_sm
   ```

5. **Install the Package in Editable Mode**

   ```bash
   pip install -e .
   ```

6. **Verify Installation**

   Check if the `docprocess` command is available:

   ```bash
   docprocess --help
   ```

## Configuration

Chunk Manager is highly configurable through CLI arguments, allowing customization of its behavior to suit various processing needs.

### Global Arguments

- `--log-level`: Set the logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`). Default is `INFO`.
- `--model-name`: Specify the model name for token counting (`gpt-3.5`, `gpt-4`, `gpt-4-32k`, `claude`, `claude-2`, `gpt-4o`). Default is `gpt-4`.
- `--spacy-model`: SpaCy model name for NLP tasks. Default is `en_core_web_sm`.
- `--max-concurrent-files`: Maximum number of files to process concurrently. Default is `10`.
- `--chunk-reduction-factor`: Factor to reduce chunk size. Default is `1.0`.
- `--output-dir`, `--output`: Directory to store processed documents. Default is `output`.
- `--chunk-dir`: Directory to store document chunks. Default is `chunks`.

### Example Configuration

```bash
docprocess import \
    --input ./corpus \
    --output-dir ./data \
    --chunk-dir ./chunks \
    --verify \
    --verify-mode lenient \
    --chunk-reduction-factor 0.4
```

## Usage

Chunk Manager provides a suite of commands to manage documents and their chunks. Below are detailed explanations and examples of each command.

### Importing Documents

**Command:** `import`

**Description:** Import text documents from a specified directory or individual file, process them into chunks, and optionally verify their integrity.

**Usage:**

```bash
docprocess import \
    --input <INPUT_PATH> \
    --output-dir <OUTPUT_DIRECTORY> \
    --chunk-dir <CHUNK_DIRECTORY> \
    [--verify] \
    [--verify-mode {strict,lenient,token}] \
    [--chunk-reduction-factor <FLOAT>] \
    [--model-name <MODEL_NAME>] \
    [--spacy-model <SPACY_MODEL>] \
    [--max-concurrent-files <INT>] \
    [--log-level <LOG_LEVEL>]
```

**Parameters:**

- `--input`: **(Required)** Path to the input directory or file containing `.txt` documents.
- `--output-dir`: **(Optional)** Directory to store processed documents. Default is `output`.
- `--chunk-dir`: **(Optional)** Directory to store document chunks. Default is `chunks`.
- `--verify`: **(Optional)** Flag to perform verification after import.
- `--verify-mode`: **(Optional)** Mode of verification (`strict`, `lenient`, `token`). Default is `strict`.
- `--chunk-reduction-factor`: **(Optional)** Factor to reduce chunk size. Default is `1.0`.
- `--model-name`: **(Optional)** Model name for token counting. Default is `gpt-4`.
- `--spacy-model`: **(Optional)** SpaCy model for NLP tasks. Default is `en_core_web_sm`.
- `--max-concurrent-files`: **(Optional)** Max concurrent files to process. Default is `10`.
- `--log-level`: **(Optional)** Set logging level. Default is `INFO`.

**Example:**

```bash
docprocess import \
    --input ./corpus \
    --output-dir ./data \
    --chunk-dir ./chunks \
    --verify \
    --verify-mode lenient \
    --chunk-reduction-factor 0.4
```

### Retrieving Documents

**Command:** `get`

**Description:** Retrieve metadata of a specific document by its ID. Optionally, save the entire document content to a file.

**Usage:**

```bash
docprocess get \
    --doc-id <DOCUMENT_ID> \
    [--output-file <OUTPUT_FILE_PATH>] \
    [--log-level <LOG_LEVEL>]
```

**Parameters:**

- `--doc-id`: **(Required)** ID of the document to retrieve.
- `--output-file`: **(Optional)** Path to save the document content as a single text file.
- `--log-level`: **(Optional)** Set logging level. Default is `INFO`.

**Example:**

```bash
docprocess get \
    --doc-id 12345 \
    --output-file ./updated_document.txt
```

### Retrieving Chunks

**Command:** `get-chunk`

**Description:** Retrieve metadata of a specific chunk within a document. Optionally, save the chunk content to a file.

**Usage:**

```bash
docprocess get-chunk \
    --doc-id <DOCUMENT_ID> \
    --chunk-id <CHUNK_ID> \
    [--output-file <OUTPUT_FILE_PATH>] \
    [--log-level <LOG_LEVEL>]
```

**Parameters:**

- `--doc-id`: **(Required)** ID of the document containing the chunk.
- `--chunk-id`: **(Required)** ID of the chunk to retrieve.
- `--output-file`: **(Optional)** Path to save the chunk content as a text file.
- `--log-level`: **(Optional)** Set logging level. Default is `INFO`.

**Example:**

```bash
docprocess get-chunk \
    --doc-id 12345 \
    --chunk-id 12345-chunk-2 \
    --output-file ./updated_chunk_2.txt
```

### Updating Documents

**Command:** `update`

**Description:** Update the content of an existing document. This will re-chunk the document based on the new content and update associated metadata. Optionally, perform verification after the update.

**Usage:**

```bash
docprocess update \
    --doc-id <DOCUMENT_ID> \
    --input-file <NEW_CONTENT_FILE_PATH> \
    [--verify] \
    [--verify-mode {strict,lenient,token}] \
    [--output-dir <OUTPUT_DIRECTORY>] \
    [--chunk-dir <CHUNK_DIRECTORY>] \
    [--chunk-reduction-factor <FLOAT>] \
    [--log-level <LOG_LEVEL>]
```

**Parameters:**

- `--doc-id`: **(Required)** ID of the document to update.
- `--input-file`: **(Required)** Path to the file containing the new content for the document.
- `--verify`: **(Optional)** Flag to perform verification after updating.
- `--verify-mode`: **(Optional)** Mode of verification (`strict`, `lenient`, `token`). Default is `strict`.
- `--output-dir`: **(Optional)** Directory where processed documents are stored. Default is `output`.
- `--chunk-dir`: **(Optional)** Directory where document chunks are stored. Default is `chunks`.
- `--chunk-reduction-factor`: **(Optional)** Factor to reduce chunk size. Default is `1.0`.
- `--log-level`: **(Optional)** Set logging level. Default is `INFO`.

**Example:**

```bash
docprocess update \
    --doc-id 12345 \
    --input-file ./new_document_content.txt \
    --verify \
    --verify-mode strict \
    --output-dir ./data \
    --chunk-dir ./chunks \
    --chunk-reduction-factor 0.4
```

### Deleting Documents

**Command:** `delete`

**Description:** Delete a document and all its associated chunks from the system.

**Usage:**

```bash
docprocess delete \
    --doc-id <DOCUMENT_ID> \
    [--log-level <LOG_LEVEL>]
```

**Parameters:**

- `--doc-id`: **(Required)** ID of the document to delete.
- `--log-level`: **(Optional)** Set logging level. Default is `INFO`.

**Example:**

```bash
docprocess delete \
    --doc-id 12345
```

### Updating Chunks (Coming Soon)

**Command:** `update-chunk`

**Description:** Update the content of a specific chunk within a document. *(Note: This feature is under development.)*

**Usage:**

```bash
docprocess update-chunk \
    --doc-id <DOCUMENT_ID> \
    --chunk-id <CHUNK_ID> \
    --input-file <NEW_CHUNK_CONTENT_FILE_PATH> \
    [--verify] \
    [--verify-mode {strict,lenient,token}] \
    [--output-dir <OUTPUT_DIRECTORY>] \
    [--chunk-dir <CHUNK_DIRECTORY>] \
    [--log-level <LOG_LEVEL>]
```

**Parameters:**

- `--doc-id`: **(Required)** ID of the document containing the chunk.
- `--chunk-id`: **(Required)** ID of the chunk to update.
- `--input-file`: **(Required)** Path to the file containing the new content for the chunk.
- `--verify`: **(Optional)** Flag to perform verification after updating.
- `--verify-mode`: **(Optional)** Mode of verification (`strict`, `lenient`, `token`). Default is `strict`.
- `--output-dir`: **(Optional)** Directory where processed documents are stored. Default is `output`.
- `--chunk-dir`: **(Optional)** Directory where document chunks are stored. Default is `chunks`.
- `--log-level`: **(Optional)** Set logging level. Default is `INFO`.

**Example:**

```bash
docprocess update-chunk \
    --doc-id 12345 \
    --chunk-id 12345-chunk-2 \
    --input-file ./new_chunk_content.txt \
    --verify \
    --verify-mode strict \
    --output-dir ./data \
    --chunk-dir ./chunks
```

**Note:** The `update-chunk` command is currently under development and will be available in future releases.

## Logging

Chunk Manager utilizes Python's built-in `logging` module to provide detailed logs of its operations. Logs are output to both the console and log files stored in the `logs/` directory.

### Log Levels

- `DEBUG`: Detailed information, typically of interest only when diagnosing problems.
- `INFO`: Confirmation that things are working as expected.
- `WARNING`: An indication that something unexpected happened, or indicative of some problem.
- `ERROR`: Due to a more serious problem, the software has not been able to perform some function.

### Configuring Log Level

You can set the log level using the `--log-level` global argument. For example, to set the log level to `DEBUG`:

```bash
docprocess import \
    --input ./corpus \
    --output-dir ./data \
    --chunk-dir ./chunks \
    --log-level DEBUG
```

### Log Files

Logs are stored in the `logs/` directory within your project. Each run generates a new log file with a timestamp, ensuring that logs are organized and easily accessible for troubleshooting.

## Verification

After processing documents or chunks, you can optionally perform verification to ensure data integrity. Chunk Manager offers three verification modes:

- **Strict (`strict`)**
  - Verifies token counts and content hashes for all chunks.
  - Ensures that the total token count matches the document's metadata.
  
- **Lenient (`lenient`)**
  - Checks for the existence of all expected chunks.
  - Does not verify token counts or content hashes.
  
- **Token-only (`token`)**
  - Verifies the total token count of the document without checking individual chunks.
  
### Performing Verification

Use the `--verify` flag along with the `--verify-mode` argument in commands that support verification (`import`, `update`, `update-chunk`).

**Example:**

```bash
docprocess import \
    --input ./corpus \
    --output-dir ./data \
    --chunk-dir ./chunks \
    --verify \
    --verify-mode strict
```

## Project Structure

A well-organized project structure enhances maintainability and scalability. Here's an overview of Chunk Manager's structure:

```
chunk_manager/
│
├── venv/                       # Virtual environment directory
│
├── src/
│   └── document_processor/
│       ├── __init__.py
│       │
│       ├── cli/
│       │   └── commands.py      # CLI command implementations
│       │
│       ├── core/
│       │   ├── __init__.py
│       │   ├── models.py        # Data models using Pydantic
│       │   ├── processor.py     # Core processing logic
│       │   └── verifier.py      # Verification logic
│       │
│       ├── services/
│       │   ├── __init__.py
│       │   ├── chunk_service.py    # Managing chunk-related operations
│       │   ├── document_service.py # Managing document-related operations
│       │   └── metadata_service.py # Managing metadata operations
│       │
│       └── utils/
│           ├── __init__.py
│           ├── config.py          # Configuration management
│           └── logging.py         # Logging setup
│
├── setup.py                    # Package setup script
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
└── logs/                       # Directory to store log files
```

### Key Components

- **CLI (`cli/commands.py`):** Implements the command-line interface, handling user inputs and mapping commands to functionalities.
- **Core (`core/`):**
  - **Models (`models.py`):** Defines data structures using Pydantic for data validation and serialization.
  - **Processor (`processor.py`):** Contains the core logic for processing documents, chunking, and managing metadata.
  - **Verifier (`verifier.py`):** Implements verification processes to ensure data integrity.
- **Services (`services/`):**
  - **Chunk Service (`chunk_service.py`):** Manages operations related to document chunks.
  - **Document Service (`document_service.py`):** Handles operations related to entire documents.
  - **Metadata Service (`metadata_service.py`):** Manages metadata creation and updates.
- **Utils (`utils/`):**
  - **Configuration (`config.py`):** Manages application configuration settings.
  - **Logging (`logging.py`):** Sets up and configures logging mechanisms.

## Contributing

Contributions are welcome! To contribute to Chunk Manager, please follow these steps:

1. **Fork the Repository**

   Click the "Fork" button at the top-right corner of the repository page to create a personal copy.

2. **Clone Your Fork**

   ```bash
   git clone https://github.com/yourusername/chunk_manager.git
   cd chunk_manager
   ```

3. **Create a Virtual Environment**

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

5. **Make Your Changes**

   - Implement new features or fix bugs.
   - Ensure code follows PEP 8 standards.

6. **Run Tests**

   *(If tests are implemented)*

   ```bash
   pytest
   ```

7. **Commit Your Changes**

   ```bash
   git add .
   git commit -m "Description of your changes"
   ```

8. **Push to Your Fork**

   ```bash
   git push origin main
   ```

9. **Create a Pull Request**

   Go to the original repository and create a pull request detailing your changes.

### Reporting Issues

If you encounter any issues or have feature requests, please open an issue on the [GitHub Issues](https://github.com/yourusername/chunk_manager/issues) page.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgements

- [Pydantic](https://pydantic-docs.helpmanual.io/): Data validation and settings management using Python type annotations.
- [SpaCy](https://spacy.io/): Industrial-strength Natural Language Processing (NLP) library.
- [tiktoken](https://github.com/openai/tiktoken): Tokenizer for OpenAI's models.
- [Argparse](https://docs.python.org/3/library/argparse.html): Parser for command-line options, arguments, and sub-commands.
- [Aiofiles](https://github.com/Tinche/aiofiles): Async file support for Python.

---

Feel free to reach out if you have any questions or need further assistance with Chunk Manager!