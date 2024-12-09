# Chunky Monkey

<img src="chunkymonkey.webp" alt="Chunky Monkey" width="100"/>

**Chunky Monkey** is a powerful command-line tool designed to process and manage large text documents by splitting them into manageable chunks based on token limits. Leveraging advanced tokenization libraries like [tiktoken](https://github.com/openai/tiktoken) and [spaCy](https://spacy.io/), Chunky Monkey ensures efficient and accurate text processing, making it ideal for applications in natural language processing, machine learning, and data analysis.

## Table of Contents

- [Chunky Monkey](#chunky-monkey)
  - [Table of Contents](#table-of-contents)
  - [Features](#features)
  - [Installation](#installation)
    - [Prerequisites](#prerequisites)
    - [Steps](#steps)
  - [Quick Start](#quick-start)
  - [Usage](#usage)
    - [Import Command](#import-command)
      - [Syntax](#syntax)
      - [Global Options](#global-options)
      - [Import Options](#import-options)
      - [Example Command](#example-command)
  - [Configuration](#configuration)
    - [Example `MODEL_CONFIGS` Entry](#example-model_configs-entry)
  - [Output Directory Structure](#output-directory-structure)
  - [Logging and Progress Tracking](#logging-and-progress-tracking)
    - [Logging Levels](#logging-levels)
    - [Progress Tracking](#progress-tracking)
  - [Troubleshooting](#troubleshooting)
    - [Common Issues](#common-issues)
    - [Steps to Resolve Issues](#steps-to-resolve-issues)
  - [Contributing](#contributing)
    - [How to Contribute](#how-to-contribute)
    - [Code of Conduct](#code-of-conduct)
  - [License](#license)

## Features

- **Efficient Document Importing:** Seamlessly import large `.txt` files and automatically split them into manageable chunks.
- **Customizable Chunking:** Adjust chunk sizes using the `--chunk-reduction-factor` to fit your specific token limits.
- **Progress Tracking:** Real-time monitoring of processing progress, including files processed, chunks created, and tokens counted.
- **Robust Error Handling:** Gracefully handles empty or malformed files without interrupting the entire import process.
- **Comprehensive Logging:** Detailed logs for easy debugging and monitoring of processing activities.
- **Metadata Management:** Maintains detailed metadata for each document and chunk, facilitating easy retrieval and management.

## Installation

### Prerequisites

- **Python 3.8 or higher** is required.
- **pip** package manager.

### Steps

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/yourusername/chunky_monkey.git
   cd chunky_monkey
   ```

2. **Create a Virtual Environment (Optional but Recommended):**

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies:**

   ```bash
   pip install -e .
   ```

   **Note:** The `-e` flag installs the package in editable mode, allowing you to make changes to the source code without reinstalling.

## Quick Start

To quickly import documents from a directory, use the following command:

```bash
chunky_monkey --chunk-reduction-factor 0.2 --model-name gpt-4-32k --output-dir ./data import --input-dir ./corpus
```

This command will process all `.txt` files in the `./corpus` directory, split them into chunks based on the specified `chunk-reduction-factor`, and store the processed data in the `./data` directory.

## Usage

### Import Command

The `import` command is the primary functionality of Chunky Monkey, allowing you to ingest and process text documents.

#### Syntax

```bash
chunky_monkey [GLOBAL OPTIONS] import --input-dir <INPUT_DIR> [IMPORT OPTIONS]
```

#### Global Options

- `--log-level`: Set the logging level. Choices are `DEBUG`, `INFO`, `WARNING`, `ERROR`. Default is `INFO`.
  
  ```bash
  --log-level DEBUG
  ```

- `--model-name`: Specify the model name to use for token encoding. Choices are defined in the `MODEL_CONFIGS`. Default is `gpt-4`.
  
  ```bash
  --model-name gpt-4-32k
  ```

- `--spacy-model`: Specify the SpaCy model name for NLP tasks. Default is `en_core_web_sm`.
  
  ```bash
  --spacy-model en_core_web_sm
  ```

- `--max-concurrent-files`: Maximum number of files to process concurrently. Default is `10`.
  
  ```bash
  --max-concurrent-files 5
  ```

- `--output-dir`: Directory to store processed documents. Default is `output`.
  
  ```bash
  --output-dir ./data
  ```

- `--chunk-reduction-factor`: Factor to reduce chunk size. Must be greater than `0` and at most `1.0`. Default is `1.0`.
  
  ```bash
  --chunk-reduction-factor 0.2
  ```

#### Import Options

- `--input-dir`: **(Required)** Input directory containing `.txt` documents to import.
  
  ```bash
  --input-dir ./corpus
  ```

- `--verify`: Enable verification after import. Optional flag.
  
  ```bash
  --verify
  ```

- `--verify-mode`: Specify the verification mode. Choices are `strict`, `lenient`, `token`. Default is `strict`.
  
  ```bash
  --verify-mode lenient
  ```

#### Example Command

```bash
chunky_monkey --log-level DEBUG --chunk-reduction-factor 0.2 --model-name gpt-4-32k --output-dir ./data import --input-dir ./corpus --verify --verify-mode lenient
```

This command will:

1. Set the logging level to `DEBUG` for detailed logs.
2. Reduce the chunk size by a factor of `0.2`.
3. Use the `gpt-4-32k` model for token encoding.
4. Store processed data in the `./data` directory.
5. Import all `.txt` files from the `./corpus` directory.
6. Perform verification after import using `lenient` mode.

## Configuration

Chunky Monkey can be configured using command-line arguments as shown above. Additionally, you can modify the `MODEL_CONFIGS` in the `utils/config.py` file to add or adjust model-specific configurations such as token limits and encodings.

### Example `MODEL_CONFIGS` Entry

```python
MODEL_CONFIGS = {
    "gpt-4-32k": {
        "tokens": 32768,
        "encoding": "cl100k_base"  # Ensure this matches the model's encoding
    },
    "gpt-3.5-turbo": {
        "tokens": 4096,
        "encoding": "cl100k_base"
    },
    # Add other models as needed
}
```

## Output Directory Structure

After importing, the `./data` directory (or your specified `--output-dir`) will have the following structure:

```
./data/
├── <doc_id_1>/
│   ├── source/
│   │   └── byzantium.txt
│   ├── chunks/
│   │   ├── <doc_id_1>-chunk-1.txt
│   │   ├── <doc_id_1>-chunk-1.json
│   │   ├── <doc_id_1>-chunk-2.txt
│   │   ├── <doc_id_1>-chunk-2.json
│   │   └── ...
│   ├── document_info.json
│   └── processing_state.json
├── <doc_id_2>/
│   ├── source/
│   │   └── egypt.txt
│   ├── chunks/
│   │   ├── <doc_id_2>-chunk-1.txt
│   │   ├── <doc_id_2>-chunk-1.json
│   │   └── ...
│   ├── document_info.json
│   └── processing_state.json
├── manifest.json
└── processing_<id>.json
```

- **`<doc_id_x>/`:** Unique directory for each processed document.
  - **`source/`:** Contains the original `.txt` file.
  - **`chunks/`:** Contains both the text and JSON metadata for each chunk.
  - **`document_info.json`:** Metadata about the document.
  - **`processing_state.json`:** Tracks the processing state (complete or incomplete).
- **`manifest.json`:** Aggregated manifest of all processed documents.
- **`processing_<id>.json`:** Metadata tracking the processing status of each document.

## Logging and Progress Tracking

Chunky Monkey provides comprehensive logging to help you monitor and debug the import process.

### Logging Levels

- **DEBUG:** Detailed information, typically of interest only when diagnosing problems.
- **INFO:** Confirmation that things are working as expected.
- **WARNING:** An indication that something unexpected happened, or indicative of some problem in the near future.
- **ERROR:** Due to a more serious problem, the software has not been able to perform some function.

### Progress Tracking

A `ProcessingProgress` instance monitors:

- **Total Files:** Total number of `.txt` files to process.
- **Processed Files:** Number of files successfully processed.
- **Current File:** The file currently being processed.
- **Processed Chunks:** Total number of chunks created across all files.
- **Total Tokens:** Total number of tokens across all chunks.
- **Elapsed Time:** Time elapsed since the start of the import process.

**Sample Progress Log:**

```plaintext
2024-12-09 05:26:36,252 [INFO] document_processor.core.processor: Processed 1/4 files. Current file: byzantium.txt. Processed chunks: 5. Total tokens: 1250. Elapsed time: 0:00:05
2024-12-09 05:26:36,388 [INFO] document_processor.core.processor: Processed 2/4 files. Current file: carthage.txt. Processed chunks: 10. Total tokens: 2500. Elapsed time: 0:00:10
```

## Troubleshooting

### Common Issues

1. **Chunks and Tokens Reported as 0:**
   
   - **Cause:** Improper progress tracking or issues during chunking/token counting.
   - **Solution:**
     - Ensure that input `.txt` files are not empty and contain valid text.
     - Verify that the `--chunk-reduction-factor` is set correctly (greater than `0` and at most `1.0`).
     - Run the import command with `--log-level DEBUG` to inspect detailed logs.

2. **JSON Parsing Errors (`Expecting value: line 1 column 1 (char 0)`):**
   
   - **Cause:** Attempting to parse empty or malformed JSON files.
   - **Solution:**
     - Check the problematic `.json` files in the output directory for content.
     - Ensure that the import process is writing JSON files correctly without interruptions.
     - Verify that the input `.txt` files are properly formatted and not empty.

3. **File Processing Failures:**
   
   - **Cause:** Issues copying files, writing chunks, or updating metadata.
   - **Solution:**
     - Ensure that the output directory has the necessary write permissions.
     - Check disk space availability.
     - Inspect logs for specific error messages related to file operations.

### Steps to Resolve Issues

1. **Review Logs:**
   
   - Run the import command with `--log-level DEBUG` to capture detailed logs.
   - Identify where the process is failing by examining error messages.

2. **Validate Input Files:**
   
   - Ensure that all `.txt` files in the input directory are non-empty and properly formatted.
   - Remove or fix any files that are causing parsing errors.

3. **Inspect Output Directory:**
   
   - Check for incomplete or empty JSON files.
   - Ensure that chunk files are being created correctly with corresponding metadata.

4. **Reinstall and Clear Caches:**
   
   - If issues persist, consider reinstalling the package and clearing Python caches.
   
   ```bash
   # Navigate to the project root
   cd ~/projects/chunk_manager
   
   # Remove __pycache__ directories and .pyc files
   find . -type d -name "__pycache__" -exec rm -r {} +
   find . -type f -name "*.pyc" -delete
   
   # Uninstall and reinstall the package
   pip uninstall chunky_monkey -y
   pip install -e .
   ```

5. **Seek Support:**
   
   - If problems continue, consider reaching out for support by opening an issue on the [GitHub repository](https://github.com/yourusername/chunky_monkey/issues).

## Contributing

Contributions are welcome! Whether it's bug fixes, new features, or documentation improvements, your help is appreciated.

### How to Contribute

1. **Fork the Repository:**

   Click the "Fork" button on the [GitHub repository](https://github.com/yourusername/chunky_monkey) page.

2. **Clone Your Fork:**

   ```bash
   git clone https://github.com/yourusername/chunky_monkey.git
   cd chunky_monkey
   ```

3. **Create a New Branch:**

   ```bash
   git checkout -b feature/your-feature-name
   ```

4. **Make Your Changes:**

   Implement your feature or fix.

5. **Commit Your Changes:**

   ```bash
   git commit -m "Add feature XYZ"
   ```

6. **Push to Your Fork:**

   ```bash
   git push origin feature/your-feature-name
   ```

7. **Open a Pull Request:**

   Go to the original repository and open a pull request detailing your changes.

### Code of Conduct

Please adhere to the [Contributor Covenant Code of Conduct](https://www.contributor-covenant.org/version/2/0/code_of_conduct/) in all interactions.

## License

This project is licensed under the [MIT License](LICENSE).

---

*Developed with ❤️ by [Keith Williams](https://github.com/kaw393939).*