

### Retrieval-Augmented Generation (RAG) App

Welcome to the **Retrieval-Augmented Generation (RAG) App**! This Streamlit application allows users to upload various document types, extract their text, and generate responses using a Generative AI model integrated with a retrieval mechanism.

### Demo

You can try out the app live at:
- [Talk to Files Demo](https://talk-to-files.streamlit.app/)

## Features

- **Upload Files**: Supports `.txt`, `.pdf`, and `.docx` formats.
- **Text Extraction**: Extracts and processes text from uploaded files.
- **Text Embedding**: Utilizes a pre-trained BERT model for generating text embeddings.
- **Generative AI Queries**: Interact with the uploaded text using a Generative AI model.

## Installation

To run this app locally, follow these steps:

### Prerequisites

Make sure you have Python 3.7 or higher installed. You can check your Python version by running:
  
   
    python --version


### Clone the Repository

```bash
git clone https://github.com/yourusername/rag-app.git
cd rag-app
```

### Set Up a Virtual Environment

```bash
# Using conda
conda create -n myenv python=3.9
conda activate myenv

# OR Using venv
python -m venv myenv
source myenv/bin/activate  # On Windows use `myenv\Scripts\activate`
```

### Install Dependencies

```bash
pip install streamlit torch transformers PyPDF2 python-docx google-generativeai
```

## Usage

1. **Run the App**:

   ```bash
   streamlit run app.py
   ```
  ![RAG](https://github.com/user-attachments/assets/4e5f5765-aad0-4eca-bbb2-7fa73c0c9aa8)

2. **Upload a File**: Choose a `.txt`, `.pdf`, or `.docx` file to upload.

3. **Enter Your Query**: Type your query related to the uploaded text.

4. **Get Response**: Click the **Submit** button to see the response from the Generative AI model.

## Code Overview

- **Main Logic**: The main logic for text extraction and embedding is contained in `app.py`.
- **Model Loading**: The BERT model and tokenizer are loaded efficiently using Streamlit's caching features.

## Contributing

Feel free to fork the repository and submit pull requests. We welcome contributions to improve this project!

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgements

- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [Streamlit](https://streamlit.io/)
- [Google Generative AI](https://cloud.google.com/generative-ai)

## Contact

For any questions or feedback, feel free to reach out!

---

**Happy exploring!**
```

### Updates Made

1. **Project Name**: Changed to reflect it's a RAG implementation.
2. **Repository Name**: Updated to `rag-app`. Change this to your actual repository URL.

Save this content in a `README.md` file in your project directory. Let me know if you need any more changes!
