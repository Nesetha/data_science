# PrenatalAI

**PrenatalAI** is a healthcare AI research prototype combining prenatal ultrasound anatomical-plane classification with a Retrieval-Augmented Generation (RAG) assistant for selected medical research literature.

> **Important limitation:** This is a research prototype, not a medical device. The computer-vision model classifies anatomical planes; it does not detect or diagnose Spina Bifida or any other condition. Outputs must not be used for clinical decision-making.

## 1. Project Overview

The application contains two main workflows:

1. **Ultrasound Classification:** A PyTorch MobileNetV2 model classifies uploaded ultrasound images into six anatomical-plane categories.
2. **Medical Research Assistant:** A RAG pipeline retrieves relevant passages from research PDFs using MongoDB Atlas Vector Search and generates answers using Llama 3.2 3B through Ollama.

A Streamlit dashboard connects the workflows and provides case-history and research-assistant interfaces.

## 2. Features

- Upload prenatal ultrasound images.
- Display predicted anatomical-plane category, confidence, and class probabilities.
- Store analysis metadata in MongoDB Atlas.
- Ask questions about the ingested research documents.
- Retrieve semantically relevant passages using MongoDB Atlas Vector Search.
- Generate answers using a locally hosted Llama 3.2 3B model.
- Display source-document and page references.
- Apply basic input, response-phrase, citation-format, and fallback checks.

## 3. Dataset

The computer-vision model uses **FETAL_PLANES_DB**, containing approximately **12,400 ultrasound images from 1,792 patients**.

### Six classification categories

- Other
- Maternal cervix
- Fetal abdomen
- Fetal brain
- Fetal femur
- Fetal thorax

The task is multi-class anatomical-plane classification, not disease diagnosis.

The training workflow used an official patient-level split to reduce patient leakage, and training was performed for 8 epochs in Google Colab using an NVIDIA Tesla T4 GPU.

The dataset is not included in this repository. Obtain it from its official distribution source and follow its terms of use.

## 4. Deep Learning Model – MobileNetV2

The ultrasound classification component uses MobileNetV2, a lightweight convolutional neural network architecture.

### Why MobileNetV2?

- Depthwise separable convolutions reduce computational cost.
- Inverted residual blocks improve feature extraction efficiency.
- Linear bottlenecks help preserve useful information.
- Suitable for computationally efficient image classification.

### Model configuration

| Parameter | Details |
|---|---|
| Architecture | MobileNetV2 |
| Framework | PyTorch |
| Dataset | FETAL_PLANES_DB |
| Number of classes | 6 |
| Training environment | Google Colab |
| GPU | NVIDIA Tesla T4 |
| Training duration | 8 epochs |
| Saved checkpoint | `models/anatomical_plane_mobilenetv2.pth` |

### Model Evaluation Results

| Metric | Result |
|---|---:|
| Test Accuracy | 82.26% |
| Balanced Accuracy | 78.85% |
| Macro F1-score | 79.34% |

Accuracy reports overall correct predictions. Balanced accuracy averages class-wise recall, while macro F1 averages the class-wise F1-scores with equal weight for each class.

These metrics describe anatomical-plane classification only. They do not measure Spina Bifida diagnostic performance. Confidence scores are model outputs, not clinical certainty.

## 5. RAG-Based Medical Research Assistant

The second major component is a Retrieval-Augmented Generation system that allows users to ask questions about selected prenatal imaging and spinal dysraphism research documents.

### Research documents

The current research corpus contains:

1. `Antenatal_Spinal_Dysraphisms_2025.pdf`
2. `ISUOG_CNS_Guidelines_2020.pdf`

### RAG Implementation

The pipeline follows these steps:

1. **PDF Extraction:** Extracts text from research documents using PyPDF.
2. **Text Chunking:** Splits extracted text into manageable chunks.
3. **Embedding Generation:** Converts text chunks into numerical vector representations.
4. **Vector Storage:** Stores embeddings and metadata in MongoDB Atlas.
5. **Semantic Retrieval:** Retrieves relevant passages based on the user's question.
6. **Context Construction:** Combines retrieved passages into a context for the language model.
7. **Response Generation:** Uses Llama 3.2 3B through Ollama to generate research-grounded answers.
8. **Source References:** Displays document names and page information.

### RAG Configuration

| Component | Configuration |
|---|---|
| PDF Extraction | PyPDF |
| Text Splitter | RecursiveCharacterTextSplitter |
| Chunk Size | 800 characters |
| Chunk Overlap | 120 characters |
| Embedding Model | `sentence-transformers/all-MiniLM-L6-v2` |
| Embedding Dimensions | 384 |
| Vector Database | MongoDB Atlas |
| Vector Index | `prenatalai_vector_index` |
| Similarity Metric | Cosine |
| Language Model | Llama 3.2 3B |
| Local Inference | Ollama |

### Document Ingestion Results

| Research Document | Chunks |
|---|---:|
| Antenatal Spinal Dysraphisms (2025) | 45 |
| ISUOG CNS Guidelines (2020) | 63 |
| **Total** | **108** |

The pipeline successfully processed two research documents and stored 108 searchable chunks in MongoDB Atlas.

RAG can improve grounding in the selected documents, but retrieval and citation display do not guarantee that every generated statement is supported by its cited passage. Answers should be checked against the original research.

## 6. System Architecture

### Ultrasound Classification Workflow

```text
Ultrasound Image
       |
       v
Image Preprocessing
       |
       v
MobileNetV2 Model (PyTorch)
       |
       v
Six-Class Anatomical-Plane Prediction
       |
       v
Predicted Class + Confidence + Probabilities
       |
       v
Optional Case-History Storage (MongoDB Atlas)
```

### Research Assistant Workflow

```text
Research PDFs
       |
       v
PDF Text Extraction
       |
       v
Text Chunking
(800 characters, 120 overlap)
       |
       v
Embedding Generation
(all-MiniLM-L6-v2, 384 dimensions)
       |
       v
MongoDB Atlas Vector Search
       |
       v
User Question and Semantic Retrieval
       |
       v
Retrieved Context and Source Labels
       |
       v
Llama 3.2 3B through Ollama
       |
       v
Basic Safety and Citation Checks
       |
       v
Generated Answer with Source References
```

## 7. Safety and Guardrail Implementation

Since PrenatalAI operates in a healthcare-related domain, basic safeguards were implemented to reduce unsafe outputs.

### Implemented safeguards

**1. Input Validation**
- Checks for selected patient-specific diagnosis requests.
- Blocks selected treatment and medication recommendation requests.
- Provides a message directing users to qualified healthcare professionals.

**2. Response Validation**
- Checks generated responses for predefined unsafe diagnostic phrases.
- Blocks responses containing flagged wording.

**3. Citation Validation**
- Checks whether citation labels exist in the retrieved context.
- Rejects unsupported citation formats, such as `[2]` instead of `[S2]`.

**4. Fallback Safety Check**
- Applies the response checker to fallback responses when the LLM is unavailable.

**5. Grounding Instructions**
- Instructs the LLM to use retrieved research passages.
- Avoids unsupported medical claims and invented statistics.
- Requires acknowledgement when evidence is insufficient.
- Prohibits diagnosis and treatment recommendations.

### Safety Limitations

These are prototype-level safeguards, not a comprehensive medical safety system. Citation validation checks format and label existence, not whether a source semantically supports a claim. The guardrails have not been clinically validated.

A response passing these checks must not be interpreted as medically safe or clinically correct.

## 8. Database Implementation – MongoDB Atlas

MongoDB Atlas is used for two major purposes.

### Research Document Storage
- Stores extracted research text chunks.
- Stores source-document and page metadata.
- Stores 384-dimensional embeddings.
- Supports semantic retrieval through Atlas Vector Search.

### Case History Storage
- Stores ultrasound analysis records.
- Stores prediction results and associated metadata.
- Allows users to revisit previously processed cases.

### Vector Search Configuration

| Setting | Value |
|---|---|
| Index Name | `prenatalai_vector_index` |
| Vector Field | `embedding` |
| Dimensions | 384 |
| Similarity | Cosine |

The vector index configuration must match the embedding model used to generate the stored vectors.

## 9. Frontend and Application Integration

The application is developed using Streamlit.

### Dashboard Modules

- **Overview:** Main project dashboard and navigation.
- **Ultrasound Workspace:** Upload ultrasound images and view anatomical-plane predictions.
- **Medical Research Assistant:** Ask questions about the ingested research papers.
- **Research Library:** Access the research knowledge base.
- **Case History:** View stored ultrasound analysis records.
- **Evaluation:** Access model evaluation information.

The frontend integrates the computer-vision model, RAG pipeline, and MongoDB database into a single application.

## 10. Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Computer Vision | PyTorch, MobileNetV2 |
| Image Processing | PIL, torchvision |
| Dataset | FETAL_PLANES_DB |
| PDF Processing | PyPDF |
| Text Chunking | LangChain Text Splitter |
| Embeddings | Sentence Transformers |
| Vector Database | MongoDB Atlas Vector Search |
| Large Language Model | Llama 3.2 3B |
| Local Inference | Ollama |
| Frontend | Streamlit |
| Training Environment | Google Colab |
| Development Tools | VS Code |

## 11. Repository Structure

```text
PrenatalAI/
│
├── backend/
│
├── frontend/
│   └── app.py
│
├── src/
│   ├── models/
│   │   └── predict.py
│   ├── safety.py
│   └── database.py
│
├── rag/
│   ├── ingest.py
│   ├── retrieval.py
│   ├── vector_store.py
│   └── knowledge_base.py
│
├── research/
│   ├── Antenatal_Spinal_Dysraphisms_2025.pdf
│   └── ISUOG_CNS_Guidelines_2020.pdf
│
├── models/
│   └── anatomical_plane_mobilenetv2.pth
│
├── data/
│
├── test_retrieval.py
├── .env
└── README.md
```

The structure above describes the main project components. Local folders may contain additional files.

Do not commit secrets, private datasets, or credentials.

## 12. Installation and Setup

### Prerequisites

- Python 3.11
- Git
- MongoDB Atlas account with Vector Search enabled
- Ollama installed locally
- Trained model checkpoint
- FETAL_PLANES_DB dataset if reproducing model training

### Step 1: Clone the Repository

```bash
git clone <your-repository-url>
cd PrenatalAI
```

### Step 2: Create a Virtual Environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Step 3: Install Dependencies

If the repository contains a `requirements.txt` file:

```bash
pip install -r requirements.txt
```

### Step 4: Configure Environment Variables

Create a `.env` file in the project root.

```env
MONGODB_URI=<your-mongodb-atlas-connection-string>
MONGODB_DATABASE=prenatalai_db

LLM_PROVIDER=ollama
OLLAMA_MODEL=llama3.2:3b
```

Use your own MongoDB credentials.

**Security:** Never commit `.env` or share database connection strings. Configure MongoDB Atlas Network Access to permit connections from your development environment.

### Step 5: Configure MongoDB Vector Search

Create a vector search index named:

```text
prenatalai_vector_index
```

Configure it with:

- Vector field: `embedding`
- Dimensions: `384`
- Similarity: `cosine`

### Step 6: Set Up Ollama

Ensure Ollama is installed and running.

Download the configured language model:

```bash
ollama pull llama3.2:3b
```

### Step 7: Ingest Research Documents

Place the permitted research PDFs inside the `research/` folder.

Run the ingestion script from the project root:

```bash
python -m rag.ingest
```

The ingestion process extracts, chunks, embeds, and stores the research content in MongoDB Atlas.

Avoid repeated ingestion unless the ingestion workflow is configured to replace or deduplicate existing records.

### Step 8: Run the Streamlit Application

From the project root:

```bash
python -m streamlit run frontend/app.py
```

Open the local URL printed by Streamlit in your browser.

## 13. Testing

Run the retrieval test:

```bash
python test_retrieval.py
```

When MongoDB Atlas is available and correctly configured, the expected retrieval mode is:

```text
mongodb vector search
```

The test output reports:
- Retrieved research chunks.
- Generated answers.
- Source-document information.
- Answer generation mode.

Check syntax for key modules:

```bash
python -m py_compile rag/retrieval.py
python -m py_compile src/safety.py
```

A successful test run verifies execution of the tested workflow. It does not establish clinical validity or guarantee factual correctness.

## 14. Future Improvements

Potential future enhancements include:

- Evaluate per-class precision, recall, F1-score, and confusion matrix.
- Assess robustness across scanners, institutions, image quality, and patient populations.
- Improve retrieval quality and evaluate retrieval relevance systematically.
- Add semantic citation-support evaluation rather than checking citation labels alone.
- Expand safety testing with a carefully reviewed test set.
- Add experiment tracking, audit logging, and reproducible evaluation.
- Obtain appropriate expert review and validation before considering any clinical use.

## 15. Responsible Use

PrenatalAI is intended for educational and research prototyping.

It must not be used to make decisions about pregnancy, fetal health, diagnosis, treatment, or clinical management.

Anyone with medical concerns should consult a qualified healthcare professional.

## 16. Acknowledgements

- FETAL_PLANES_DB dataset creators and maintainers.
- Authors of the research papers and ISUOG guideline included in the research corpus.
- Open-source communities behind PyTorch, Sentence Transformers, MongoDB, Ollama, Streamlit, and related tools.
