# ATS + RAG Resume Screening Assistant

An AI-powered Applicant Tracking System with a conversational Retrieval-Augmented Generation (RAG) assistant.

Upload resumes, define job requirements, and get automated candidate rankings with detailed skill matching, experience verification, career history extraction, and qualification analysis. Then ask follow-up questions through a conversational RAG assistant that retrieves evidence directly from resumes and job descriptions.

Runs entirely on your local machine using Ollama. No data is sent to external APIs.

![Python](https://img.shields.io/badge/Python-3.11+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-red)
![LangChain](https://img.shields.io/badge/LangChain-0.3+-green)
![Ollama](https://img.shields.io/badge/Ollama-Local_LLM-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)


## Features

### ATS Resume Screening

- Upload multiple PDF and DOCX resumes
- Automated skill matching with alias support and negation detection
- Total and relevant experience calculation from dated employment records
- Education and qualification verification
- Job description overlap scoring
- Weighted ATS scoring (Skills 40%, Experience 25%, Qualifications 15%, JD Overlap 20%)
- Candidate ranking with detailed assessment reports
- AI-assisted project extraction and grounding
- AI-assisted career history extraction with verification

### RAG Assistant

- Conversational question answering about candidates and job requirements
- Native LLM tool calling with three specialized retrieval tools
- Hybrid semantic and lexical search across resumes
- Structured ATS data retrieval for exact scores and rankings
- Job description retrieval for requirements and qualifications
- Multi-turn conversation with history support
- Evidence-grounded answers from original resume text

---

## Architecture

```text
┌─────────────────────────────────────────────────────┐
│                  Streamlit Frontend                  │
│              (app.py - User Interface)               │
└──────────────────────┬──────────────────────────────┘
                       │
          ┌────────────┴────────────┐
          │                         │
          v                         v
┌─────────────────┐      ┌──────────────────┐
│   ATS Pipeline  │      │  RAG Pipeline    │
│                 │      │                  │
│  resume_parser  │      │  pipeline.py     │
│  ranking.py     │      │  tools.py        │
│  matching_chain │      │                  │
└────────┬────────┘      └────────┬─────────┘
         │                        │
         v                        v
┌─────────────────┐      ┌──────────────────┐
│  Qwen 3.5 LLM  │      │  Qwen 3.5 LLM   │
│  (Extraction)   │      │  (Tool Calling)  │
│                 │      │                  │
│  Project Chain  │      │  get_ats_data    │
│  Career Chain   │      │  search_resumes  │
│                 │      │  search_jd       │
└────────┬────────┘      └────────┬─────────┘
         │                        │
         v                        v
┌─────────────────────────────────────────────┐
│              Ollama Runtime                 │
│        (Local Model Execution)              │
│                                             │
│  LLM: qwen3.5:0.8b                         │
│  Embeddings: nomic-embed-text               │
└─────────────────────────────────────────────┘
```

### RAG Tool-Calling Flow

```text
User Question
      │
      v
Qwen 3.5 (bind_tools)
      │
      v
Tool Selection
      │
      ├── get_ats_data ──────> Structured ATS Data
      │
      ├── search_resumes ────> Hybrid Retrieval
      │                          ├── Semantic Search
      │                          └── Lexical Search
      │
      └── search_job_description > JD Retrieval
      │
      v
Tool Results → Qwen 3.5 → Final Answer
```

---

## Tech Stack

| Technology | Purpose |
|---|---|
| Python 3.11+ | Core programming language |
| Streamlit | Web interface and dashboard |
| LangChain | LLM integration, LCEL chaining, tool calling, RAG |
| Ollama | Local LLM and embedding model runtime |
| Qwen 3.5 (0.8B) | Resume analysis, tool selection, answer generation |
| nomic-embed-text | Text embedding for semantic search |
| InMemoryVectorStore | Vector storage and similarity search |
| PyPDF | PDF text extraction |
| python-docx | DOCX text extraction |
| Pydantic | Structured output validation |

---

## Project Structure

```text
ats-rag-assistant/
│
├── app.py                      # Streamlit application
│
├── Ats/
│   ├── __init__.py
│   ├── resume_parser.py        # PDF and DOCX text extraction
│   ├── ranking.py              # ATS scoring and candidate ranking
│   └── matching_chain.py       # LLM-based project and career extraction
│
├── rag/
│   ├── __init__.py
│   ├── tools.py                # RAG retrieval tools
│   └── pipeline.py             # Vectorstore, tool calling, answer generation
│
├── requirements.txt
└── README.md
```

---

## Installation

### Prerequisites

- Python 3.11 or higher
- Ollama installed and running

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/ats-rag-assistant.git
cd ats-rag-assistant
```

### Step 2: Create Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Pull Ollama Models

```bash
ollama pull qwen3.5:0.8b
ollama pull nomic-embed-text
```

### Step 5: Start Ollama

```bash
ollama serve
```

### Step 6: Run the Application

```bash
streamlit run app.py
```

The application will open at `http://localhost:8501`.

---

## Requirements

```text
streamlit>=1.35.0
langchain>=0.3.0
langchain-core>=0.3.0
langchain-ollama>=0.3.0
langchain-text-splitters>=0.3.0
pypdf>=4.0.0
python-docx>=1.0.0
pydantic>=2.0.0
```

---

## Usage

### Step 1: Define Job Requirements

- Enter the job title
- Paste the full job description
- Set minimum and maximum experience requirements
- List required skills (comma-separated)
- List accepted qualifications (comma-separated)

### Step 2: Upload Resumes

- Upload PDF or DOCX resume files
- Multiple resumes can be uploaded simultaneously
- Maximum file size: 20 MB per file

### Step 3: Analyze and Rank

- Click **Analyze & Rank Resumes**
- The system processes each resume through the ATS pipeline
- Candidates are ranked by weighted ATS score

### Step 4: Review Results

Each candidate report includes:

- Overall ATS match percentage
- Skills assessment (matched and missing)
- Experience assessment (total and relevant)
- Career history (company, position, dates)
- Qualification verification
- Relevant projects
- Matching evidence
- Requirements needing verification

### Step 5: Ask Questions (RAG)

- Click **Continue to RAG** to initialize the assistant
- Ask questions about candidates, comparisons, or job requirements

Example questions:

```
Who is the highest ranked candidate?
What is John's ATS score?
Compare the top two candidates.
What skills is the second ranked candidate missing?
Where is the top candidate currently working?
What projects has candidate X worked on?
What qualifications does the job require?
Does any candidate have AWS experience?
```

---

## ATS Scoring

The weighted scoring formula:

| Component | Weight | Method |
|---|---|---|
| Skills | 40% | Regex matching with aliases and negation detection |
| Experience | 25% | Date range extraction and interval merging |
| Qualifications | 15% | Education field matching with aliases |
| JD Overlap | 20% | Lexical term overlap between resume and JD |

Weights adjust automatically when components are not applicable:

- No required skills specified → Skills weight redistributed
- No qualifications specified → Qualification weight redistributed
- No job description provided → JD overlap weight redistributed

---

## RAG Tools

| Tool | Input | Output |
|---|---|---|
| `get_ats_data` | Candidate name (optional) | Structured ATS scores, rankings, skills, experience, career history |
| `search_resumes` | Query + candidate name (optional) | Resume evidence via hybrid semantic and lexical search |
| `search_job_description` | Query | Job description evidence via hybrid search |

---

## How It Works

### Resume Processing Pipeline

```text
Resume Upload → Text Extraction → ATS Analysis → Scoring → Ranking
```

1. **Text Extraction:** PyPDF for PDFs, python-docx for DOCX files
2. **Skill Matching:** Regex-based matching with skill aliases and negation detection
3. **Experience Analysis:** Date range extraction, interval merging, role-based relevance
4. **Career Extraction:** LLM extracts employment history, verified against source text
5. **Project Extraction:** LLM extracts projects, grounded against resume evidence
6. **Qualification Matching:** Education extraction with field aliases
7. **JD Overlap:** Lexical term overlap scoring
8. **Weighted Scoring:** Configurable weighted combination of all components
9. **Ranking:** Candidates sorted by overall score with tiebreaking on skill score

### RAG Pipeline

```text
Question → Qwen Tool Calling → Tool Execution → Evidence Retrieval → Answer Generation
```

1. **Tool Calling:** Qwen selects retrieval tools based on the question
2. **ATS Retrieval:** Returns structured scoring data directly
3. **Resume Search:** Hybrid semantic + lexical search over resume chunks
4. **JD Search:** Hybrid search over job description chunks
5. **Answer Generation:** Qwen produces a grounded answer from tool results

---

## Limitations

- **Model Size:** Qwen 3.5 (0.8B) is a small model. Complex multi-step reasoning and tool selection can be unreliable. Upgrading to a larger model improves accuracy.

- **Tool Calling Reliability:** The model may occasionally skip necessary tool calls or select incorrect tools. The pipeline includes a maximum round limit but does not force initial retrieval.

- **Experience Extraction:** Date extraction depends on consistent resume formatting. Unconventional date formats may not be recognized.

- **Skill Matching:** Rule-based matching cannot capture semantic skill equivalence beyond the defined alias dictionary.

- **Career Verification:** Extracted employment records are verified against the experience section text, but the validation cannot guarantee that a company name, position, and date belong to the same employment record.

- **OCR:** Scanned PDF resumes without selectable text are not supported. OCR integration is not included.

- **Language:** The system is designed for English-language resumes.

---

## Future Improvements

- [ ] Upgrade to a larger local LLM for improved tool-calling reliability
- [ ] Add OCR support for scanned PDF resumes
- [ ] Implement persistent vector storage
- [ ] Add resume export and reporting functionality
- [ ] Implement forced initial retrieval before answer generation
- [ ] Add support for additional resume formats
- [ ] Improve multi-language resume support
- [ ] Add batch comparison and summary reports

---

## License

This project is licensed under the MIT License.

---

## Acknowledgments

- [Ollama](https://ollama.com/) for local LLM execution
- [LangChain](https://www.langchain.com/) for LLM framework and tooling
- [Streamlit](https://streamlit.io/) for the web interface
- [Qwen](https://github.com/QwenLM/Qwen) for the language model
- [Nomic](https://www.nomic.ai/) for the embedding model
```

---

## `.gitignore`

```gitignore
__pycache__/
*.pyc
*.pyo
.env
venv/
.venv/
*.egg-info/
dist/
build/
.streamlit/secrets.toml
*.pdf
*.docx
```

---

## Summary

| Item | Value |
|---|---|
| **Repo Name** | `ats-rag-assistant` |
| **Description** | AI-powered ATS with conversational RAG assistant. Automated resume screening, candidate ranking, career history extraction, and hybrid retrieval QA. Python, Streamlit, LangChain, Qwen, Ollama. Runs locally. |
| **README** | Complete with architecture, installation, usage, scoring explanation, limitations, and future improvements |
| **License** | MIT |
