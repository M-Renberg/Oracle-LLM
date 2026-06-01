# Oracle-LLM

A FastAPI-based REST API that lets you upload CSV datasets and ask questions about them using a local LLM (SmolLM2).

## Requirements

- IDE 
- Python 3.10+
- Internet connection

Install dependencies:

```bash
pip install fastapi uvicorn transformers torch pandas pydantic
```

or

```bash
uv add fastapi uvicorn transformers torch pandas pydantic
```


## Project Structure

```
├── main.py          # FastAPI app and endpoints
├── datahandler.py   # CSV loading and in-memory storage
├── schemas.py       # Pydantic request/response models
└── chain/
    ├── runable.py   # Base chain abstractions (Runable, RunableSequence, etc.)
    ├── llm.py       # LLM pipeline steps (DataSelector, LLMRunner, ResponseParser, etc.)
    └── pipline.py   # Assembled chain and run_oracle function
```

## Running the API

```bash
uv run uvicorn main:app --reload --port 8001
```

The API will be available at `http://127.0.0.1:8001`. Interactive docs at `/docs`.

## Usage

### 1. Upload a CSV file

```bash
curl -X POST http://127.0.0.1:8001/data/upload \
  -F "file=@your_dataset.csv"
```

### 2. Ask a question about the data

```bash
curl -X POST http://127.0.0.1:8001/ai/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What product has sold the most units?"}'
```

### Example response

```json
{
  "question": "What product has sold the most units?",
  "answer": "Based on the data, the best-selling product is ...",
  "model": "HuggingFaceTB/SmolLM2-1.7B-Instruct"
}
```

## How it works

The query goes through a two-step LLM chain:

1. **DataSelector** — filters the most relevant rows from the dataset for the question
2. **LLMRunner** — generates an answer based on the filtered data
3. **AnalysisStep** — refines the answer with a second LLM pass
4. **ResponseParser** — extracts the final clean answer


## Model

Uses [HuggingFaceTB/SmolLM2-1.7B-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct) running locally on CPU. The first request will download the model weights (~3.5 GB).