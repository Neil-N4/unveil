# Reddit Shampoo Summarizer

A FastAPI application that summarizes Reddit discussions about shampoo products.

## How to Run

### 1. Start the Server

In your terminal, navigate to the project directory and run:

```bash
cd /Users/neilnair/Documents/GitHub/hackohioproject
python main.py
```

The server will start on `http://localhost:8083`

### 2. Test the API

Open a new terminal window and test with:

```bash
curl "http://localhost:8083/reddit-summary/best%20shampoo" | python -m json.tool
```

Or visit in your browser:
```
http://localhost:8083/reddit-summary/best%20shampoo
```

### 3. Health Check

Check if the server is running:
```bash
curl http://localhost:8083/health
```

## API Endpoints

- `GET /reddit-summary/{query}` - Get shampoo summaries for a query
- `GET /health` - Health check endpoint

## Example Queries

- `best shampoo`
- `best shampoo for oily hair`
- `clarifying shampoo`
- `anti-dandruff shampoo`

## Requirements

Install dependencies:
```bash
pip install fastapi uvicorn praw spacy rapidfuzz pandas python-multipart
python -m spacy download en_core_web_sm
```

