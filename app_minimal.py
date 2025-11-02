"""Single-file minimal demo of CustomerAssist.
Run: python app_minimal.py
Then open http://127.0.0.1:8000/docs to use the API.
"""
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
import uvicorn

app = FastAPI(title='CustomerAssist Minimal Single File')
qa = pipeline('question-answering', model='distilbert-base-cased-distilled-squad')
sum_pipe = pipeline('summarization', model='google/flan-t5-small')
gen = pipeline('text2text-generation', model='google/flan-t5-small')

class QAReq(BaseModel):
    context: str
    question: str

class TextReq(BaseModel):
    text: str

@app.post('/qa')
def do_qa(req: QAReq):
    return qa({'context': req.context, 'question': req.question})

@app.post('/summarize')
def do_sum(req: TextReq):
    out = sum_pipe(req.text, max_length=120, min_length=30, do_sample=False)
    return {'summary': out[0]['summary_text']}

@app.post('/explain')
def explain_topic(req: TextReq):
    prompt = f"Explain to a high-school student: {req.text}"
    out = gen(prompt, max_length=150, do_sample=False)
    return {'explanation': out[0]['generated_text']}

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
