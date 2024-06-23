import asyncio
import os

from fastapi.encoders import jsonable_encoder
from dotenv import load_dotenv
from uvicorn import run
from fastapi import FastAPI, Request
from sse_starlette.sse import EventSourceResponse
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import concurrent.futures
from torch import load as torch_load, tensor
from torchtext.data.utils import get_tokenizer
from model import LitGRU
from data import ProteinInput

load_dotenv()

app = FastAPI(
    title="AgGRU-Check API",
    description="Classify amyloidogenic regions in protein sequences",
    version="0.0.1"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.getenv("ALLOWED_ORIGIN")],
    allow_credentials=True,
    allow_methods=["POST"],
    allow_headers=["*"],
)

executor = concurrent.futures.ThreadPoolExecutor(max_workers=10)

async def predict_single(model, sequence, sequence_length, modelIndex):
    device = model.device
    prediction = model(sequence.unsqueeze(0).to(device), sequence_length.to("cpu"))
    return modelIndex, prediction

def split(sequence):
    return [char for char in sequence]

@app.on_event("startup")
async def startup_event():
    vocab = torch_load("./vocab.pt", map_location="cpu")
    ensemble = [
        LitGRU.load_from_checkpoint(
            "./bestCheckpoints/" + checkpoint, vocab=vocab, map_location="cpu"
        )
        for checkpoint in os.listdir("./bestCheckpoints")
    ]
    tokenizer = get_tokenizer(split)
    app.package = {"ensemble": ensemble, "vocab": vocab, "tokenizer": tokenizer}

@app.post("/api/predict")
async def do_predict(request: Request):
    data = await request.json()

    try:
        validated_data = ProteinInput(sequences=data["sequenceList"])
    except ValueError as ve:
        return StreamingResponse(error_stream(str(ve)), media_type="text/event-stream")

    async def predict_stream():
        for sequence in validated_data.sequences:
            sequence = [
                app.package["vocab"][token]
                for token in app.package["tokenizer"](sequence)
            ]
            sequence_tensor = tensor(sequence)
            sequence_length = tensor([len(sequence)]).int()

            prediction_tasks = [
                predict_single(model, sequence_tensor, sequence_length, i)
                for i, model in enumerate(app.package["ensemble"])
            ]

            for future in asyncio.as_completed(prediction_tasks):
                index, prediction = await future
                yield f"data: {jsonable_encoder({f'model_{index}': prediction.item()})}\n\n"

    return StreamingResponse(predict_stream(), media_type="text/event-stream")

def error_stream(error_message):
    yield f"event: error\ndata: Error: {error_message}\n\n"

if __name__ == "__main__":
    run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("DEPLOYMENT_PORT")),
        reload=True,
        log_config="log.ini",
    )
