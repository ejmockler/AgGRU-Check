import asyncio
import os
from dotenv import load_dotenv

from uvicorn import run
from fastapi import FastAPI, Request
from sse_starlette.sse import EventSourceResponse

from fastapi.logger import logger
from fastapi.encoders import jsonable_encoder
from fastapi.responses import StreamingResponse
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware

import concurrent.futures
from torch import load as torch_load
from torch import tensor, stack, mean
from torchtext.data.utils import get_tokenizer

from model import LitGRU
from data import ProteinInput

load_dotenv()

app = FastAPI(
    title="AgGRU-Check API",
    description="Classify amyloidogenic regions in protein sequences",
    version="0.0.1",
    terms_of_service=None,
    contact=None,
    license_info=None,
)

# Allow CORS for local debugging
app.add_middleware(CORSMiddleware, allow_origins=["*"])

# Load custom exception handlers
# app.add_exception_handler(RequestValidationError, validation_exception_handler)
# app.add_exception_handler(Exception, python_exception_handler)

executor = concurrent.futures.ThreadPoolExecutor(max_workers=10)


async def predict_single(model, sequence, sequence_length, modelIndex):
    device = model.device
    prediction = model(sequence.unsqueeze(0).to(device), sequence_length.to("cpu"))
    return modelIndex, prediction


# Tokenize peptide sequences by splitting into individual amino acids
def split(sequence):
    return [char for char in sequence]


@app.on_event("startup")
async def startup_event():
    """
    Initialize FastAPI and add variables
    """
    vocab = torch_load("./vocab.pt", map_location="cpu")
    ensemble = [
        LitGRU.load_from_checkpoint(
            "./bestCheckpoints/" + checkpoint, vocab=vocab, map_location="cpu"
        )
        for checkpoint in os.listdir("./bestCheckpoints")
    ]
    tokenizer = get_tokenizer(split)
    # add model and vocab to app state
    app.package = {"ensemble": ensemble, "vocab": vocab, "tokenizer": tokenizer}


@app.get("/api/predict", response_class=EventSourceResponse)
async def do_predict(request: Request):
    """
    Perform prediction on input data
    """
    data = await request.json()
    try:
        validated_data = ProteinInput(**data)
    except ValueError as ve:

        def error_stream():
            yield f"event: error\ndata: Error: {ve}\n\n"

        return StreamingResponse(error_stream(), media_type="text/event-stream")

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


if __name__ == "__main__":
    # server api
    run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("DEPLOYMENT_PORT")),
        reload=True,
        log_config="log.ini",
    )
