import asyncio
import json
import logging
import os
from math import exp  # Add to imports at top
import math
import traceback
import dataclasses

from dotenv import load_dotenv
from uvicorn import run
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

from fastapi.middleware.cors import CORSMiddleware
import concurrent.futures
from torch import (
    load as torch_load, 
    tensor, 
    no_grad, 
)
from torchtext.data.utils import get_tokenizer
from model import LitGRU
from data import ProteinInput
from asyncio import Semaphore
from contextlib import asynccontextmanager
import numpy as np
from json import JSONEncoder

load_dotenv()

app = FastAPI(
    title="AgGRU-Check API",
    description="Classify amyloidogenic regions in protein sequences",
    version="0.0.1"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'], # [os.getenv("ALLOWED_ORIGIN")],
    allow_credentials=True,
    allow_methods=["POST"],
    allow_headers=["*"],
)

executor = concurrent.futures.ThreadPoolExecutor(max_workers=10)

# Limit concurrent sequence processing to 1
sequence_semaphore = Semaphore(1)  # Only process one sequence at a time

@asynccontextmanager
async def acquire_sequence_slot(sequence_index: int):
    try:
        await sequence_semaphore.acquire()
        yield
    finally:
        sequence_semaphore.release()

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

def analyze_sequence_regions(model, sequence_tensor, sequence_length):
    """Analyze sequence regions by examining how positions influence the whole-sequence prediction"""
    model.eval()
    device = model.device
    
    with no_grad():
        seq_length = sequence_length[0]
        
        # First get baseline prediction for full sequence
        full_seq_pred = model(sequence_tensor.unsqueeze(0).to(device), sequence_length)
        is_amyloid_sequence = full_seq_pred.item() > 0.5
        
        # Adjust window sizes and scoring based on sequence prediction
        window_sizes = [min(size, seq_length) for size in [15, 21, 27]]
        position_scores = [[] for _ in range(seq_length)]
        
        for window_size in window_sizes:
            for start in range(0, seq_length - window_size + 1, 2):
                end = start + window_size
                
                if is_amyloid_sequence:
                    # For amyloid sequences, use more stringent masking test
                    # Only consider regions that significantly reduce amyloid prediction when masked
                    masked_seq = sequence_tensor.clone()
                    masked_seq[start:end] = app.package["vocab"]["<pad>"]
                    masked_pred = model(masked_seq.unsqueeze(0).to(device), sequence_length)
                    
                    # Require stronger evidence - region must cause substantial drop
                    impact = max(0, (full_seq_pred - masked_pred).item())
                    # Apply nonlinear scaling to emphasize strong effects
                    impact = impact ** 2 if impact > 0.3 else 0
                    
                else:
                    # For non-amyloid sequences, require very strong evidence
                    window_seq = sequence_tensor[start:end]
                    window_len = tensor([window_size]).int()
                    window_pred = model(window_seq.unsqueeze(0).to(device), window_len)
                    
                    # Much higher threshold for non-amyloid sequences
                    # Window must very confidently predict amyloid
                    impact = max(0, window_pred.item() - 0.7)  # Increased threshold
                    impact = impact ** 2  # Nonlinear scaling
                
                # Weight positions by distance from window center and impact
                if impact > 0:  # Only consider significant impacts
                    window_center = (start + end) / 2
                    for pos in range(start, end):
                        dist_from_center = abs(pos - window_center)
                        weight = exp(-0.5 * (dist_from_center / (window_size/4)) ** 2)
                        position_scores[pos].append(impact * weight)
        
        # More conservative score aggregation
        final_scores = []
        for pos_scores in position_scores:
            if pos_scores:
                # Emphasize strong signals more
                max_score = max(pos_scores)
                mean_score = sum(pos_scores) / len(pos_scores)
                # Use max score more heavily for stronger signals
                final_scores.append(0.8 * max_score + 0.2 * mean_score)
            else:
                final_scores.append(0.0)
        
        # Normalize scores
        scores = tensor(final_scores)
        if scores.max() > scores.min():
            scores = (scores - scores.min()) / (scores.max() - scores.min())
            
            # Apply additional threshold to reduce noise
            scores = scores * (scores > 0.2).float()
        
        return scores.tolist()

@app.post("/api/predict")
async def do_predict(request: Request):
    try:
        data = await request.json()
        logging.info(f"Received request with data: {data}")
        
        try:
            validated_data = ProteinInput(sequences=data["sequenceList"])
            logging.info(f"Validated sequences: {len(validated_data.sequences)}")
            
            for idx, seq in enumerate(validated_data.sequences):
                logging.info(f"Sequence {idx}: Length={len(seq)}, First 50 chars: {seq[:50]}...")
        except ValueError as ve:
            logging.error(f"Validation error: {str(ve)}")
            return StreamingResponse(error_stream(str(ve)), media_type="text/event-stream")

        async def predict_stream():
            try:
                for sequence_index, full_sequence in enumerate(validated_data.sequences):
                    if await request.is_disconnected():
                        logging.info("Client disconnected, stopping prediction")
                        return

                    async with acquire_sequence_slot(sequence_index):
                        sequence = full_sequence.upper()
                        sequence_length = len(sequence)
                        position_scores = [[0.0 for _ in range(len(app.package["ensemble"]))]
                                        for _ in range(sequence_length)]
                        models_completed = [False] * len(app.package["ensemble"])

                        # Send sequence start message
                        yield f"data: {json.dumps({'type': 'sequence_start', 'sequence_index': sequence_index, 'total_models': len(app.package['ensemble'])})}\n\n"

                        for model_idx, model in enumerate(app.package["ensemble"]):
                            if await request.is_disconnected():
                                return

                            # Send model start message
                            yield f"data: {json.dumps({'type': 'model_start', 'sequence_index': sequence_index, 'model_index': model_idx})}\n\n"

                            sequenceVector = [app.package["vocab"][token]
                                           for token in app.package["tokenizer"](sequence)]
                            sequence_tensor = tensor(sequenceVector, dtype=int)
                            seq_length = tensor([len(sequenceVector)]).int()

                            # Get results directly instead of using async for
                            final_results = analyze_sequence_regions(
                                model, sequence_tensor, seq_length
                            )
                            
                            # Process the results
                            if final_results:  # Final position-level results
                                # Update position scores
                                for pos, score in enumerate(final_results):
                                    position_scores[pos][model_idx] = score
                                    
                                    # Calculate mean score and confidence
                                    valid_scores = [s for idx, s in enumerate(position_scores[pos])
                                                  if models_completed[idx] or idx == model_idx]
                                    mean_score = sum(valid_scores) / len(valid_scores)
                                    
                                    # Calculate confidence based on model agreement
                                    if len(valid_scores) > 1:
                                        variance = sum((s - mean_score) ** 2 for s in valid_scores) / len(valid_scores)
                                        std_dev = variance ** 0.5
                                        model_agreement = max(0, min(1, 1 - (std_dev * 2)))
                                        confidence = model_agreement * (len(valid_scores) / len(app.package["ensemble"]))
                                    else:
                                        confidence = 0.5 * (len(valid_scores) / len(app.package["ensemble"]))
                                    
                                    message = {
                                        'type': 'position_result',
                                        'position': pos,
                                        'saliency': None if math.isnan(mean_score) else float(mean_score),
                                        'confidence': confidence,
                                        'sequence_index': sequence_index,
                                        'models_completed': sum(models_completed),
                                        'total_models': len(app.package["ensemble"])
                                    }
                                    yield f"data: {json.dumps(message)}\n\n"
                                    await asyncio.sleep(0.001)

                            models_completed[model_idx] = True
                            yield f"data: {json.dumps({'type': 'model_complete', 'sequence_index': sequence_index, 'model_index': model_idx, 'models_completed': sum(models_completed), 'total_models': len(app.package['ensemble'])})}\n\n"

                if not await request.is_disconnected():
                    yield "event: end\n\n"

            except Exception as e:
                logging.error(f"Error in prediction stream: {str(e)}", exc_info=True)
                error_message = {
                    'type': 'error',
                    'error': str(e),
                    'traceback': traceback.format_exc()
                }
                yield f"data: {json.dumps(error_message)}\n\n"

        return StreamingResponse(predict_stream(), media_type="text/event-stream")
        
    except Exception as e:
        logging.error(f"Error in do_predict: {str(e)}", exc_info=True)
        return StreamingResponse(
            error_stream(f"Server error: {str(e)}"), 
            media_type="text/event-stream"
        )

def error_stream(error_message):
    yield f"data: {json.dumps({'type': 'error', 'error': error_message})}\n\n"

class NumpyJSONEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if dataclasses.is_dataclass(obj):
            return dataclasses.asdict(obj)
        return super().default(obj)

if __name__ == "__main__":
    run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("DEPLOYMENT_PORT")),
        reload=True,
        log_config="log.ini",
    )
