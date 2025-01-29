import asyncio
import json
import logging
import os
from math import exp  # Add to imports at top
import math

from fastapi.encoders import jsonable_encoder
from dotenv import load_dotenv
from uvicorn import run
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import concurrent.futures
from torch import (
    load as torch_load, 
    tensor, 
    float32, 
    no_grad, 
    autograd, 
    cat, 
    squeeze, 
    sigmoid,
    randperm
)
import torch
from torchtext.data.utils import get_tokenizer
from model import LitGRU
from data import ProteinInput
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

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
    """
    Analyze sequence regions with improved filtering and streaming results
    """
    model.eval()
    device = model.device
    
    with no_grad():
        # Get sequence length
        seq_length = sequence_length[0]
        
        # Analysis parameters
        n_permutations = 5
        
        # Adjust window sizes based on sequence length, but maintain minimum biological relevance
        max_window = min(27, seq_length)  # Cap at original max or sequence length
        min_window = min(15, max(15, seq_length // 3))  # Minimum 15 for biological relevance
        mid_window = min(21, (max_window + min_window) // 2)  # Try to keep original mid size when possible
        
        window_sizes = [size for size in [min_window, mid_window, max_window] 
                       if size <= seq_length]  # Only use windows that fit
        
        stride = max(1, min(2, seq_length // 10))  # Adjust stride for short sequences
        
        # Track both window-specific and aggregate scores
        position_scores = [[] for _ in range(seq_length)]
        window_results = []  # Store intermediate window results
        
        # Analyze windows of different sizes
        for window_size in window_sizes:
            for start in range(0, seq_length - window_size + 1, stride):
                end = start + window_size
                
                window_seq = sequence_tensor[start:end]
                window_len = tensor([window_size]).int()
                
                # Get prediction for actual window
                window_tensor = window_seq.unsqueeze(0).to(device)
                true_pred = model(window_tensor, window_len)
                
                # Get predictions for permuted windows
                perm_preds = []
                for _ in range(n_permutations):
                    perm_indices = tensor(list(range(window_size))).long()
                    perm_indices = perm_indices[randperm(window_size)]
                    perm_seq = window_seq[perm_indices]
                    
                    perm_tensor = perm_seq.unsqueeze(0).to(device)
                    perm_pred = model(perm_tensor, window_len)
                    perm_preds.append(perm_pred.item())
                
                # Calculate window statistics
                mean_perm_pred = sum(perm_preds) / len(perm_preds)
                perm_std = (sum((p - mean_perm_pred) ** 2 for p in perm_preds) / len(perm_preds)) ** 0.5
                
                # Score criteria:
                # 1. True prediction should be higher than permuted
                # 2. Permutations should have low variance (consistent non-amyloid signal)
                # 3. True prediction should be significantly above permutation mean
                base_score = max(0, true_pred.item() - mean_perm_pred)
                significance = float(base_score / (perm_std + 1e-6))  # Convert to float
                consistency = float(1 / (1 + perm_std))  # Convert to float
                
                window_score = float(base_score * significance * consistency)  # Convert to float
                
                # Store window result for streaming
                window_results.append({
                    'start': int(start),  # Convert to int
                    'end': int(end),     # Convert to int
                    'size': int(window_size),  # Convert to int
                    'score': float(window_score),  # Convert to float
                    'true_pred': float(true_pred.item()),  # Convert to float
                    'perm_mean': float(mean_perm_pred),  # Convert to float
                    'perm_std': float(perm_std),  # Convert to float
                    'significance': float(significance),  # Convert to float
                    'consistency': float(consistency)  # Convert to float
                })
                
                # Weight positions by distance from window center
                window_center = (start + end) / 2
                for pos in range(start, end):
                    # Gaussian weighting centered on window
                    dist_from_center = abs(pos - window_center)
                    weight = exp(-0.5 * (dist_from_center / (window_size/4)) ** 2)
                    position_scores[pos].append(window_score * weight)
        
        # Aggregate position scores with additional filtering
        final_scores = []
        for pos, scores in enumerate(position_scores):
            if scores:
                # Consider score distribution at this position
                scores = tensor(scores)
                mean_score = scores.mean().item()
                score_std = scores.std().item()
                
                # Filter out positions with inconsistent scores across windows
                score_consistency = 1 / (1 + score_std)
                final_score = mean_score * score_consistency
                final_scores.append(final_score)
            else:
                final_scores.append(0.0)
        
        # Normalize final scores
        scores = tensor(final_scores)
        if scores.max() > scores.min():
            scores = (scores - scores.min()) / (scores.max() - scores.min())
        
        return scores.tolist(), window_results

@app.post("/api/predict")
async def do_predict(request: Request):
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
                logging.info(f"Starting prediction for sequence {sequence_index}")
                sequence = full_sequence.upper()
                sequence_length = len(sequence)
                
                # Initialize position scores for ensemble
                position_scores = [[0.0 for _ in range(len(app.package["ensemble"]))] 
                                for _ in range(sequence_length)]
                models_completed = [False] * len(app.package["ensemble"])

                # Process with each model
                for model_idx, model in enumerate(app.package["ensemble"]):
                    # Prepare sequence
                    sequenceVector = [app.package["vocab"][token] 
                                   for token in app.package["tokenizer"](sequence)]
                    sequence_tensor = tensor(sequenceVector, dtype=int)
                    seq_length = tensor([len(sequenceVector)]).int()

                    # Analyze sequence regions and stream window results
                    scores, window_results = analyze_sequence_regions(
                        model, sequence_tensor, seq_length)

                    # Stream window-level results first
                    for window in window_results:
                        window_message = {
                            'type': 'window_result',
                            'sequence_index': sequence_index,
                            'model_index': model_idx,
                            'window': {
                                'start': window['start'],
                                'end': window['end'],
                                'score': window['score'],
                                'significance': window['significance'],
                                'consistency': window['consistency']
                            }
                        }
                        yield f"data: {json.dumps(window_message)}\n\n"
                        await asyncio.sleep(0.001)

                    # Update position scores for this model
                    for pos, score in enumerate(scores):
                        position_scores[pos][model_idx] = score
                        
                        # Get valid scores for this position
                        valid_scores = [s for idx, s in enumerate(position_scores[pos]) 
                                      if models_completed[idx] or idx == model_idx]
                        
                        # Calculate mean score first (needed for both cases)
                        mean_score = sum(valid_scores) / len(valid_scores)
                        
                        if len(valid_scores) > 1:  # Need at least 2 scores for agreement
                            # Calculate variance and normalize to confidence
                            variance = sum((s - mean_score) ** 2 for s in valid_scores) / len(valid_scores)
                            std_dev = variance ** 0.5
                            # Higher agreement = lower std_dev = higher confidence
                            # Scale confidence to be between 0 and 1
                            model_agreement = max(0, min(1, 1 - (std_dev * 2)))
                            # Weight confidence by proportion of models completed
                            confidence = model_agreement * (len(valid_scores) / len(app.package["ensemble"]))
                        else:
                            # Single model case - lower confidence
                            confidence = 0.5 * (len(valid_scores) / len(app.package["ensemble"]))

                        message = {
                            'type': 'position_result',
                            'position': pos,
                            'saliency': None if math.isnan(mean_score) else float(mean_score),
                            'confidence': confidence,
                            'sequence': full_sequence,
                            'sequence_index': sequence_index,
                            'models_completed': len(valid_scores),
                            'total_models': len(app.package["ensemble"])
                        }
                        yield f"data: {json.dumps(message)}\n\n"
                        await asyncio.sleep(0.001)

                    # Mark this model as completed
                    models_completed[model_idx] = True

                    # Stream progress update
                    progress_message = {
                        'type': 'model_complete',
                        'sequence_index': sequence_index,
                        'model_index': model_idx,
                        'models_completed': sum(models_completed),
                        'total_models': len(app.package["ensemble"])
                    }
                    yield f"data: {json.dumps(progress_message)}\n\n"

                logging.info(f"Completed prediction for sequence {sequence_index}")

            # Move the end event outside the sequence loop
            if not await request.is_disconnected():
                yield "event: end\n\n"

        except Exception as e:
            logging.error(f"Error in prediction stream: {e}", exc_info=True)
            yield f"event: error\ndata: Error: {str(e)}\n\n"

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
