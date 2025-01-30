import { produce } from "sveltekit-sse";
import { json } from "@sveltejs/kit";
import { API_URL } from "$env/static/private";
import type { RequestHandler } from "./$types";

interface ModelPrediction {
  model_0?: number;
  model_1?: number;
  model_2?: number;
  sequence?: string;
}

// Define the structure for saliency results
interface SaliencyResult {
  position: number;
  saliency: number;
  confidence: number;
  activeWindows?: WindowResult[]; // Track windows affecting this position
  activeDomains?: DomainResult[]; // Add domain tracking
  sequence?: string;
  sequence_index?: number;
  models_completed?: number;
  total_models?: number;
  statistics?: {
    q25: number;
    q75: number;
    iqr: number;
  };
}

// Define structure for model progress
interface ModelProgress {
  sequence_index: number;
  model_index: number;
  models_completed: number;
  total_models: number;
}

// Add new interfaces
interface WindowResult {
  start: number;
  end: number;
  score: number;
  significance: number;
  consistency: number;
}

interface WindowMessage {
  type: 'window_result';
  sequence_index: number;
  model_index: number;
  window: WindowResult;
}

// Add new interfaces for domain results
interface DomainResult {
  start: number;
  end: number;
  peak_score: number;
  mean_score: number;
  confidence: number;
  stability: number;
}

interface DomainMessage {
  type: 'domain_result';
  sequence_index: number;
  model_index: number;
  domains: DomainResult[];
}

// Custom JSON parser that handles NaN
function parseJSON(text: string) {
  return JSON.parse(text.replace(/:\s*NaN\b/g, ': null'));
}

export const POST: RequestHandler = async ({ request }) => {
  const data = await request.json();
  const sequences = data.sequences;
  
  const sequenceList = splitSequences(sequences).filter(
    (seq) => !data.processedSequences.includes(seq.toLocaleUpperCase())
  );

  console.log("Processing sequences:", {
    originalInput: sequences,
    splitSequences: sequenceList,
    count: sequenceList.length
  });

  if (sequenceList.length === 0) {
    return json({ error: "No sequences provided" }, { status: 400 });
  }

  return produce(async ({ emit }) => {
    // Initialize maps to track state
    const allCurrentResults = new Map<number, SaliencyResult[]>();
    const sequenceProgress = new Map<number, ModelProgress>();
    const cleanedSequences = new Map<number, string>();

    try {
      for (let i = 0; i < sequenceList.length; i++) {
        const sequence = sequenceList[i];
        const cleanSequence = extractSequence(sequence);
        cleanedSequences.set(i, cleanSequence);
      }

      const response = await fetch(`${API_URL}/api/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sequenceList }),
      });

      if (!response.ok) {
        throw new Error(`Prediction failed: ${response.statusText}`);
      }

      const reader = response.body?.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (reader) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const resultData = parseJSON(line.slice(6));
              console.log("SSE Data:", resultData); // Log all SSE data

              if (resultData.type === 'position_result') {
                const { sequence_index, position, saliency, confidence } = resultData;
                
                // Handle null/NaN saliency
                const validSaliency = saliency === null ? 0 : saliency;

                // Initialize results array for this sequence if needed
                if (!allCurrentResults.has(sequence_index)) {
                  // Get the cleaned sequence from the map
                  const cleanSequence = cleanedSequences.get(sequence_index);
                  if (!cleanSequence) {
                    console.error("Clean sequence not found for index:", sequence_index);
                    continue; // Skip if sequence not found
                  }
                  const sequenceLength = cleanSequence.length;
                  
                  allCurrentResults.set(sequence_index, new Array(sequenceLength).fill(null).map((_, idx) => ({
                    position: idx,
                    saliency: 0,
                    confidence: 0,
                    activeWindows: [],
                    activeDomains: [],
                    sequence_index,
                  })));
                }

                // Update the result for this position
                const currentResults = allCurrentResults.get(sequence_index)!;
                currentResults[position] = {
                  position,
                  saliency: validSaliency,
                  confidence,
                  activeWindows: [],
                  activeDomains: [],
                  sequence_index,
                };

                // Emit progress update with partial results
                emit("message", JSON.stringify({
                  type: 'progress',
                  sequence: sequenceList[sequence_index],
                  sequence_index,
                  partialResults: currentResults.map(res => ({
                    position: res.position,
                    score: res.saliency,
                    confidence: res.confidence,
                  })),
                  models_completed: resultData.models_completed,
                  total_models: resultData.total_models,
                }));

              } else if (resultData.type === 'model_complete') {
                // Update sequence progress
                sequenceProgress.set(resultData.sequence_index, resultData);
                
                // Emit progress update
                emit("message", JSON.stringify({
                  type: 'model_progress',
                  ...resultData
                }));
              } else if (resultData.type === 'window_result') {
                const { sequence_index, model_index, window } = resultData;
                
                // Get existing results array if it exists
                const currentResults = allCurrentResults.get(sequence_index);
                if (currentResults) {
                  // Update positions affected by this window
                  for (let pos = window.start; pos < window.end; pos++) {
                    if (!currentResults[pos].activeWindows) {
                      currentResults[pos].activeWindows = [];
                    }
                    // Add window to affected positions
                    currentResults[pos].activeWindows.push(window);
                  }

                  // Emit progress update with window information
                  emit("message", JSON.stringify({
                    type: 'window_progress',
                    sequence_index,
                    model_index,
                    window,
                    partialResults: currentResults.map(res => ({
                      position: res.position,
                      score: res.saliency,
                      confidence: res.confidence,
                      activeWindows: res.activeWindows,
                    })),
                  }));
                }
              } else if (resultData.type === 'sequence_start') {
                console.log("SSE Event: sequence_start", resultData); // Log sequence_start
                // Handle sequence_start event if needed in +server.ts
              } else if (resultData.type === 'model_start') {
                console.log("SSE Event: model_start", resultData); // Log model_start
                // Handle model_start event if needed in +server.ts
              } else if (resultData.type === 'domain_result') {
                const { sequence_index, domains } = resultData;
                const currentResults = allCurrentResults.get(sequence_index);
                
                if (currentResults) {
                  // Update positions affected by domains
                  domains.forEach(domain => {
                    for (let pos = domain.start; pos <= domain.end; pos++) {
                      if (!currentResults[pos].activeDomains) {
                        currentResults[pos].activeDomains = [];
                      }
                      currentResults[pos].activeDomains.push(domain);
                    }
                  });

                  // Emit progress update with domain information
                  emit("message", JSON.stringify({
                    type: 'domain_progress',
                    sequence_index,
                    domains,
                    partialResults: currentResults.map(res => ({
                      position: res.position,
                      score: res.saliency,
                      confidence: res.confidence,
                      activeDomains: res.activeDomains,
                    })),
                  }));
                }
              } else if (resultData.type === 'sequence_prediction') {
                emit("message", JSON.stringify({
                  type: 'sequence_prediction',
                  sequence_index: resultData.sequence_index,
                  is_amyloid: resultData.is_amyloid,
                  prediction_score: resultData.prediction_score,
                  confidence: resultData.confidence,
                  model_predictions: resultData.model_predictions
                }));
              }

            } catch (e) {
              console.warn('Failed to parse data line:', line, e);
            }
          } else if (line.startsWith('event: end')) {
            // Send final results for all sequences
            for (const [sequence_index, results] of allCurrentResults.entries()) {
              emit("message", JSON.stringify({
                type: 'result',
                sequence: sequenceList[sequence_index],
                sequence_index,
                results: results.map(res => ({
                  position: res.position,
                  score: res.saliency,
                  confidence: res.confidence,
                })),
              }));
            }
            emit("message", "end");
            return;
          } else if (line.startsWith('event: error')) {
            const errorData = line.slice(13);
            console.error("SSE Error Event Received:", errorData);
            emit("message", JSON.stringify({
              type: 'error',
              error: errorData
            }));
            emit("message", "end");
            return;
          }
        }
      }

    } catch (error) {
      console.error("SSE Production error:", error);
      emit("message", JSON.stringify({
        type: 'error',
        error: error instanceof Error ? error.message : "An unexpected error occurred"
      }));
      emit("message", "end");
    }
  });
};

function extractSequence(input: string): string {
  const lines = input.split(/\r?\n/);

  // If FASTA format, remove header and join sequence lines
  if (lines[0].startsWith('>')) {
    return lines.slice(1).join('').replace(/\s+/g, '');
  }

  // If FASTQ format, return just the sequence line
  if (lines[0].startsWith('@') && lines.length >= 4) {
    return lines[1].trim();
  }

  // Otherwise, clean up and return the raw sequence
  return input.replace(/\s+/g, '');
}

/**
 * Split input sequences into a list of sequences, preserving headers and sequences together.
 * Handles both FASTA/FASTQ formatted sequences and plain sequences.
 * @param {string} sequences - The input sequences as a single string.
 * @returns {Array<string>} - An array of sequences.
 */
function splitSequences(sequences: string): Array<string> {
  const lines = sequences.split(/\r?\n/);
  let currentBlock: string[] = [];
  const sequenceBlocks: string[] = [];
  let isFastaFastq = false;

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i].trim();

    // Detect start of a new FASTA/FASTQ block
    if (line.startsWith(">") || line.startsWith("@")) {
      if (currentBlock.length > 0) {
        sequenceBlocks.push(currentBlock.join("\n"));
        currentBlock = [];
      }
      isFastaFastq = true;
    }

    currentBlock.push(line);

    // Handle FASTQ '+' line and its corresponding quality scores line
    if (isFastaFastq && line.startsWith("+")) {
      // Include the next line as the quality scores
      if (i + 1 < lines.length) {
        currentBlock.push(lines[++i].trim());
      }
    }
  }

  // Push the last block if any
  if (currentBlock.length > 0) {
    sequenceBlocks.push(currentBlock.join("\n"));
  }

  return sequenceBlocks;
}
