import { produce } from "sveltekit-sse";
import { json } from "@sveltejs/kit";
import { API_URL } from "$env/static/private";
import type { RequestHandler } from "./$types";

export const POST: RequestHandler = async ({ request }) => {
  const data = await request.json();
  const sequences = data.sequences;

  // Split the input into individual sequences
  console.log(data);
  const sequenceList = splitSequences(sequences).filter(
    (seq) => !data.processedSequences.includes(seq.toLocaleUpperCase())
  );

  console.log("sequenceList", sequenceList);

  if (sequenceList.length === 0) {
    return json({ error: "No sequences provided" }, { status: 400 });
  }

  // Start the SSE production
  return produce(async ({ emit }) => {
    try {
      const payload = { sequenceList };
      console.log("api url", API_URL);
      const response = await fetch(`${API_URL}/api/predict`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(payload),
      });

      const reader = response.body?.getReader();
      const decoder = new TextDecoder("utf-8");
      let buffer = "";

      async function pushData() {
        try {
          const { done, value } = await reader.read();
          if (done) {
            if (buffer.trim()) {
              processChunk(buffer);
            }
            emit("end");
            return;
          }

          const chunk = decoder.decode(value, { stream: true });
          console.log("Received chunk:", chunk); // Add this line
          buffer += chunk;
          const lines = buffer.split("\n\n");
          buffer = lines.pop() || ""; // Keep the last incomplete chunk in the buffer

          for (const line of lines) {
            processChunk(line);
          }

          await pushData();
        } catch (error) {
          console.error("Error in pushData:", error);
        }
      }

      function processChunk(chunk) {
        console.log("Raw chunk received:", chunk);
        if (chunk.trim()) {
          if (chunk.startsWith("event: end")) {
            console.log("Received end event");
            emit("message", "end");
          } else if (chunk.startsWith("data: ")) {
            try {
              const jsonString = chunk.slice(6).trim();
              console.log("Attempting to parse JSON:", jsonString);
              const jsonData = JSON.parse(jsonString);
              console.log("Received data:", jsonData);
              emit("message", JSON.stringify(jsonData));
            } catch (error) {
              console.error("Error parsing JSON:", error);
              console.error("Problematic chunk:", chunk);
            }
          } else {
            console.log("Unexpected chunk format:", chunk);
          }
        }
      }

      await pushData();
    } catch (error) {
      console.error("SSE Production error:", error);
      emit("error", error);
    }
  });
};

/**
 * Split input sequences into a list of sequences, preserving headers and sequences together.
 * Handles both FASTA/FASTQ formatted sequences and plain sequences.
 * @param {string} sequences - The input sequences as a single string.
 * @returns {Array<string>} - An array of sequences.
 */
function splitSequences(sequences: string): Array<string> {
  const lines = sequences.split(/\r?\n/);
  let currentSequence = [];
  const result = [];
  let isFastaFastq = false;

  lines.forEach((line) => {
    if (line.startsWith(">") || line.startsWith("@")) {
      isFastaFastq = true;
      if (currentSequence.length > 0) {
        result.push(currentSequence.join("\n"));
        currentSequence = [];
      }
    } else if (isFastaFastq && line.startsWith("+")) {
      currentSequence.push(line);
      return;
    }
    currentSequence.push(line);
  });

  if (currentSequence.length > 0) {
    result.push(currentSequence.join("\n"));
  }

  if (!isFastaFastq) {
    return sequences.split(/\r?\n/).filter((seq) => seq.trim().length > 0);
  }

  return result;
}
