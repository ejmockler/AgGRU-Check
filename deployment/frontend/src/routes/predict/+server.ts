import { produce } from "sveltekit-sse";
import { json } from "@sveltejs/kit";
import { API_URL } from "$env/static/private";
import type { RequestHandler } from "./$types";

export const POST: RequestHandler = async ({ request }) => {
  const data = await request.json();
  const sequences = data.sequences;

  // Split the input into individual sequences
  const sequenceList = splitSequences(sequences).filter(
    (seq) => !data.processedSequences.includes(seq.toLocaleUpperCase())
  );

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
        try {
          console.log("Raw chunk received:", chunk);
          if (chunk.trim()) {
            if (chunk.startsWith("event: end")) {
              console.log("Received end event");
              emit("message", "end");
            } else if (chunk.startsWith("event: error")) {
              const errorData = chunk.slice(chunk.indexOf("data: ") + 6).trim();
              const error = { error: errorData };
              emit("message", JSON.stringify(error));
            } else if (chunk.startsWith("data: ")) {
              const jsonString = chunk.slice(6).trim();
              console.log("Attempting to parse JSON:", jsonString);
              const jsonData = JSON.parse(jsonString);
              console.log("Received data:", jsonData);
              emit("message", JSON.stringify(jsonData)); // Process data as normal
            } else {
              console.log("Unexpected chunk format:", chunk);
            }
          }
        } catch (error) {
          console.error("Error processing chunk:", error);
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
