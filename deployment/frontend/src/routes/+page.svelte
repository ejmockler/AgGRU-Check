<script lang="ts">
  import { onMount, onDestroy } from "svelte";
  import { source } from "sveltekit-sse";
  import Result from "./Result.svelte";
  import { browser } from "$app/environment";
  import Modal from "./Modal.svelte";
  import InputPanel from '$lib/components/InputPanel.svelte';
  import AboutContent from '$lib/components/AboutContent.svelte';
  import TermsContent from '$lib/components/TermsContent.svelte';
  import type { PositionResult } from '$lib/types';

  let isNewInput = true;
  let previousInput: FormDataEntryValue;
  let sequenceInput: HTMLTextAreaElement;
  let processedSequences = new Set();
  let results: {
    sequence: string;
    results: PositionResult[];
    error: string | null;
    isLoading: boolean;
    sequence_index: number;
    progress?: {
      position: number;
      totalLength: number;
      models_completed: number;
      total_models: number;
    };
  }[] = [];
  let isLoading = false;

  let eventSourceConnection: any;

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

  function findMatchingSequenceIndex(results: { sequence_index: number } [], targetIndex: number): number {
    return results.findIndex(r => r.sequence_index === targetIndex);
  }

  async function handleSequenceSubmit(sequences: string) {
    if (!sequences) return;
    
    isLoading = true;
    
    // Initialize results array with loading state immediately
    const sequenceList = splitSequences(sequences);
    results = sequenceList.map((sequence, index) => ({
      sequence,
      results: [],
      error: null,
      isLoading: true,
      sequence_index: index,
      progress: undefined
    }));

    const payload = {
      sequences,
      processedSequences: Array.from(processedSequences),
    };

    try {
      if (eventSourceConnection) {
        eventSourceConnection.close();
      }

      eventSourceConnection = source("/predict", {
        options: {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify(payload),
        },
      });

      const messageStore = eventSourceConnection.select("message");

      const unsubscribe = messageStore.subscribe((message) => {
        console.log("Received SSE message:", message);
        
        if (message === "end") {
          // Only close connection after all sequences are done
          const allComplete = results.every(result => !result.isLoading);
          if (allComplete) {
            isLoading = false;
            eventSourceConnection.close();
            unsubscribe();
          }
        } else if (message && message.trim() !== "") {
          try {
            const parsedData = JSON.parse(message);
            console.log("Parsed message data:", {
              type: parsedData.type,
              sequenceIndex: parsedData.sequence_index,
              data: parsedData
            });

            if (parsedData.type === 'progress') {
              const resultIndex = findMatchingSequenceIndex(results, parsedData.sequence_index);
              if (resultIndex !== -1) {
                // Update progress information
                results[resultIndex].progress = {
                  position: parsedData.partialResults.length - 1,
                  totalLength: parsedData.partialResults.length,
                  models_completed: parsedData.models_completed,
                  total_models: parsedData.total_models
                };
                
                // Update partial results with confidence
                if (parsedData.partialResults) {
                  results[resultIndex].results = parsedData.partialResults.map(r => ({
                    ...r,
                    isUpdated: true // Mark as recently updated for animation
                  }));
                }
                results = [...results]; // Trigger reactivity
              }
            } else if (parsedData.type === 'model_complete') {
              const resultIndex = findMatchingSequenceIndex(results, parsedData.sequence_index);
              if (resultIndex !== -1) {
                // If all models are complete for this sequence, clear progress
                if (parsedData.models_completed === parsedData.total_models) {
                  results[resultIndex].progress = undefined;
                  results[resultIndex].isLoading = false;
                } else if (results[resultIndex].progress) {
                  // Keep existing progress info but update model count
                  results[resultIndex].progress = {
                    ...results[resultIndex].progress!,
                    models_completed: parsedData.models_completed,
                    total_models: parsedData.total_models,
                    // Keep current position to avoid visual jump
                    position: results[resultIndex].progress.position,
                    totalLength: results[resultIndex].progress.totalLength
                  };
                }
                results = [...results];
              }
            } else if (parsedData.type === 'result') {
              const resultIndex = findMatchingSequenceIndex(results, parsedData.sequence_index);
              if (resultIndex !== -1) {
                results[resultIndex].results = parsedData.results;
                results[resultIndex].isLoading = false;
                // Clear the progress when sequence is complete
                results[resultIndex].progress = undefined;
                results = [...results];
              }
            } else if (parsedData.type === 'error') {
              const resultIndex = findMatchingSequenceIndex(results, parsedData.sequence_index);
              if (resultIndex !== -1) {
                results[resultIndex].error = parsedData.error;
                results[resultIndex].isLoading = false;
                results = [...results];
              }
            }
          } catch (error) {
            console.error("Failed to parse message:", error, "Data:", message);
          }
        }
      });

    } catch (error) {
      console.error("Error during submission:", error);
      processError("Failed to connect to server");
      isLoading = false;
    }
  }

  function sanitizeSequence(sequence: string): string {
    return sequence.replace(/\s+/g, "").toLocaleUpperCase(); // Remove all whitespace and convert to uppercase
  }

  function processError(errorData) {
    // Update all results to show the error
    results = results.map(result => ({
      ...result,
      error: typeof errorData === 'string' ? errorData : 'Server error occurred',
      isLoading: false
    }));
  }

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

  function appendFastaToTextarea(fastaContent: string) {
    if (sequenceInput) {
      const currentValue = sequenceInput.value.trim();

      // Check if the content already exists
      if (!currentValue.includes(fastaContent)) {
        const newValue = currentValue
          ? `${currentValue}\n${fastaContent}`
          : fastaContent;
        sequenceInput.value = newValue;
        isNewInput = true; // Mark the input as new
      }
    }
  }

  const inputMessage = `Is your protein amyloidgenic?

Enter up to 5 amino acid sequences
(either raw or FASTA/FASTQ)`;

  let activeModal = null;

  function toggleModal(modalName) {
    activeModal = activeModal === modalName ? null : modalName;
  }

  let handleClickOutside;

  onMount(() => {
    handleClickOutside = (event) => {
      const modalElement = document.querySelector(".modal");
      if (modalElement && !modalElement.contains(event.target)) {
      }
    };
  });

  $: if (browser && activeModal) {
    document.addEventListener("click", handleClickOutside);
  } else if (browser) {
    document.removeEventListener("click", handleClickOutside);
  }

  onDestroy(() => {
    if (browser) {
      document.removeEventListener("click", handleClickOutside);
    }
  });

  function createCSV(results: typeof results): string {
    // Create CSV header
    const header = 'Sequence Name,Position,Amino Acid,Amyloid Propensity,Confidence\n';
    
    // Create CSV rows for each sequence and position
    const rows = results.flatMap(result => {
      // Extract sequence name from FASTA header if present
      const sequenceName = result.sequence.split('\n')[0].replace(/^>/, '').trim();
      
      return result.results.map((pos, index) => {
        // Get amino acid at this position from the sequence
        const sequence = extractSequence(result.sequence);
        const aminoAcid = sequence[index];
        
        return [
          `"${sequenceName}"`,  // Quote sequence name to handle commas
          index + 1,            // 1-based position
          aminoAcid,
          pos.score.toFixed(3),
          pos.confidence.toFixed(3)
        ].join(',');
      });
    }).join('\n');

    return header + rows;
  }

  function downloadCSV() {
    const csv = createCSV(results);
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.setAttribute('hidden', '');
    a.setAttribute('href', url);
    a.setAttribute('download', 'amyloid_predictions.csv');
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    window.URL.revokeObjectURL(url);
  }
</script>

<div class="app-wrapper">
  <!-- Molecular Background -->
  <div class="molecular-background">
    <!-- We'll add subtle animated circles here -->
  </div>

  <div class="app-container">
    <InputPanel 
      onSubmit={handleSequenceSubmit}
      isLoading={isLoading}
    >
      <div slot="nav-links">
        <button class="nav-link" on:click|stopPropagation={() => toggleModal("about")}>about</button>
        <button class="nav-link" on:click|stopPropagation={() => toggleModal("terms")}>terms</button>
        <button class="nav-link" on:click|stopPropagation={() => window.open("https://github.com/ejmockler/AgGRU-Check", "_blank")}>github</button>
      </div>
    </InputPanel>

    <!-- Right Panel -->
    <main class="results-panel glass-panel">
      {#if results.length === 0}
        <div class="empty-state">
          <div class="molecular-placeholder" />
          <p>Enter a protein sequence to see predictions</p>
        </div>
      {:else}
        <div class="results-container">
          <!-- Add download button when all sequences are complete -->
          {#if results.every(r => !r.isLoading && !r.error)}
            <button 
              class="download-button"
              on:click={downloadCSV}
            >
              <svg viewBox="0 0 20 20" fill="currentColor" class="download">
                <path fill-rule="evenodd" d="M10 3a1 1 0 00-1 1v6.293L6.707 8a1 1 0 10-1.414 1.414l4 4a1 1 0 001.414 0l4-4a1 1 0 00-1.414-1.414L11 10.293V4a1 1 0 00-1-1z" clip-rule="evenodd" />
                <path fill-rule="evenodd" d="M3 17a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1z" clip-rule="evenodd" />
              </svg>
              Download Results (CSV)
            </button>
          {/if}

          {#each results as result (result.sequence_index)}
            <Result
              sequence={result.sequence}
              results={result.results}
              count={result.sequence_index + 1}
              error={result.error}
              isLoading={result.isLoading}
              progress={result.progress}
            />
          {/each}
        </div>
      {/if}
    </main>
  </div>

  {#if activeModal === "about"}
    <Modal on:close={() => activeModal = null} title="AgGRU-Check">
      <AboutContent />
    </Modal>
  {/if}

  {#if activeModal === "terms"}
    <Modal on:close={() => activeModal = null} title="Terms of Use">
      <TermsContent />
    </Modal>
  {/if}
</div>

<style lang="scss">
  // Core variables
  $colors: (
    primary: #2A9D8F,
    secondary: #264653,
    surface: #FFFFFF,
    background: #F8FAFC,
    panel: #F1F5F9,
    text: #1A2B3B,
    border: #E2E8F0
  );

  // Z-index system
  $z-layers: (
    background: -1,
    canvas: 0,
    content: 1,
    panels: 2,
    modals: 3
  );

  .app-wrapper {
    position: relative;
    min-height: 100vh;
    z-index: map-get($z-layers, content);
  }

  // Glass Panel Styling
  .glass-panel {
    background: rgba(255, 255, 255, 0.25);  // Increased base opacity
    backdrop-filter: blur(4px);  // Increased blur for better text contrast
    border: 1px solid rgba(255, 255, 255, 0.3);
    box-shadow: 
      0 4px 24px rgba(0, 0, 0, 0.08),
      inset 0 0 0 1px rgba(255, 255, 255, 0.2);
  }

  .app-container {
    position: relative;
    display: grid;
    grid-template-columns: minmax(400px, 35%) 1fr;
    min-height: 100vh;
    gap: 2rem;
    padding: 2rem;
    z-index: map-get($z-layers, panels);
  }

  // Input Panel - Make it sticky
  :global(.input-panel) {
    position: sticky;
    top: 2rem;
    max-height: calc(100vh - 4rem); // Full height minus padding
    overflow-y: auto;
    overflow-x: hidden;
  }

  // Input Panel
  .input-panel {
    border-radius: 24px;
    overflow: hidden;
    display: flex;
    flex-direction: column;
    gap: 2rem;

    .branding {
      h1 {
        font-size: 2.8rem;
        font-weight: 800;
        color: rgba(26, 43, 59, 0.95);  // Solid color instead of gradient
        margin: 0.5rem 0 1.5rem;
        text-shadow: 0 1px 2px rgba(255, 255, 255, 0.2);
      }
    }

    .nav-links {
      padding: 0.75rem 0;
      border-bottom: 1px solid rgba(42, 157, 143, 0.2);
      
      .nav-link {
        font-size: 0.875rem;
        font-weight: 600;
        color: rgba(42, 157, 143, 0.9);  // Increased contrast
        padding: 0.5rem 0.75rem;
        border-radius: 6px;
        
        &:hover {
          background: rgba(42, 157, 143, 0.15);
          color: rgba(42, 157, 143, 1);
        }
      }
    }
  }

  // Form Styling
  form {
    display: flex;
    flex-direction: column;
    gap: 1.5rem;
    flex-grow: 1;

    .example-sequences {
      display: flex;
      gap: 0.75rem;
      flex-wrap: wrap;
    }

    .sequence-pill {
      background: rgba(42, 157, 143, 0.15);
      color: rgba(42, 157, 143, 0.9);  // Increased contrast
      font-weight: 600;
      border: 1px solid rgba(42, 157, 143, 0.3);
      letter-spacing: 0.01em;
      padding: 0.5rem 1rem;
      border-radius: 20px;
      font-size: 0.875rem;
      cursor: pointer;
      transition: all 0.2s;

      &:hover {
        background: rgba(42, 157, 143, 0.25);
        color: rgba(42, 157, 143, 1);
      }
    }

    textarea {
      flex-grow: 1;
      min-height: 200px;
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.875rem;
      line-height: 1.7;  // Increased line height
      letter-spacing: 0.02em;  // Slightly increased spacing
      padding: 1.5rem;
      border-radius: 12px;
      background: rgba(255, 255, 255, 0.3);  // Increased background opacity
      border: 1px solid rgba(42, 157, 143, 0.3);
      color: rgba(26, 43, 59, 0.95);  // Darker text
      resize: none;
      transition: all 0.2s;
      
      &::placeholder {
        color: rgba(26, 43, 59, 0.6);  // More visible placeholder
      }
      
      &:focus {
        outline: none;
        background: rgba(255, 255, 255, 0.4);
        border-color: rgba(42, 157, 143, 0.8);
      }
    }

    .submit-button {
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 0.5rem;
      background: rgba(42, 157, 143, 0.95);  // More solid button
      color: white;
      padding: 1rem;
      border-radius: 12px;
      font-weight: 600;
      transition: all 0.2s;
      font-size: 0.875rem;
      letter-spacing: 0.02em;
      text-transform: uppercase;
      border: 1px solid rgba(42, 157, 143, 0.3);

      svg {
        width: 1.25rem;
        height: 1.25rem;
        transition: transform 0.2s;
      }

      &:hover {
        background: rgba(42, 157, 143, 1);
        box-shadow: 0 4px 12px rgba(42, 157, 143, 0.3);
        
        svg {
          transform: translateX(2px);
        }
      }
    }
  }

  // Results Panel
  .results-panel {
    position: relative;
    z-index: 1;
    border-radius: 24px;
    height: fit-content;
    min-height: calc(100vh - 4rem);
    overflow: visible; // Allow tooltips to overflow
    
    // Empty state styling
    .empty-state {
      height: 100%;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      gap: 2rem;
      opacity: 0.9;  // Increased opacity
      
      .molecular-placeholder {
        width: 120px;
        height: 120px;
        border-radius: 50%;
        background: linear-gradient(
          45deg,
          rgba(42, 157, 143, 0.4),
          rgba(42, 157, 143, 0.2)
        );
        position: relative;
        
        &::before {
          content: '';
          position: absolute;
          inset: -10px;
          border-radius: 50%;
          border: 2px dashed rgba(42, 157, 143, 0.6);
        }
      }

      p {
        font-size: 1rem;
        font-weight: 600;
        color: rgba(26, 43, 59, 0.95);  // Much darker text
        letter-spacing: 0.01em;
      }
    }

    .results-container {
      position: relative;
      z-index: 1;
      max-width: 900px;
      margin: 0 auto;
      display: grid;
      gap: 1.5rem;
      overflow: visible; // Allow tooltips to overflow
      
      :global(.result-card) {
        background: rgba(255, 255, 255, 0.3);  // Increased opacity
        border-radius: 16px;
        border: 1px solid rgba(42, 157, 143, 0.3);
        padding: 1.5rem;
        transition: all 0.2s;

        &:hover {
          transform: translateY(-2px);
          background: rgba(255, 255, 255, 0.35);
          border-color: rgba(42, 157, 143, 0.4);
        }
      }
    }
  }

  @keyframes rotate {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
  }

  // Ensure modals appear above everything
  :global(.modal) {
    z-index: map-get($z-layers, modals);
  }

  // Responsive Design
  @media (max-width: 1024px) {
    .app-container {
      grid-template-columns: 1fr;
      height: auto;
    }

    :global(.input-panel) {
      position: relative;
      top: 0;
      height: auto;
      max-height: none;
    }

    .results-panel {
      height: auto;
      min-height: 0;
    }
  }

  .download-button {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0.5rem;
    background: rgba(42, 157, 143, 0.95);
    color: white;
    padding: 0.75rem 1.25rem;
    border-radius: 8px;
    font-weight: 600;
    font-size: 0.875rem;
    letter-spacing: 0.02em;
    border: none;
    cursor: pointer;
    transition: all 0.2s;
    margin-bottom: 1rem;
    width: fit-content;
    margin-left: auto;
    
    svg {
      width: 1.25rem;
      height: 1.25rem;
    }
    
    &:hover {
      background: rgba(42, 157, 143, 1);
      transform: translateY(-1px);
      box-shadow: 0 4px 12px rgba(42, 157, 143, 0.2);
    }
    
    &:active {
      transform: translateY(0);
    }
  }
</style>
