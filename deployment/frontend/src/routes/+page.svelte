<script lang="ts">
  import { onMount, onDestroy } from "svelte";
  import { source } from "sveltekit-sse";
  import Result from "./Result.svelte";
  import { browser } from "$app/environment";
  import Modal from "./Modal.svelte";

  let isNewInput = true;
  let previousInput: FormDataEntryValue;
  let sequenceInput: HTMLTextAreaElement;
  let processedSequences = new Set();
  let results = [];
  let isLoading = false;

  let eventSourceConnection;

  async function handleFormSubmission(event) {
    event.preventDefault();

    const formData = new FormData(event.target);
    const sequences = formData.get("sequences");

    if (sequences !== previousInput) {
      // Remove the global isLoading and make it specific to each result
      const sequenceList = splitSequences(sequences);

      // Initialize results array with isLoading set to true for each result
      results = sequenceList.map((sequence) => ({
        sequence: sequence,
        models: [],
        error: null,
        isLoading: true, // Each result starts with loading state true
      }));

      const payload = {
        sequences,
        processedSequences: Array.from(processedSequences),
      };

      const actionUrl = event.target.action;

      if (eventSourceConnection) {
        eventSourceConnection.close();
      }

      eventSourceConnection = source(actionUrl, {
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
        if (message.startsWith('{"error":')) {
          const errorData = JSON.parse(message).error;
          processError(errorData);
        } else if (message === "end") {
          // Handle the end of the stream
          results.forEach((result) => (result.isLoading = false));
          eventSourceConnection.close();
          unsubscribe();
        } else if (message && message.trim() !== "") {
          try {
            const parsedData = JSON.parse(message);
            processData(parsedData);
          } catch (error) {
            console.error("Failed to parse JSON:", error, "Data:", message);
          }
        }
      });

      isNewInput = false;
      previousInput = sequences;
    }
  }

  function sanitizeSequence(sequence: string): string {
    return sequence.replace(/\s+/g, "").toLocaleUpperCase(); // Remove all whitespace and convert to uppercase
  }

  function processData(parsedData) {
    const modelKey = Object.keys(parsedData).find((key) =>
      key.startsWith("model_")
    );
    const confidence = parsedData[modelKey];
    const sequence = parsedData.sequence?.toLocaleUpperCase(); // Ensure sequence is in uppercase for comparison

    if (sequence) {
      // Find the latest result whose sequence contains the sanitized server-returned sequence
      const resultIndex = results.findIndex((result) =>
        sanitizeSequence(result.sequence).includes(sequence)
      );

      if (resultIndex !== -1) {
        const existingResult = results[resultIndex];

        // Add the model result to the result models array
        existingResult.models.push({ model: modelKey, confidence });

        // Mark this specific result as loaded
        existingResult.isLoading = false;

        results = [...results]; // Trigger reactivity
      }
    } else {
      // If no sequence is provided, handle it as an error or update the first result with no error
      const fallbackIndex = results.findIndex(
        (result) => result.error === null && result.models.length === 0
      );

      if (fallbackIndex !== -1) {
        const fallbackResult = results[fallbackIndex];
        fallbackResult.models.push({ model: modelKey, confidence });
        fallbackResult.isLoading = false;
        results = [...results]; // Trigger reactivity
      } else {
        console.error("No valid result found to update");
      }
    }
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

Enter up to 15 amino acid sequences 
(either raw, FASTA or FASTQ)`;

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
</script>

<section>
  <div class="hero">
    <div class="header">
      <div class="header__links">
        <button class="about__link" on:click={() => toggleModal("about")}
          >about</button
        >
        <button class="about__link" on:click={() => toggleModal("terms")}
          >terms</button
        >
        <a
          href="https://github.com/ejmockler/AgGRU-Check?tab=readme-ov-file"
          target="_blank"
          class="about__link">github</a
        >
      </div>

      <h1>AgGRU? Check:</h1>

      <div class="link-container">
        <a
          href="#"
          class="fasta-link"
          on:click={() =>
            appendFastaToTextarea(
              ">TDP-43 Amyloid Domain (318-343)\nINPAMMAAAQAALKSSWGMMGMLASQ"
            )}
        >
          TDP-43 (amyloid domain)
        </a>
        <a
          href="#"
          class="fasta-link"
          on:click={() =>
            appendFastaToTextarea(
              ">MOTS-c (short mitochondrial protein)\nMRWQEMGYIFYPRKLR"
            )}
        >
          MOTS-c (short mitochondrial protein)
        </a>
      </div>
    </div>

    {#if activeModal === "about"}
      <Modal title="AgGRU-Check" on:close={() => (activeModal = null)}>
        <p>Classify aggregation-prone proteins, via gated recurrent units</p>
        <ul style="margin:1em">
          <li>
            <i>What's the output?</i> <br />A likelihood whether the input
            protein is amyloidogenic—between 0 and 1. High is "probably amyloid"
            while low is "maybe not". Likelihoods are provided in triplicate
            across an ensemble of 3 models—each model has been trained on a
            different slice of data.
          </li>
          <li>
            <i>How does it work?</i> <br />The model is a recurrent neural
            network, trained on a dataset of known amyloidogenic proteins.
          </li>
          <li>
            <i>Where can I learn more?</i> <br /> Check out the
            <a
              href="https://github.com/ejmockler/AgGRU-Check?tab=readme-ov-file"
              target="_blank">GitHub</a
            >
            and
            <a
              href="https://github.com/ejmockler/AgGRU-Check/blob/main/details.pdf"
              target="_blank">technical paper</a
            > to run this model for yourself & read more about this implementation.
          </li>
          <li>
            <i>Can I cite this?</i>
            <pre class="citation">
@misc&#123;
  ejmockler2024aggrucheck,
  author = &#123;Eric Jing Mockler&#125;,
  title = &#123;Gated recurrent units classify amyloidogenic proteins from sequence alone&#125;,
  year = &#123;2024&#125;,
  publisher = &#123;Zenodo&#125;,
  doi = &#123;10.5281/zenodo.13147167&#125;,
&#125;
            </pre>
          </li>
        </ul>
      </Modal>
    {/if}

    {#if activeModal === "terms"}
      <Modal title="Terms of Use" on:close={() => (activeModal = null)}>
        <div class="terms-of-service">
          <ol>
            <li>
              <strong>Purpose:</strong> AgGRU-Check is provided for research and
              educational purposes only.
            </li>
            <li>
              <strong>No Warranty:</strong> The service is provided "as is" without
              any guarantees of accuracy or availability.
            </li>
            <li>
              <strong>Usage Limits:</strong> Users are limited to 5 protein sequences
              per query to ensure fair usage.
            </li>
            <li>
              <strong>Data Privacy:</strong> We do not store or share submitted protein
              sequences.
            </li>
            <li>
              <strong>Citation:</strong> Users should cite the tool as specified
              in the "about" section when used in publications.
            </li>
            <li>
              <strong>Responsible Use:</strong> Users agree to use the tool responsibly
              and not overwhelm the service.
            </li>
            <li>
              <strong>Modifications:</strong> We reserve the right to modify or discontinue
              the service at any time.
            </li>
            <li>
              <strong>Liability:</strong> We are not liable for any damages or losses
              resulting from the use of AgGRU-Check.
            </li>
            <li>
              <strong>Agreement:</strong> By using AgGRU-Check, you agree to these
              terms.
            </li>
          </ol>
        </div>
      </Modal>
    {/if}

    <form method="POST" action="/predict" on:submit={handleFormSubmission}>
      <textarea
        name="sequences"
        bind:this={sequenceInput}
        aria-label={inputMessage}
        placeholder={inputMessage}
        minlength="3"
        required
        on:input={() => {
          isNewInput = true;
        }}
      />
      <button type="submit">Infer</button>
    </form>

    <ul class="results">
      {#each results as result, index}
        <Result
          sequence={result.sequence}
          models={result.models}
          count={index + 1}
          error={result.error}
          {isLoading}
        />
      {/each}
    </ul>
  </div>
</section>

<style lang="scss">
  .hero {
    position: relative;
    margin-top: 25vh;
  }
  .header {
    display: flex;
    position: relative;
    flex-direction: column;
    justify-content: space-between;
    align-items: center;
    padding: 0.5em;
    margin: 1em;
    background: linear-gradient(163deg, #ffd166, #ef476f, #118ab2, #073b4c);
    background-size: 300% 300%;
    animation: headerGradient 120s ease infinite;
    border-radius: 7px;
    opacity: 0.7;
    z-index: 2;
    &__links {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 1em;
    }
  }
  @keyframes headerGradient {
    0% {
      background-position: 0% 83%;
    }
    50% {
      background-position: 100% 16%;
    }
    100% {
      background-position: 0% 83%;
    }
  }
  .about {
    &__link {
      all: unset;
      color: gainsboro;
      text-decoration: underline;
      cursor: pointer;
    }

    &__link:hover {
      color: white;
    }

    &__modal {
      position: absolute;
      top: 0;
      left: 0;
      padding: 1em;
      background-color: white;
      color: black;
      border-radius: 7px;
      opacity: 1;
      z-index: 10;
      user-select: text;
    }
  }

  h1 {
    color: white;
    font-size: xx-large;
    text-align: center;
    padding: 0.25em;
    margin: 0.25em;
    margin-bottom: 0.1em;
    padding-bottom: 0;
  }

  section {
    z-index: 1;
    display: block;
    margin: auto;
    position: relative;
    width: 50%;
  }

  form {
    display: flex;
    justify-content: flex-start;
    flex-direction: column;
    gap: 0.75rem;
  }

  button {
    height: 2rem;
    background: linear-gradient(163deg, #ef476f, #118ab2, #073b4c);
    background-size: 300% 300%;
    animation: headerGradient 120s ease infinite;
    color: white;
    user-select: text;
  }

  textarea {
    font-family: "Courier New", Courier, monospace;
    font-size: calc(7pt + 0.5vw + 0.5vh);
    min-height: 20vh;
    color: black;
    background-color: whitesmoke;
    border: 5px solid skyblue;
    opacity: 0.8;
  }

  .link-container {
    text-align: center;

    .fasta-link {
      color: gainsboro;
      text-decoration: underline;
      cursor: pointer;
      margin: 0.2em;
      display: inline-block;

      &:hover {
        color: white;
      }
    }
  }

  ul {
    display: flex;
    flex-wrap: wrap;
    flex-direction: row;
    justify-content: center;
    align-items: center;
    gap: 1rem;
    z-index: 1;
    opacity: 0.9;
    padding: 1em 0;
    position: relative;
    left: 0;
    right: 0;

    & .results {
      margin: 1em;
    }
  }

  .citation {
    font-family: "Courier New", Courier, monospace;
    font-size: 0.9em;
    background-color: #f5f5f5;
    border: 1px solid #ddd;
    border-radius: 4px;
    font-size: x-small;
    padding: 0.5em;
    overflow-x: auto;
    white-space: pre-wrap;
    word-wrap: break-word;
  }

  .terms-of-service {
    font-family: Arial, sans-serif;
    line-height: 1.6;
    color: #333;
    max-width: 800px;
    margin: 0 auto;
    padding: 20px;
    border-top: 2px solid #3498db;

    ol {
      counter-reset: item;
      padding-left: 0;
    }

    li {
      display: block;
      margin-bottom: 15px;
      position: relative;
      padding-left: 35px;

      &:before {
        content: counter(item) ".";
        counter-increment: item;
        position: absolute;
        left: 0;
        top: 0;
        font-weight: bold;
        color: #3498db;
      }

      strong {
        color: #2c3e50;
        font-weight: 600;
      }
    }
  }
</style>
