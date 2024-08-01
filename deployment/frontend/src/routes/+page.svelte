<script lang="ts">
  import { source } from "sveltekit-sse";
  import Result from "./Result.svelte";
  import { browser } from "$app/environment";

  let output = "";
  let eventSourceValue;
  let isNewInput = true;
  let previousInput: FormDataEntryValue;
  let processedSequences = new Set();
  let results = [];

  function handleFormSubmission(event) {
    event.preventDefault();

    const formData = new FormData(event.target);
    const sequences = formData.get("sequences");

    if (sequences !== previousInput) {
      const payload = {
        sequences,
        processedSequences: Array.from(processedSequences),
      };
      console.log("Sending data:", payload);

      const actionUrl = event.target.action;

      eventSourceValue = source(actionUrl, {
        options: {
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify(payload),
        },
      }).select("message");

      eventSourceValue.subscribe((value) => {
        if (value.includes("data: end") || value.includes("event: end")) {
          return;
        }
        console.log(value);
        const jsonData = value
          .replace(/^data: /, "")
          .replace(/'/g, '"')
          .trim();

        console.log("Received data:", jsonData);
        try {
          const parsedData = JSON.parse(jsonData);
          processedSequences.add(parsedData.sequence);
          console.log("Received prediction:", parsedData);
          output += JSON.stringify(parsedData) + "\n";
          console.log(processedSequences.entries());

          // Parse results and group them by sequence
          const sequence = parsedData.sequence;
          const models = Object.keys(parsedData)
            .filter((key) => key.startsWith("model_"))
            .map((key) => ({
              model: key.split("_")[1],
              confidence: parsedData[key],
            }));

          // Add to results array
          const existingSequence = results.find(
            (result) => result.sequence === sequence
          );
          if (existingSequence) {
            existingSequence.models.push(...models);
          } else {
            results = [...results, { sequence, models }];
          }
          console.log("Results:", results);
        } catch (error) {
          console.error("Failed to parse JSON:", error);
        }
      });
      isNewInput = false;
      previousInput = sequences;
    }
  }

  const inputMessage = `Is your protein amyloidgenic?

Enter up to 5 protein sequences 
(either raw, FASTA or FASTQ)`;

  let aboutOpen = false;
  let aboutButtonRef: HTMLButtonElement;
  let modalRef: HTMLDivElement;

  function toggleAbout() {
    aboutOpen = !aboutOpen;
  }

  function handleClickOutside(event: MouseEvent) {
    if (
      aboutOpen &&
      modalRef &&
      !modalRef.contains(event.target as Node) &&
      !aboutButtonRef.contains(event.target as Node)
    ) {
      aboutOpen = false;
    }
  }

  $: if (browser) {
    if (aboutOpen) {
      setTimeout(() => {
        document.addEventListener("click", handleClickOutside);
      }, 0);
    } else {
      document.removeEventListener("click", handleClickOutside);
    }
  }
</script>

<svelte:window on:blur={() => (aboutOpen = false)} />

<section>
  <div class="header">
    <div class="header__links">
      <button
        class="about__link"
        on:click={toggleAbout}
        bind:this={aboutButtonRef}
      >
        about
      </button>
      {#if aboutOpen}
        <div class="about__modal" bind:this={modalRef}>
          <b>AgGRU-Check</b>
          <br />
          Classify aggregation-prone proteins, via gated recurrent units
          <br />
          <ul style="margin:1em">
            <li>
              <i>What's the output?</i> <br />A likelihood whether the input
              protein is amyloidogenic; between 0 and 1. High is "probably
              amyloid" while low is "maybe not".
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
        </div>
      {/if}
      <a
        href="https://github.com/ejmockler/AgGRU-Check?tab=readme-ov-file"
        target="_blank"
        class="about__link">github</a
      >
    </div>

    <h1>AgGRU? Check:</h1>
  </div>
  <form method="POST" action="/predict" on:submit={handleFormSubmission}>
    <textarea
      name="sequences"
      aria-label={inputMessage}
      placeholder={inputMessage}
      minlength="3"
      required
      on:input={() => {
        isNewInput = true;
      }}
    />
    <button type="submit">Submit</button>
  </form>
</section>

<ul class="results">
  {#each results as result, index}
    <Result
      sequence={result.sequence}
      models={result.models}
      count={index + 1}
    />
  {/each}
</ul>

<style lang="scss">
  .header {
    display: flex;
    position: relative;
    flex-direction: column;
    justify-content: space-between;
    align-items: center;
    padding: 0.5em;
    margin: 1em;
    background-color: slategray;
    border-radius: 7px;
    opacity: 0.9;
    z-index: 2;
    &__links {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 1em;
    }
  }

  .about {
    &__link {
      all: unset;
      color: gainsboro;
      text-decoration: underline;
      cursor: pointer;
    }
    &__modal {
      position: absolute;
      top: 0;
      left: 0;
      padding: 1em;
      background-color: white;
      color: black;
      border-radius: 7px;
      opacity: 0.9;
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
    background-color: slategray;
  }

  section {
    z-index: 1;
    margin: 25vh 0 2vh;
    display: flex;
    position: relative;
    flex-direction: column;
    justify-content: center;
    align-items: center;
  }

  form {
    display: flex;
    justify-content: flex-start;
    flex-direction: column;
    gap: 0.75rem;
  }

  button {
    height: 2rem;
    background-color: slategray;
    color: white;
    user-select: text;
  }

  textarea {
    font-family: "Courier New", Courier, monospace;
    font-size: calc(7pt + 0.5vw + 0.5vh);
    min-height: 20vh;
    min-width: 60vw;
    color: black;
    background-color: whitesmoke;
    border: 5px solid skyblue;
    opacity: 0.8;
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
    padding-left: 0;
    margin-bottom: 1em;

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
</style>
