<script lang="ts">
  import { source } from "sveltekit-sse";
  import Result from "./Result.svelte";

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

  const inputMessage = `Could a protein be amyloidgenic?

Enter up to 5 protein sequences 
(either raw, FASTA or FASTQ)`;
</script>

<section>
  <div class="header">
    <a href="#">about</a>
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

<ul>
  {#each results as result}
    <Result sequence={result.sequence} models={result.models} />
  {/each}
</ul>

<style lang="scss">
  .header {
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    align-items: center;
    padding: 0.5em;
    margin: 1em;
    background-color: slategray;
    border-radius: 7px;
    opacity: 0.9;
  }

  a {
    color: gainsboro;
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
    display: flex;
    position: sticky;
    margin: 0;
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
  }

  textarea {
    font-family: "Courier New", Courier, monospace;
    font-size: 11pt;
    min-height: 20vh;
    min-width: 60vw;
    color: black;
    background-color: whitesmoke;
    border: 5px solid skyblue;
    opacity: 0.7;
  }

  ul {
    display: flex;
    flex-direction: row;
    justify-content: center;
    align-items: center;
    gap: 1rem;
    z-index: 1;
    opacity: 0.9;
  }
</style>
