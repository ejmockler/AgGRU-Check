<script>
  import { source } from "sveltekit-sse";

  let output = "";
  let eventSourceValue;

  function handleFormSubmission(event) {
    event.preventDefault();

    const formData = new FormData(event.target);
    const sequences = formData.get("sequences");
    const payload = { sequences };

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
      output += value + "\n";
      console.log("Received message:", value);
    });
  }
</script>

<section style="z-index: 1">
  <h1>AgGRU-Check</h1>
  <form method="POST" action="/predict" on:submit={handleFormSubmission}>
    <textarea
      name="sequences"
      aria-label="Enter up to 5 protein sequences (either raw, FASTA or FASTQ)"
      placeholder="Enter up to 5 protein sequences (either raw, FASTA or FASTQ)"
    />
    <button type="submit">Submit</button>
  </form>
  <pre id="output">{output}</pre>
</section>

<style lang="scss">
  h1 {
    color: white;
    font-size: xx-large;
    text-align: center;
  }

  section {
    margin: 4em;
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
    min-height: 20vh;
    min-width: 60vw;
    color: black;
    background-color: whitesmoke;
  }

  pre {
    background-color: #f0f0f0;
    padding: 1em;
    white-space: pre-wrap;
  }
</style>
