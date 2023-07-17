<script>
  import { enhance } from "$app/forms";
</script>

<section>
  <h1>AgGRU-Check</h1>
  <form
    method="POST"
    action="?/predict"
    use:enhance={({ formElement, formData, action, cancel, submitter }) => {
      // `formElement` is this `<form>` element
      // `formData` is its `FormData` object that's about to be submitted
      // `action` is the URL to which the form is posted
      // calling `cancel()` will prevent the submission
      // `submitter` is the `HTMLElement` that caused the form to be submitted
      return async ({ result, update }) => {
        // `result` is an `ActionResult` object
        // `update` is a function which triggers the default logic that would be triggered if this callback wasn't set
      };
    }}
  >
    <textarea
      name="sequences"
      aria-label="Enter up to 5 protein sequences (raw, FASTA or FASTQ)"
      placeholder="Enter up to 5 protein sequences (raw, FASTA or FASTQ)"
    />
    <button>Submit</button>
  </form>
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
    color: white;
    background-color: darkslategray;
  }
</style>
