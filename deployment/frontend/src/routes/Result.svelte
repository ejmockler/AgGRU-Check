<script>
  export let sequence;
  export let models;

  // Function to get color based on confidence level
  const getColor = (confidence) => {
    const red = Math.min(255, 255 - Math.round(confidence * 255));
    const green = Math.min(255, Math.round(confidence * 255));
    return `rgb(${red},${green},0)`;
  };
</script>

<div class="result-container">
  <div class="sequence">{sequence}</div>
  {#each models as { model, confidence }}
    <div class="model-container">
      <div class="model-label">Model {Number(model) + 1}:</div>
      <div
        class="confidence-bar"
        style="background-color: {getColor(confidence)}; width: {confidence *
          100}%"
      ></div>
      <div class="confidence-label">{confidence.toFixed(2)}</div>
    </div>
  {/each}
</div>

<style>
  .result-container {
    margin: 1em 0;
    padding: 1em;
    border: 1px solid #ccc;
    border-radius: 8px;
    width: 100%;
    background-color: #f9f9f9;
    width: 220px;
    height: 105px;
  }

  .sequence {
    font-weight: bold;
    font-size: 1.2em;
    margin-bottom: 0.5em;
    max-width: 100%;
    text-overflow: ellipsis;
    overflow: hidden;
  }

  .model-container {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.75em;
  }

  .model-label {
    font-size: 0.9em;
    white-space: nowrap;
    text-wrap: nowrap;
  }

  .confidence-bar {
    height: 1em;
    width: 100%;
    margin-left: 0.5em;
    border-radius: 4px;
    transition: width 0.3s ease;
  }

  .confidence-label {
    font-size: 0.9em;
  }
</style>
