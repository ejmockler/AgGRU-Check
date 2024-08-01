<script>
  export let sequence;
  export let models;
  export let count;

  // Function to get color based on confidence level
  const getColor = (confidence) => {
    const red = Math.min(255, 255 - Math.round(confidence * 255));
    const green = Math.min(255, Math.round(confidence * 255));
    return `rgb(${red},${green},0)`;
  };
</script>

<div class="result-container">
  <span class="count">{count}</span>
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
    padding: 1em;
    border: 1px solid #ccc;
    border-radius: 8px;
    background-color: #f9f9f9;
    width: 105px;
    min-width: 105px;
    height: 105px;
    position: relative;
  }

  .sequence {
    font-weight: bold;
    font-size: 1.2em;
    margin-bottom: 0.5em;
    max-width: 100%;
    text-overflow: ellipsis;
    overflow: hidden;
  }

  .count {
    font-size: 0.6em;
    font-weight: bold;
    margin-bottom: 0.5em;
    width: 10px;
    height: 10px;
    line-height: 10px;
    text-align: center;
    position: absolute;
    top: 4px;
    right: 4px;
    background-color: gainsboro;
    border-radius: 100%;
    padding: 0.3em;
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
