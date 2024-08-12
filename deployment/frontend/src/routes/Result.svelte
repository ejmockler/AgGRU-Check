<script>
  export let sequence;
  export let models;
  export let count;

  // Function to get color based on confidence level
  const getColor = (confidence) => {
    const hue = confidence * 120; // 0 is red, 120 is green
    return `hsl(${hue}, 100%, 50%)`;
  };

  // Sort models by confidence in descending order
  $: sortedModels = models.sort((a, b) => b.confidence - a.confidence);
</script>

<div class="result-container">
  <span class="count">{count}</span>
  <div class="sequence">{sequence}</div>
  {#each sortedModels as { model, confidence }, i}
    <div class="model-container">
      <div class="model-label">Model {Number(model.split("_")[1]) + 1}:</div>
      <div class="bar-container">
        <div
          class="confidence-bar"
          style="width: {confidence * 100}%; background-color: {getColor(
            confidence
          )};"
        ></div>
        <div class="confidence-label">
          {(confidence * 100).toFixed(1)}%
        </div>
      </div>
    </div>
  {/each}
</div>

<style>
  .result-container {
    padding: 1em;
    border: 1px solid #ccc;
    border-radius: 8px;
    background-color: #f9f9f9;
    position: relative;
    width: 300px;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
  }

  .sequence {
    font-weight: bold;
    font-size: 1.2em;
    margin-bottom: 1em;
    max-width: 100%;
    text-overflow: ellipsis;
    overflow: hidden;
    white-space: nowrap;
  }

  .count {
    position: absolute;
    top: -10px;
    left: -10px;
    background-color: #007bff;
    color: white;
    width: 24px;
    height: 24px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: bold;
    font-size: 0.9em;
  }

  .model-container {
    display: flex;
    align-items: center;
    margin-bottom: 0.75em;
  }

  .model-label {
    font-size: 0.9em;
    width: 80px;
    white-space: nowrap;
  }

  .bar-container {
    flex-grow: 1;
    height: 20px;
    background-color: #e0e0e0;
    border-radius: 10px;
    overflow: hidden;
    position: relative;
  }

  .confidence-bar {
    height: 100%;
    transition:
      width 0.3s ease,
      background-color 0.3s ease;
  }

  .confidence-label {
    position: absolute;
    right: 5px;
    top: 50%;
    transform: translateY(-50%);
    font-size: 0.8em;
    font-weight: bold;
    text-shadow: 0 0 2px white;
  }
</style>
