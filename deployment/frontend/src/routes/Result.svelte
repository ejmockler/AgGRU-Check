<script>
  export let sequence;
  export let models = []; // Ensure models is at least an empty array
  export let count;
  export let isLoading = false; // This is now a per-result loading state
  export let error = null;

  const modelNames = ["model_0", "model_1", "model_2"];

  const getColor = (confidence) => {
    const hue = confidence * 120;
    return `hsl(${hue}, 100%, 50%)`;
  };

  // Populate a default loading state for models
  $: modelPlaceholders = modelNames.map((modelName, index) => {
    const modelData = models.find((m) => m.model === modelName);
    return modelData
      ? { ...modelData }
      : { model: modelName, confidence: null, loading: true };
  });
</script>

<div
  class="result-container {isLoading ? 'loading' : ''}"
  style={error ? "border-color: red; width: fit-content" : ""}
>
  <span class="count">{count}</span>
  {#if error}
    <div class="error-message">{error}</div>
  {:else}
    <div class="sequence">{sequence}</div>
    {#each modelPlaceholders as { model, confidence, loading }, i}
      <div class="model-container">
        <div class="model-label">Model {Number(model.split("_")[1]) + 1}:</div>
        <div class="bar-container">
          {#if loading}
            <div class="loading-bar"></div>
          {:else}
            <div
              class="confidence-bar"
              style="width: {confidence * 100}%; background-color: {getColor(
                confidence
              )};"
            ></div>
            <div class="confidence-label">
              {(confidence * 100).toFixed(1)}%
            </div>
          {/if}
        </div>
      </div>
    {/each}
  {/if}
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

  .result-container.loading {
    opacity: 0.7;
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

  .loading-bar {
    height: 100%;
    width: 100%;
    background: linear-gradient(90deg, #e0e0e0 25%, #f0f0f0 50%, #e0e0e0 75%);
    background-size: 200% 100%;
    animation: loading 1.5s infinite;
  }

  .error-message {
    color: red;
    font-weight: bold;
  }

  @keyframes loading {
    0% {
      background-position: 200% 0;
    }
    100% {
      background-position: -200% 0;
    }
  }
</style>
