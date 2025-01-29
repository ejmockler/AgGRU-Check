<script lang="ts">
  import type { PositionResult } from '$lib/types';
  import SequenceVisualizer from '$lib/components/SequenceVisualizer.svelte';
  import { fade } from 'svelte/transition';

  export let sequence: string; // Expecting full sequence (with header if present)
  export let results: PositionResult[] = [];
  export let count: number;
  export let error: string | null = null;
  export let isLoading = false;
  export let progress: { 
    position: number; 
    totalLength: number;
    models_completed: number;
    total_models: number;
  } | undefined = undefined;

  // Parse FASTA header if present
  $: {
    const lines = sequence.split(/\r?\n/);
    const hasFastaHeader = lines[0].startsWith('>');
    header = hasFastaHeader ? lines[0].slice(1).trim() : null;
    cleanSequence = hasFastaHeader ? lines.slice(1).join('\n') : sequence;
  }

  let header: string | null = null;
  let cleanSequence: string;
</script>

<div class="result-card">
  <div class="result-header">
    {#if header}
      <h3>{header}</h3>
    {:else}
      <h3>Sequence {count}</h3>
    {/if}
    {#if error}
      <div class="error-message">{error}</div>
    {/if}
  </div>

  <div class="result-content">
    {#if isLoading}
      <div class="progress-bar">
        <div class="progress-text" transition:fade|local={{ duration: 200 }}>
          {#if progress}
            Processing with model {progress.models_completed} of {progress.total_models}
          {:else}
            Initializing analysis...
          {/if}
        </div>
        <div class="progress-track">
          <div 
            class="progress-fill"
            class:progress-indeterminate={!progress}
            style={progress ? `width: ${(progress.models_completed / progress.total_models * 100)}%` : undefined}
          />
        </div>
      </div>
    {/if}

    <SequenceVisualizer
      sequence={cleanSequence}
      results={results}
      {isLoading}
    />

  </div>
</div>

<style lang="scss">
  .result-card {
    position: relative;
    z-index: 1;
    padding: 1.5rem;
    border-radius: 12px;
    background: rgba(255, 255, 255, 0.8);
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
    width: 100%;
    display: flex;
    flex-direction: column;
    overflow: visible;
  }

  .result-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 1rem;
    flex-wrap: wrap;
    gap: 1rem;

    h3 {
      font-size: 1.25rem;
      font-weight: 600;
      color: #1a2b3b;
      margin: 0;
      word-break: break-word;
      max-width: 100%;
    }
  }

  .result-content {
    position: relative;
    z-index: 1;
    flex: 1;
    min-width: 0;
    width: 100%;
    isolation: isolate;
    overflow: visible;
  }

  .error-message {
    color: #ef4444;
    font-weight: 500;
    font-size: 0.875rem;
  }

  .legend {
    margin-top: 1rem;
    padding: 1rem;
    background: rgba(255, 255, 255, 0.5);
    border-radius: 8px;
    display: flex;
    flex-direction: column;
    align-items: center;
    width: 100%;
    
    .gradient-legend {
      display: flex;
      flex-direction: column;
      gap: 0.5rem;
      width: 100%;
      
      .gradient-bar {
        height: 16px;
        border-radius: 4px;
        background: linear-gradient(
          to right,
          rgba(22, 163, 74, 1),    // Brighter green
          rgba(22, 163, 74, 1) 30%, // Hold green longer
          rgba(234, 179, 8, 1) 40%, // Bright yellow
          rgba(234, 179, 8, 1) 60%, // Hold yellow
          rgba(220, 38, 38, 1) 70%, // Bright red
          rgba(220, 38, 38, 1) 100% // Hold red
        );
        box-shadow: 
          0 2px 4px rgba(0, 0, 0, 0.1),
          0 0 0 1px rgba(0, 0, 0, 0.05);
      }
      
      .gradient-labels {
        display: flex;
        justify-content: space-between;
        font-size: 0.875rem;
        color: #1a2b3b;
        padding: 0 2px;
      }
    }
    
    .legend-label {
      margin-top: 0.5rem;
      font-size: 0.875rem;
      color: #1a2b3b;
      text-align: center;
    }
  }

  .progress-bar {
    margin-bottom: 1rem;
    padding: 0.5rem;
    background: rgba(255, 255, 255, 0.1);
    border-radius: 8px;

    .progress-text {
      font-size: 0.875rem;
      color: #1a2b3b;
      margin-bottom: 0.5rem;
      text-align: center;
    }

    .progress-track {
      height: 4px;
      background: rgba(0, 0, 0, 0.1);
      border-radius: 2px;
      overflow: hidden;

      .progress-fill {
        height: 100%;
        background: #2A9D8F;
        transition: width 0.3s ease;

        &.progress-indeterminate {
          width: 30%;
          animation: indeterminate 1.5s ease-in-out infinite;
        }
      }
    }
  }

  @keyframes indeterminate {
    0% {
      transform: translateX(-100%);
    }
    100% {
      transform: translateX(400%);
    }
  }
</style>
