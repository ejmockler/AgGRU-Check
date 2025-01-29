<script lang="ts">
  import { onMount, afterUpdate } from 'svelte';
  import type { PositionResult } from '$lib/types';
  import { fade } from 'svelte/transition';
  
  interface WindowResult {
    start: number;
    end: number;
    score: number;
    significance: number;
    consistency: number;
  }

  interface PartialResult extends PositionResult {
    isUpdated?: boolean;
    confidence?: number;
    activeWindows?: WindowResult[];
  }

  interface TooltipPosition {
    flip: boolean;
    shift: 'left' | 'right' | null;
  }

  export let sequence: string;
  export let results: PartialResult[] = [];
  export let isLoading = false;

  const WINDOW_SIZE = 25;
  const MIN_CELL_SIZE = 28;
  let container: HTMLDivElement;
  let cellSize = MIN_CELL_SIZE;

  // Reactive declaration for grid dimensions
  $: if (container && sequence) {
    updateLayout();
  }

  function updateLayout() {
    if (container) {
      const containerWidth = container.clientWidth - 32;
      const possibleCells = Math.floor(containerWidth / MIN_CELL_SIZE);
      cellSize = containerWidth / possibleCells;
    }
  }

  onMount(() => {
    const resizeObserver = new ResizeObserver(() => {
      updateLayout();
    });
    resizeObserver.observe(container);
    return () => resizeObserver.disconnect();
  });

  // Calculate background color based on saliency score and confidence
  function getScoreColor(score: number, confidence: number = 1): string {
    // Ensure we have valid numbers
    if (score === undefined || score === null) return 'transparent';
    
    // Define color stops
    const colors = {
      low: [22, 163, 74],    // Green
      mid: [234, 179, 8],    // Yellow
      high: [220, 38, 38]    // Red
    };
    
    // Calculate alpha based on confidence
    const alpha = Math.max(0.1, Math.min(confidence, 0.95));
    
    let rgb;
    if (score <= 0.3) {
      rgb = colors.low;
    } else if (score <= 0.4) {
      const t = (score - 0.3) / 0.1;
      rgb = colors.low.map((start, i) => 
        Math.round(start + (colors.mid[i] - start) * t)
      );
    } else if (score <= 0.7) {
      rgb = colors.mid;
    } else {
      const t = (score - 0.7) / 0.3;
      rgb = colors.mid.map((start, i) => 
        Math.round(start + (colors.high[i] - start) * t)
      );
    }
    
    // Always show at least a faint color, even for score = 0
    const minAlpha = 0.1;
    const adjustedAlpha = minAlpha + (alpha - minAlpha) * score;
    
    return `rgba(${rgb[0]}, ${rgb[1]}, ${rgb[2]}, ${adjustedAlpha})`;
  }

  // Calculate text color based on background color and confidence
  function getTextColor(score: number, confidence: number = 1): string {
    const brightness = score * confidence;
    return brightness > 0.4 ? '#1a2b3b' : '#ffffff';
  }

  // Track which cells were recently updated
  let recentUpdates = new Set<number>();
  
  $: {
    recentUpdates = new Set(
      results.filter(r => r.isUpdated).map(r => r.position)
    );
    
    if (recentUpdates.size > 0) {
      setTimeout(() => {
        recentUpdates = new Set();
      }, 1000);
    }
  }

  // Track active windows for hover effects
  let hoveredWindow: WindowResult | null = null;

  function isPositionInWindow(position: number, window: WindowResult): boolean {
    return position >= window.start && position < window.end;
  }

  function getWindowHighlight(position: number): string {
    if (!hoveredWindow) return 'transparent';
    return isPositionInWindow(position, hoveredWindow) 
      ? 'rgba(255, 255, 255, 0.2)' 
      : 'transparent';
  }

  function getTooltipPosition(position: number): TooltipPosition {
    if (!container) return { flip: false, shift: null };
    
    const cellsPerRow = Math.floor((container.clientWidth - 32) / cellSize);
    const col = position % cellsPerRow;
    
    // Calculate if tooltip would overflow horizontally
    const leftSide = col < cellsPerRow / 3;
    const rightSide = col > (cellsPerRow * 2/3);
    
    let shift: 'left' | 'right' | null = null;
    if (leftSide) {
      shift = 'right';
    } else if (rightSide) {
      shift = 'left';
    }
    
    return { 
      flip: false, // We'll handle this with pure CSS
      shift 
    };
  }
</script>

<div class="sequence-visualizer glass-panel" bind:this={container}>
  <div class="sequence-container">
    {#each sequence.split('') as char, i}
      {@const result = results[i]}
      {@const tooltipPos = getTooltipPosition(i)}
      <div class="cell-wrapper">
        <div
          class="amino-cell"
          class:updated={recentUpdates.has(i)}
          class:processing={isLoading && (!result || result.confidence < 0.5)}
          class:low-confidence={result?.confidence < 0.5}
          style="
            width: {cellSize}px;
            height: {cellSize}px;
            background-color: {result?.score !== undefined 
              ? getScoreColor(result.score, result.confidence) 
              : 'transparent'};
            color: {result?.score !== undefined 
              ? getTextColor(result.score, result.confidence) 
              : 'inherit'};
            box-shadow: inset 0 0 0 2px {getWindowHighlight(i)};
          "
          on:mouseenter={() => {
            if (result?.activeWindows?.length) {
              hoveredWindow = result.activeWindows[result.activeWindows.length - 1];
            }
          }}
          on:mouseleave={() => {
            hoveredWindow = null;
          }}
        >
          <span class="amino-text">{char}</span>
          {#if result}
            <div 
              class="score-tooltip" 
              class:flip={tooltipPos.flip}
              class:shift-left={tooltipPos.shift === 'left'}
              class:shift-right={tooltipPos.shift === 'right'}
            >
              <div class="tooltip-header">
                <div class="tooltip-title">Position {i + 1}</div>
                <div class="tooltip-metrics">
                  <div class="metric">
                    <span class="metric-label">Saliency</span>
                    <span class="metric-value">{(result.score * 100).toFixed(1)}%</span>
                  </div>
                  <div class="metric">
                    <span class="metric-label">Agreement</span>
                    <span class="metric-value">{((result.confidence || 0) * 100).toFixed(0)}%</span>
                  </div>
                </div>
              </div>

              {#if result.activeWindows?.length}
                <div class="tooltip-section">
                  <div class="metric">
                    <span class="metric-label">Active Windows</span>
                    <span class="metric-value">{result.activeWindows.length}</span>
                  </div>
                  {#if hoveredWindow}
                    <div class="window-details">
                      <div class="window-header">Window {hoveredWindow.start + 1}-{hoveredWindow.end}</div>
                      <div class="metric">
                        <span class="metric-label">Score</span>
                        <span class="metric-value">{(hoveredWindow.score * 100).toFixed(1)}%</span>
                      </div>
                      <div class="metric">
                        <span class="metric-label">Significance</span>
                        <span class="metric-value">{(hoveredWindow.significance * 100).toFixed(1)}%</span>
                      </div>
                      <div class="metric">
                        <span class="metric-label">Consistency</span>
                        <span class="metric-value">{(hoveredWindow.consistency * 100).toFixed(1)}%</span>
                      </div>
                    </div>
                  {/if}
                </div>
              {/if}

              <div class="tooltip-footer">
                {#if result.confidence < 0.5}
                  <div class="warning">Low confidence prediction</div>
                {/if}
                <div class="propensity">
                  {#if result.score > 0.9}
                    Strong amyloid propensity
                  {:else if result.score < 0.3}
                    Low amyloid propensity
                  {:else}
                    Moderate amyloid propensity
                  {/if}
                </div>
              </div>
            </div>
          {/if}
        </div>
        <div class="position-marker">
          {(i + 1) % 10 === 0 ? i + 1 : ''}
        </div>
      </div>
    {/each}
  </div>
</div>

<style lang="scss">
  .sequence-visualizer {
    position: relative;
    width: 100%;
    padding: 1rem;
    margin: 1rem 0;
    isolation: isolate;
    z-index: 1;
    overflow: visible; // Allow tooltips to overflow
  }

  .sequence-container {
    position: relative;
    display: flex;
    flex-wrap: wrap;
    gap: 1px;
    width: 100%;
    font-family: 'JetBrains Mono', monospace;
    font-size: 14px;
    line-height: 1;
    overflow: visible; // Allow tooltips to overflow
  }

  .cell-wrapper {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 4px;
  }

  .amino-cell {
    position: relative;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 4px;
    transition: all 0.3s ease;
    cursor: pointer;
    z-index: 1;

    &:hover {
      transform: scale(1.1);
      z-index: 100;

      .score-tooltip {
        opacity: 1;
        visibility: visible;
      }
    }

    .amino-text {
      font-weight: 600;
      transition: color 0.3s ease;
    }
  }

  .score-tooltip {
    position: absolute;
    left: 50%;
    bottom: calc(100% + 4px); // Reduced from 8px to be closer to the cell
    transform: translateX(-50%);
    background: rgba(0, 0, 0, 0.95);
    color: white;
    padding: 0.75rem;
    border-radius: 6px;
    font-size: 0.75rem;
    white-space: normal;
    opacity: 0;
    visibility: hidden;
    transition: all 0.2s ease;
    pointer-events: none;
    z-index: 1000;
    min-width: 180px;
    max-width: 240px;
    line-height: 1.4;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);

    // If tooltip would go off the top of the screen, show below instead
    @media (pointer: fine) { // Only apply to devices with hover capability
      .amino-cell:hover & {
        opacity: 1;
        visibility: visible;
      }
    }

    &::before {
      content: '';
      position: absolute;
      width: 8px;
      height: 8px;
      background: inherit;
      left: 50%;
      bottom: -4px;
      transform: translateX(-50%) rotate(45deg);
    }

    &.shift-left {
      transform: translateX(-90%);
      
      &::before {
        left: 90%;
      }
    }

    &.shift-right {
      transform: translateX(-10%);
      
      &::before {
        left: 10%;
      }
    }

    // When tooltip hits top of screen, flip to bottom
    @media (max-height: 300px) {
      bottom: auto;
      top: calc(100% + 4px); // Reduced from 8px to be closer to the cell

      &::before {
        bottom: auto;
        top: -4px;
      }
    }

    .tooltip-header {
      margin-bottom: 0.5rem;

      .tooltip-title {
        font-weight: 600;
        margin-bottom: 0.25rem;
      }

      .tooltip-metrics {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 0.5rem;
      }
    }

    .metric {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 0.5rem;

      .metric-label {
        color: rgba(255, 255, 255, 0.7);
        font-size: 0.7rem;
      }

      .metric-value {
        font-weight: 600;
      }
    }

    .tooltip-section {
      margin-top: 0.5rem;
      padding-top: 0.5rem;
      border-top: 1px solid rgba(255, 255, 255, 0.2);

      .window-details {
        margin-top: 0.25rem;
        padding: 0.25rem;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 4px;

        .window-header {
          font-size: 0.7rem;
          color: rgba(255, 255, 255, 0.7);
          margin-bottom: 0.25rem;
        }
      }
    }

    .tooltip-footer {
      margin-top: 0.5rem;
      padding-top: 0.5rem;
      border-top: 1px solid rgba(255, 255, 255, 0.2);
      font-size: 0.7rem;

      .warning {
        color: #fbbf24;
        margin-bottom: 0.25rem;
      }

      .propensity {
        font-weight: 600;
      }
    }
  }

  .position-marker {
    font-size: 10px;
    color: rgba(0, 0, 0, 0.5);
    text-align: center;
    user-select: none;
    pointer-events: none;
    height: 14px;
  }

  .amino-cell.updated {
    animation: pulse 1s ease-out;
  }
  
  .amino-cell.processing {
    animation: pulse 2s ease-in-out infinite;
    opacity: 0.7;
  }
  
  @keyframes pulse {
    0% { transform: scale(1); }
    50% { transform: scale(1.1); }
    100% { transform: scale(1); }
  }
</style> 