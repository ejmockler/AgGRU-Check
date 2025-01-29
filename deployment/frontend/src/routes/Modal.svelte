<script lang="ts">
  export let title: string;

  import { createEventDispatcher } from "svelte";
  import CustomScroll from '$lib/components/CustomScroll.svelte';
  const dispatch = createEventDispatcher();

  function closeModal() {
    dispatch("close");
  }
</script>

<div class="modal-backdrop" on:click|self={() => dispatch('close')}>
  <div class="modal glass-panel">
    <header>
      <h2>{title}</h2>
      <button class="close-button" on:click={() => dispatch('close')}>
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M6 18L18 6M6 6l12 12" />
        </svg>
      </button>
    </header>
    <CustomScroll>
      <slot />
    </CustomScroll>
  </div>
</div>

<style lang="scss">
  @import '$lib/styles/glass.scss';
  @import '$lib/styles/colors.scss';

  .modal-backdrop {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: rgba(0, 0, 0, 0.3);
    backdrop-filter: blur(12px);
    display: grid;
    place-items: center;
    z-index: 1000;
    overflow: hidden;
  }

  .modal {
    @include glass-panel;
    background: rgba(255, 255, 255, 0.5);
    backdrop-filter: none;
    min-width: min(600px, 90vw);
    max-width: min(90vw, 900px);
    width: max-content;
    max-height: 90vh;
    display: flex;
    flex-direction: column;
    overflow: hidden;
    animation: slide-in 0.2s ease-out;
  }

  header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 1.5rem;
    border-bottom: 2px solid alpha(color(secondary), low);
    padding-bottom: 1rem;
    flex-shrink: 0;
    padding: 1rem 1.5rem;

    h2 {
      margin: 0;
      font-size: 1.5rem;
      font-weight: 700;
      color: color(secondary);
      text-shadow: 0 1px 2px rgba(255, 255, 255, 0.2);
    }
  }

  .close-button {
    width: 32px;
    height: 32px;
    border-radius: 6px;
    border: none;
    background: transparent;
    color: color(secondary);
    cursor: pointer;
    transition: all 0.2s;
    padding: 0.25rem;
    display: grid;
    place-items: center;

    svg {
      width: 20px;
      height: 20px;
    }

    &:hover {
      color: color(primary);
      transform: translateY(-1px);
    }

    &:active {
      transform: translateY(0);
    }
  }

  .content {
    overflow-y: auto;
    overflow-x: hidden;
    flex-grow: 1;
    padding-right: 1rem;
    color: alpha(color(secondary), high);
    line-height: 1.6;
    
    &::-webkit-scrollbar {
      width: 8px;
      height: 8px;
    }
    
    &::-webkit-scrollbar-track {
      background: alpha(color(secondary), subtle);
      border-radius: 4px;
    }
    
    &::-webkit-scrollbar-thumb {
      background: rgba(42, 157, 143, 0.6);
      border-radius: 4px;
      
      &:hover {
        background: rgba(42, 157, 143, 0.8);
      }
    }

    :global(pre) {
      white-space: pre-wrap;
      word-break: break-word;
    }
    
    :global(a) {
      color: color(primary);
      text-decoration: none;
      font-weight: 500;
      
      &:hover {
        text-decoration: underline;
      }
    }
    
    :global(ul), :global(ol) {
      padding-left: 1.5rem;
      margin: 1.5rem 0;
      max-width: 100%;
      
      :global(li) {
        margin-bottom: 1rem;
        padding-right: 1rem;
        
        :global(i), :global(strong) {
          color: color(primary);
          font-weight: 500;
        }
      }
    }
  }

  @keyframes slide-in {
    from {
      transform: translateY(20px);
      opacity: 0;
    }
    to {
      transform: translateY(0);
      opacity: 1;
    }
  }

  :global(.custom-scroll) {
    flex: 1;
    padding: 0 1.5rem 1.5rem;
    overflow-y: auto;
    overflow-x: auto;
    
    /* Firefox */
    scrollbar-width: thin;
    scrollbar-color: rgba(42, 157, 143, 0.6) rgba(26, 43, 59, 0.1);
    
    /* For other browsers, keep minimal styling */
    &::-webkit-scrollbar {
      width: 8px;
      height: 8px;
    }
    
    &::-webkit-scrollbar-thumb {
      background: rgba(42, 157, 143, 0.6);
      border-radius: 4px;
      
      &:hover {
        background: rgba(42, 157, 143, 0.8);
      }
    }
  }
</style>
