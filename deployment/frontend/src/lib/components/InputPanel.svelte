<script lang="ts">
  import type { Sequence } from '$lib/types';
  
  // Props
  export let onSubmit: (sequences: string) => void;
  export let isLoading: boolean = false;
  
  // Example sequences
  const exampleSequences = [
    {
      name: "TDP-43",
      fasta: ">TDP-43 (TAR DNA-binding protein 43)\nMSEYIRVTEDENDEPIEIPSEDDGTVLLSTVTAQFPGACGLRYRNPVSQCMRGVRLVEGILHAPDAGWGNLVYVVNYPKDNKRKMDETDASSAVKVKRAVQKTSDLIVLGLPWKTTEQDLKEYFSTFGEVLMVQVKKDLKTGHSKGFGFVRFTEYETQVKVMSQRHMIDGRWCDCKLPNSKQSQDEPLRSRKVFVGRCTEDMTEDELREFFSQYGDVMDVFIPKPFRAFAFVTFADDQIAQSLCGEDLIIKGISVHISNAEPKHNSNRQLERSGRFGGNPGGFGNQGGFGNSRGGGAGLGNNQGSNMGGGMNFGAFSINPAMMAAAQAALQSSWGMMGMLASQQNQSGPSGNNQNQGNMQREPNQAFGSGNNSYSGSNSGAAIGWGSASNAGSGSGFNGGFGSSMDSKSSGWGM"
    },
    {
      name: "MOTS-c",
      fasta: ">MOTS-c (short mitochondrial protein)\nMRWQEMGYIFYPRKLR"
    }
  ];

  // State
  let sequenceInput: HTMLTextAreaElement;
  let isNewInput = true;

  // Methods
  function appendFastaToTextarea(fastaContent: string) {
    if (!sequenceInput) return;
    
    const currentValue = sequenceInput.value;
    if (!currentValue.includes(fastaContent)) {
      sequenceInput.value = currentValue 
        ? `${currentValue}\n${fastaContent}` 
        : fastaContent;
      isNewInput = true;
    }
  }

  function handleSubmit(event: Event) {
    event.preventDefault();
    if (sequenceInput && !isLoading) {
      // Strip empty lines and trim whitespace
      const cleanedInput = sequenceInput.value
        .split(/\r?\n/)
        .filter(line => line.trim())
        .join('\n');
      onSubmit(cleanedInput);
    }
  }
</script>

<div class="input-panel glass-panel">
  <header class="branding">
    <h1>AgGRU? Check:</h1>
    <nav class="nav-links">
      <slot name="nav-links" />
    </nav>
  </header>

  <form on:submit={handleSubmit}>
    <div class="example-sequences">
      {#each exampleSequences as sequence}
        <button 
          type="button"
          class="sequence-pill"
          on:click={() => appendFastaToTextarea(sequence.fasta)}
        >
          <span class="pill-text">{sequence.name}</span>
        </button>
      {/each}
    </div>

    <textarea
      bind:this={sequenceInput}
      name="sequences"
      placeholder="Is your protein amyloidgenic?

Enter up to 5 amino acid sequences
(either raw or FASTA/FASTQ)"
      minlength="3"
      required
      on:input={() => { isNewInput = true; }}
      disabled={isLoading}
    />

    <button 
      type="submit" 
      class="submit-button"
      disabled={isLoading}
    >
      <span>{isLoading ? 'Processing...' : 'Infer'}</span>
      <svg viewBox="0 0 20 20" fill="currentColor" class="arrow-right">
        <path fill-rule="evenodd" d="M3 10a.75.75 0 01.75-.75h10.638L10.23 5.29a.75.75 0 111.04-1.08l5.5 5.25a.75.75 0 010 1.08l-5.5 5.25a.75.75 0 11-1.04-1.08l4.158-3.96H3.75A.75.75 0 013 10z" clip-rule="evenodd" />
      </svg>
    </button>
  </form>
</div>

<style lang="scss">
  @import '$lib/styles/glass.scss';
  @import '$lib/styles/colors.scss';
  
  .input-panel {
    @include glass-panel;
    display: flex;
    flex-direction: column;
    gap: 2rem;
    height: 100%;
    overflow-y: auto;

    // Add some padding to prevent tooltip clipping
    padding: 1.5rem;
    padding-right: 2rem; // Extra padding for potential scrollbar
  }

  :global(.results-panel) {
    @include glass-panel;
    overflow: auto;
  }

  :global(.result-card) {
    @include glass-card;
  }

  .branding {
    h1 {
      font-size: 2.8rem;
      font-weight: 800;
      color: color(secondary);
      margin: 0 0 1rem;
      text-shadow: 0 1px 2px rgba(0, 0, 0, 0.1);
    }
  }

  .nav-links {
    padding: 0.5rem 0 1.5rem;
    border-bottom: 2px solid alpha(color(secondary), low);
    display: flex;
    gap: 0.75rem;

    :global(.nav-link) {
      font-size: 1rem;
      font-weight: 600;
      color: color(secondary);
      padding: 0.5rem 1rem;
      border-radius: 6px;
      transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
      text-decoration: none;
      background: none;
      border: none;
      cursor: pointer;
      position: relative;
      
      &::after {
        content: '';
        position: absolute;
        bottom: 0;
        left: 50%;
        width: 0;
        height: 2px;
        background: color(primary);
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
        transform: translateX(-50%);
      }
      
      &:hover {
        color: color(primary);
        
        &::after {
          width: 80%;
        }
      }

      &:active {
        transform: translateY(1px);
      }
    }
  }

  form {
    display: flex;
    flex-direction: column;
    gap: 2rem;
    flex: 1;
    min-height: 0; // Allow flex container to shrink

    .example-sequences {
      display: flex;
      gap: 1rem;
      flex-wrap: wrap;
    }

    .sequence-pill {
      background: rgba(42, 157, 143, 0.25);
      color: #1b7268;
      font-weight: 600;
      border: 1px solid rgba(42, 157, 143, 0.35);
      letter-spacing: 0.02em;
      padding: 0.75rem 1.25rem;
      border-radius: 20px;
      font-size: 0.95rem;
      cursor: pointer;
      transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
      position: relative;
      overflow: hidden;
      backdrop-filter: blur(4px);
      box-shadow: 
        0 2px 4px rgba(42, 157, 143, 0.15),
        0 1px 2px rgba(42, 157, 143, 0.2);
      
      &::before {
        content: '';
        position: absolute;
        top: 50%;
        left: 50%;
        width: 120%;
        height: 120%;
        background: linear-gradient(
          120deg,
          rgba(42, 157, 143, 0.98),
          rgba(36, 134, 122, 0.98)
        );
        transform: translate(-50%, -50%) scale(0);
        border-radius: inherit;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        z-index: 0;
        opacity: 0;
      }
      
      .pill-text {
        position: relative;
        z-index: 1;
        transition: all 0.3s ease;
        display: block;
      }
      
      &:hover {
        background: rgba(42, 157, 143, 0.15);
        border-color: rgba(42, 157, 143, 0.5);
        transform: translateY(-2px);
        box-shadow: 
          0 4px 12px rgba(42, 157, 143, 0.25),
          0 2px 4px rgba(42, 157, 143, 0.2);
        
        .pill-text {
          color: white;
        }
        
        &::before {
          transform: translate(-50%, -50%) scale(1);
          opacity: 1;
        }
      }

      &:active {
        transform: translateY(0);
        box-shadow: 
          0 2px 6px rgba(42, 157, 143, 0.2),
          0 1px 2px rgba(42, 157, 143, 0.15);
        
        &::before {
          opacity: 1;
          background: linear-gradient(
            120deg,
            rgba(42, 157, 143, 1),
            rgba(36, 134, 122, 1)
          );
        }
      }

      &:disabled {
        opacity: 0.5;
        cursor: not-allowed;
        transform: none;
      }
    }

    textarea {
      flex: 1;
      min-height: 200px;
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.875rem;
      line-height: 1.7;
      letter-spacing: 0.02em;
      padding: 1.5rem;
      border-radius: 12px;
      background: rgba(255, 255, 255, 0.5);
      border: 1px solid rgba(42, 157, 143, 0.3);
      color: color(secondary);
      resize: none;
      transition: all 0.2s;
      
      &::placeholder {
        color: alpha(color(secondary), medium);
        font-style: italic;
      }
      
      &:focus {
        outline: none;
        background: rgba(255, 255, 255, 0.6);
        border-color: color(primary);
        box-shadow: 
          0 0 0 3px alpha(color(primary), subtle),
          inset 0 1px 2px rgba(0, 0, 0, 0.05);
        transform: translateY(-1px);
      }

      &:disabled {
        opacity: 0.7;
        cursor: not-allowed;
      }
    }

    .submit-button {
      @include glass-button;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 0.5rem;
      color: white;
      padding: 1rem;
      border-radius: 12px;
      font-weight: 600;
      font-size: 0.875rem;
      letter-spacing: 0.03em;
      text-transform: uppercase;
      transition: all 0.2s;
      cursor: pointer;

      svg {
        width: 1.25rem;
        height: 1.25rem;
        transition: transform 0.2s;
      }

      &:hover:not(:disabled) {
        transform: translateY(-1px);
        box-shadow: 
          0 8px 24px rgba(15, 118, 110, 0.25),
          0 4px 8px rgba(15, 118, 110, 0.15);
        
        svg {
          transform: translateX(4px);
        }
      }

      &:active:not(:disabled) {
        transform: translateY(0);
        box-shadow: 
          0 4px 12px rgba(15, 118, 110, 0.2),
          0 2px 4px rgba(15, 118, 110, 0.1);
      }

      &:disabled {
        opacity: 0.5;
        cursor: not-allowed;
        transform: none;
      }
    }
  }

  // Loading state animations
  @keyframes pulse {
    0% { opacity: 1; }
    50% { opacity: 0.7; }
    100% { opacity: 1; }
  }

  .loading {
    animation: pulse 1.5s ease-in-out infinite;
  }
</style> 