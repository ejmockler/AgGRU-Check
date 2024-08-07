<script lang="ts">
  export let title: string;

  import { createEventDispatcher } from "svelte";
  const dispatch = createEventDispatcher();

  function closeModal() {
    dispatch("close");
  }

  function handleClickOutside(event: MouseEvent) {
    if (event.target === event.currentTarget) {
      closeModal();
    }
  }
</script>

<div class="modal-backdrop" on:click={handleClickOutside}>
  <div class="modal">
    <div class="modal-header">
      <h2>{title}</h2>
      <button class="close-button" on:click={closeModal}>&times;</button>
    </div>
    <div class="modal-content">
      <slot></slot>
    </div>
  </div>
</div>

<style lang="scss">
  .modal-backdrop {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background-color: rgba(0, 0, 0, 0.5);
    display: flex;
    justify-content: center;
    align-items: center;
    z-index: 1000;
  }

  .modal {
    background-color: white;
    border-radius: 7px;
    padding: 1em;
    max-width: 80vw;
    max-height: 80vh;
    overflow-y: auto;
    color: black;
    position: relative;
  }

  .modal-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 1em;
  }

  h2 {
    margin: 0;
  }

  .close-button {
    background: none;
    border: none;
    font-size: 1.5em;
    cursor: pointer;
    padding: 0;
    color: #333;
  }

  .modal-content {
    font-size: 0.9em;
  }
</style>
