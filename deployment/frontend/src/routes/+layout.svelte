<main>
  <slot />
</main>

<style lang="scss">
  :global(html),
  :global(body) {
    background: linear-gradient(343deg, #07c090, #1971a9, #934646, #00ecf4);
    background-size: 800% 800%;

    animation: backgroundSwoop 14s ease infinite;
  }

  @keyframes backgroundSwoop {
    0% {
      background-position: 0% 17%;
    }
    50% {
      background-position: 100% 84%;
    }
    100% {
      background-position: 0% 17%;
    }
  }

  main {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    min-height: 100%;
  }
</style>
