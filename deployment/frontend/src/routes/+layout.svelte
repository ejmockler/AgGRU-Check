<script lang="ts">
  import { onMount, onDestroy } from "svelte";
  import Matter from "matter-js";
  import { tweened } from "svelte/motion";
  import { expoOut } from "svelte/easing";
  import { browser } from "$app/environment";
  import { fade } from 'svelte/transition';
  import { animationsPaused } from '$lib/stores/animation';

  let engine;
  let render;
  let world;
  let blobsGroup1 = [];
  let blobsGroup2 = [];
  let exclusionZoneRadius;
  let numDroplets;
  const blobs = [];
  const wallThickness = 10;
  const blobMargin = 100; // Margin from the walls to ensure no overlap

  let resizeTimeout;
  let tooltipVisible = false;

  // Tweened store for engine time scale
  const timeScale = tweened(1, {
    duration: 200,
    easing: expoOut,
  });

  // Update body class when animation state changes
  $: if (browser) {
    document.body.classList.toggle('paused', $animationsPaused);
  }

  function initializeMatterJs() {
    // Calculate initial number of droplets based on viewport size
    numDroplets = calculateNumDroplets(window.innerWidth, window.innerHeight);

    // Create Matter.js engine
    engine = Matter.Engine.create();
    world = engine.world;

    // Adjust engine parameters for better stability
    timeScale.set(0.15);  // Slower overall movement
    engine.positionIterations = 12;  // Fewer iterations = less jitter
    engine.velocityIterations = 8;   // Reduced velocity solving
    engine.constraintIterations = 4;  // Fewer constraint solves = softer bodies

    // Disable gravity
    engine.world.gravity.y = 0;

    // Create a Matter.js render object
    render = Matter.Render.create({
      element: document.querySelector("#canvas-container"),
      engine: engine,
      options: {
        width: window.innerWidth,
        height: window.innerHeight,
        background: "transparent",
        wireframes: false,
        pixelRatio: "auto",
        showBounds: false,
      },
    });

    // Create walls around the viewport
    createWalls();

    // Generate initial droplets
    generateDroplets(numDroplets);

    // Add random movement, wall repulsion, group attraction, and exclusion zone repulsion to simulate floating
    Matter.Events.on(engine, "beforeUpdate", () => {
      // Update the engine time scale from the tweened store
      timeScale.subscribe((value) => {
        engine.timing.timeScale = value;
      });

      [...blobsGroup1, ...blobsGroup2].forEach((droplet) => {
        droplet.bodies.forEach((body) => {
          const forceMagnitude = 0.002 * body.mass;
          Matter.Body.applyForce(body, body.position, {
            x: (Math.random() - 0.5) * forceMagnitude,
            y: (Math.random() - 0.5) * forceMagnitude,
          });

          // Repulsive force from walls
          const repulsionForce = 0.02 * body.mass;
          if (body.position.x < wallThickness) {
            Matter.Body.applyForce(body, body.position, {
              x: repulsionForce,
              y: 0,
            });
          }
          if (body.position.x > window.innerWidth - wallThickness) {
            Matter.Body.applyForce(body, body.position, {
              x: -repulsionForce,
              y: 0,
            });
          }
          if (body.position.y < wallThickness) {
            Matter.Body.applyForce(body, body.position, {
              x: 0,
              y: repulsionForce,
            });
          }
          if (body.position.y > window.innerHeight - wallThickness) {
            Matter.Body.applyForce(body, body.position, {
              x: 0,
              y: -repulsionForce,
            });
          }
        });
      });

      // Attractive forces within each group
      applyAttractiveForces(blobsGroup1);
      applyAttractiveForces(blobsGroup2);
    });

    const applyAttractiveForces = (group) => {
      const attractionStrength = 0.000001;
      group.forEach((droplet, i) => {
        droplet.bodies.forEach((body1) => {
          group.forEach((otherDroplet, j) => {
            if (i !== j) {
              otherDroplet.bodies.forEach((body2) => {
                const dx = body2.position.x - body1.position.x;
                const dy = body2.position.y - body1.position.y;
                const distanceSq = dx * dx + dy * dy;
                if (distanceSq > 0) {
                  const forceMagnitude =
                    (attractionStrength * body1.mass * body2.mass) / Math.sqrt(distanceSq);  // Linear falloff instead of square
                  Matter.Body.applyForce(body1, body1.position, {
                    x: (dx / Math.sqrt(distanceSq)) * forceMagnitude,
                    y: (dy / Math.sqrt(distanceSq)) * forceMagnitude,
                  });
                }
              });
            }
          });
        });
      });
    };

    // Fade in the canvas
    fadeCanvasIn();

    // Run the Matter.js engine and renderer
    let runner = Matter.Runner.create();
    
    function runAnimation() {
      if (!$animationsPaused) {
        Matter.Engine.update(engine, 1000 / 60);
        Matter.Render.world(render);
      }
      requestAnimationFrame(runAnimation);
    }
    
    runAnimation();
  }

  function destroyMatterJs() {
    if (engine && render) {
      Matter.Render.stop(render);
      Matter.Engine.clear(engine);
      Matter.World.clear(world, false);
      render.canvas.remove();
      render.canvas = null;
      render.context = null;
      render.textures = {};
    }
    blobsGroup1 = [];
    blobsGroup2 = [];
    blobs.length = 0;
  }

  function fadeCanvasIn() {
    const canvas = document.querySelector("canvas");
    if (canvas) {
      canvas.style.opacity = 0;
      canvas.style.transition = "opacity 0.5s";
      requestAnimationFrame(() => {
        canvas.style.opacity = 1;
      });
    }
  }

  function fadeCanvasOut(callback) {
    const canvas = document.querySelector("canvas");
    if (canvas) {
      canvas.style.opacity = 1;
      canvas.style.transition = "opacity 0.5s";
      canvas.style.opacity = 0;
      setTimeout(callback, 500);
    }
  }

  function debounce(func, wait) {
    let timeout;
    return function (...args) {
      const later = () => {
        clearTimeout(timeout);
        func(...args);
      };
      clearTimeout(timeout);
      timeout = setTimeout(later, wait);
    };
  }

  function toggleEngine() {
    $animationsPaused = !$animationsPaused;
    const checkbox = document.querySelector("#play-pause");
    if (!$animationsPaused) {
      fadeCanvasOut(() => {
        destroyMatterJs();
        initializeMatterJs();
        fadeCanvasIn();
      });
    } else {
      Matter.Runner.stop(engine);
      Matter.Render.stop(render);
    }
  }

  let handleResize;

  onMount(() => {
    initializeMatterJs();

    // Adjust the canvas size and blobs on window resize with debounce
    handleResize = debounce(() => {
      if (render && render.canvas) {
        render.canvas.width = window.innerWidth;
        render.canvas.height = window.innerHeight;
        Matter.Render.setPixelRatio(render, window.devicePixelRatio); // Ensure proper scaling
      }
      fadeCanvasOut(() => {
        destroyMatterJs();
        initializeMatterJs();
        if (!$animationsPaused) {
          Matter.Render.stop(render);
          Matter.Runner.stop(engine);
        }
      });
    }, 350);

    window.addEventListener("resize", () => {
      pauseEngine();
      // Immediately adjust canvas size
      if (render && render.canvas) {
        render.canvas.width = window.innerWidth;
        render.canvas.height = window.innerHeight;
        Matter.Render.setPixelRatio(render, window.devicePixelRatio);
      }
      clearTimeout(resizeTimeout);
      resizeTimeout = setTimeout(() => {
        handleResize();
        if (!!$animationsPaused) {
          resumeEngine();
        }
      }, 350);
    });

    const pauseEngine = () => {
      if (render) {
        Matter.Render.stop(render);
        Matter.Runner.stop(engine);
      }
    };

    const resumeEngine = () => {
      if (render) {
        Matter.Render.run(render);
        Matter.Runner.run(engine);
      }
    };
  });

  onDestroy(() => {
    destroyMatterJs();
    if (browser) window.removeEventListener("resize", handleResize);
  });

  const calculateNumDroplets = (width, height) => {
    const area = width * height;
    return Math.floor(area / 25000); // Adjust the divisor for desired density
  };

  const generateDroplets = (num) => {
    const colors = [
      'rgba(15, 118, 110, 0.5)',   // Primary teal
      'rgba(30, 41, 59, 0.5)',     // Slate blue
      'rgba(255, 164, 27, 0.5)',   // Warm orange
      'rgba(255, 81, 81, 0.5)'     // Vibrant red
    ];

    for (let i = blobs.length; i < num; i++) {
      const size = Math.random() * 50 + 30;
      const color = colors[Math.floor(Math.random() * colors.length)];

      // Calculate random position outside the exclusion zone and walls
      let x, y, distance;
      do {
        x = Math.random() * (window.innerWidth - 2 * blobMargin) + blobMargin;
        y = Math.random() * (window.innerHeight - 2 * blobMargin) + blobMargin;
        distance = Math.sqrt(
          Math.pow(x - window.innerWidth / 2, 2) +
            Math.pow(y - window.innerHeight / 2, 2)
        );
      } while (distance < exclusionZoneRadius);

      const droplet = createSoftBody(x, y, size, color);
      Matter.World.add(world, droplet);
      blobs.push(droplet);
      if (i < num / 2) {
        blobsGroup1.push(droplet);
      } else {
        blobsGroup2.push(droplet);
      }
    }
  };

  const adjustDropletCount = (newNumDroplets) => {
    if (newNumDroplets > blobs.length) {
      // Add more droplets
      generateDroplets(newNumDroplets);
    } else if (newNumDroplets < blobs.length) {
      // Remove excess droplets
      const excess = blobs.length - newNumDroplets;
      for (let i = 0; i < excess; i++) {
        const droplet = blobs.pop();
        Matter.World.remove(world, droplet);
        blobsGroup1 = blobsGroup1.filter((d) => d !== droplet);
        blobsGroup2 = blobsGroup2.filter((d) => d !== droplet);
      }
    }
  };

  const createWalls = () => {
    // Remove existing walls if they exist
    if (world.bodies.length > 0) {
      const existingWalls = world.bodies.filter((body) => body.isStatic);
      Matter.World.remove(world, existingWalls);
    }

    // Create new walls
    const walls = [
      Matter.Bodies.rectangle(
        window.innerWidth / 2,
        -wallThickness / 2,
        window.innerWidth,
        wallThickness,
        { isStatic: true }
      ), // top
      Matter.Bodies.rectangle(
        window.innerWidth / 2,
        window.innerHeight + wallThickness / 2,
        window.innerWidth,
        wallThickness,
        { isStatic: true }
      ), // bottom
      Matter.Bodies.rectangle(
        -wallThickness / 2,
        window.innerHeight / 2,
        wallThickness,
        window.innerHeight,
        { isStatic: true }
      ), // left
      Matter.Bodies.rectangle(
        window.innerWidth + wallThickness / 2,
        window.innerHeight / 2,
        wallThickness,
        window.innerHeight,
        { isStatic: true }
      ), // right
    ];

    Matter.World.add(world, walls);
  };

  const createSoftBody = (x, y, size, color) => {
    const particleOptions = {
      friction: 0.2,      // Less friction between particles
      frictionAir: 0.05,  // Less air resistance
      restitution: 0.1,   // Less bouncy
      density: 0.001,     // Much lighter particles
      render: { fillStyle: color },
    };

    // Increase spacing between particles in the blob
    const particles = Matter.Composites.stack(
      x, y, 
      3, 3, 
      15, 15,  // More space between particles
      (x, y) => Matter.Bodies.circle(x, y, size / 6, particleOptions)
    );

    const constraints = Matter.Composites.mesh(particles, 3, 3, false, {
      stiffness: 0.02,  // Softer connections = less jitter
      damping: 0.1,     // Add damping to reduce oscillations
      render: { visible: false },
    });

    return Matter.Composite.create({
      bodies: particles.bodies,
      constraints: constraints.constraints,
    });
  };
</script>

<main>
  <div id="canvas-container" class="canvas-container"></div>
  <button 
    class="animation-control"
    on:click={toggleEngine}
    on:mouseenter={() => tooltipVisible = true}
    on:mouseleave={() => tooltipVisible = false}
    aria-label={$animationsPaused ? "Resume background animations" : "Pause background animations"}
  >
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
      {#if $animationsPaused}
        <path d="M8 5v14l11-7z" />
      {:else}
        <path d="M6 4h4v16H6zM14 4h4v16h-4z" />
      {/if}
    </svg>
  </button>
  {#if tooltipVisible}
    <div 
      class="tooltip"
      transition:fade={{ duration: 150 }}
      role="tooltip"
    >
      {$animationsPaused ? 'Resume animations' : 'Pause animations'}
    </div>
  {/if}
  <slot />
</main>

<style lang="scss">
  @import '$lib/styles/colors.scss';
  @import '$lib/styles/glass.scss';

  :global(html) {
    font-family: "Gill Sans", "Gill Sans MT", Calibri, "Trebuchet MS",
      sans-serif;
  }
  :global(body) {
    background: linear-gradient(343deg, #07c090, #1971a9, #934646, #00ecf4);
    background-size: 800% 800%;
    animation: backgroundSwoop 74s ease infinite;
    margin: 0;
    padding: 0;
    
    &.paused {
      animation-play-state: paused;
    }
  }

  :global(*) {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
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
    overflow: scroll;
    height: calc(100vh);
    width: calc(100vw);
    margin: auto;
    padding: 0;
    margin: auto;
  }

  .canvas-container {
    position: absolute;
    display: block;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    overflow: hidden; // Ensure overflow is hidden
    border: none;
    outline: none;
  }

  canvas {
    display: block;
    opacity: 0;
    transition: opacity 0.5s;
    position: absolute; // Ensure the canvas is positioned absolutely
    top: 50%;
    left: 50%;
    width: 100vw; // Ensure the canvas is oversized
    height: 100vh; // Ensure the canvas is oversized
    transform: translate(-50%, -50%); // Center the canvas
    border: none;
    outline: none;
  }

  .animation-control {
    position: fixed;
    top: 1.5rem;
    right: 1.5rem;
    width: 2.5rem;
    height: 2.5rem;
    border-radius: 8px;
    background: rgba(255, 255, 255, 0.2);
    border: 1px solid rgba(255, 255, 255, 0.3);
    color: color(secondary);
    cursor: pointer;
    transition: all 0.2s;
    z-index: 100;
    display: grid;
    place-items: center;
    padding: 0.5rem;
    
    svg {
      width: 1.25rem;
      height: 1.25rem;
    }
    
    &:hover {
      background: rgba(255, 255, 255, 0.3);
      transform: translateY(-1px);
      color: color(primary);
    }
    
    &:active {
      transform: translateY(0);
    }
  }

  .tooltip {
    position: fixed;
    top: 4.5rem;
    right: 1.5rem;
    background: rgba(26, 43, 59, 0.95);
    color: white;
    padding: 0.5rem 1rem;
    border-radius: 6px;
    font-size: 0.875rem;
    pointer-events: none;
    z-index: 100;
    
    &::after {
      content: '';
      position: absolute;
      top: -4px;
      right: 1.25rem;
      width: 8px;
      height: 8px;
      background: inherit;
      transform: rotate(45deg);
    }
  }
</style>
