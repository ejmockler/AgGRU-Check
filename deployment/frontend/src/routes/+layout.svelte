<script>
  import { onMount, onDestroy } from "svelte";
  import Matter from "matter-js";
  import { source } from "sveltekit-sse";
  import { tweened } from "svelte/motion";
  import { expoOut } from "svelte/easing";

  let engine;
  let render;
  let world;
  let blobsGroup1 = [];
  let blobsGroup2 = [];
  let exclusionZoneRadius;
  let numDroplets;
  const blobs = [];
  const wallThickness = 20;
  const blobMargin = 100; // Margin from the walls to ensure no overlap

  let resizeTimeout;
  let isPaused = false;

  // Tweened store for engine time scale
  const timeScale = tweened(1, {
    duration: 200,
    easing: expoOut,
  });

  function initializeMatterJs() {
    // Calculate exclusion zone radius
    exclusionZoneRadius =
      Math.min(window.innerWidth, window.innerHeight) * 0.25;
    // Calculate initial number of droplets based on viewport size
    numDroplets = calculateNumDroplets(window.innerWidth, window.innerHeight);

    // Create Matter.js engine
    engine = Matter.Engine.create();
    world = engine.world;

    // Adjust engine parameters for better stability
    timeScale.set(0.2);
    engine.positionIterations = 20;
    engine.velocityIterations = 20;
    engine.constraintIterations = 20;

    // Disable gravity
    engine.world.gravity.y = 0;

    // Create a Matter.js render object
    const overscanFactor = 1.1;
    render = Matter.Render.create({
      element: document.querySelector("#canvas-container"),
      engine: engine,
      options: {
        width: window.innerWidth * overscanFactor,
        height: window.innerHeight * overscanFactor,
        background: "transparent",
        wireframes: false,
        pixelRatio: "auto", // Ensure the render scales with the device pixel ratio
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

          // Repulsive force from exclusion zone
          const dx = body.position.x - window.innerWidth / 2;
          const dy = body.position.y - window.innerHeight / 2;
          const distance = Math.sqrt(dx * dx + dy * dy);
          if (distance < exclusionZoneRadius) {
            const exclusionForceMagnitude =
              repulsionForce * (1 - distance / exclusionZoneRadius);
            Matter.Body.applyForce(body, body.position, {
              x: (dx / distance) * exclusionForceMagnitude,
              y: (dy / distance) * exclusionForceMagnitude,
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
                    (attractionStrength * body1.mass * body2.mass) / distanceSq;
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
    Matter.Engine.run(engine);
    Matter.Render.run(render);
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
    const checkbox = document.querySelector("#play-pause");
    if (checkbox.checked) {
      fadeCanvasOut(() => {
        destroyMatterJs();
        initializeMatterJs();
        fadeCanvasIn();
        isPaused = false;
      });
    } else {
      Matter.Runner.stop(engine);
      Matter.Render.stop(render);
      isPaused = true;
    }
  }

  onMount(() => {
    initializeMatterJs();

    // Adjust the canvas size and blobs on window resize with debounce
    const handleResize = debounce(() => {
      fadeCanvasOut(() => {
        destroyMatterJs();
        initializeMatterJs();
        if (isPaused) {
          Matter.Render.stop(render);
          Matter.Runner.stop(engine);
        }
      });
    }, 350);

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

    window.addEventListener("resize", () => {
      pauseEngine();
      clearTimeout(resizeTimeout);
      resizeTimeout = setTimeout(() => {
        handleResize();
        if (!isPaused) {
          resumeEngine();
        }
      }, 350);
    });

    onDestroy(() => {
      window.removeEventListener("resize", handleResize);
      destroyMatterJs();
    });
  });

  const calculateNumDroplets = (width, height) => {
    const area = width * height;
    return Math.floor(area / 25000); // Adjust the divisor for desired density
  };

  const generateDroplets = (num) => {
    for (let i = blobs.length; i < num; i++) {
      const size = Math.random() * 50 + 30;
      const color = `rgba(${Math.random() * 255}, ${Math.random() * 255}, ${Math.random() * 255}, 0.7)`;

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
      friction: 0.2,
      frictionAir: 0.05,
      restitution: 0.4,
      render: { fillStyle: color },
    };
    const particles = Matter.Composites.stack(x, y, 3, 3, 10, 10, (x, y) =>
      Matter.Bodies.circle(x, y, size / 6, particleOptions)
    );
    const constraints = Matter.Composites.mesh(particles, 3, 3, false, {
      stiffness: 0.05,
      render: { visible: false },
    });
    return Matter.Composite.create({
      bodies: particles.bodies,
      constraints: constraints.constraints,
    });
  };
</script>

<main style="height: 100vh">
  <div id="canvas-container" class="canvas-container"></div>
  <label class="play-pause">
    <input type="checkbox" id="play-pause" on:change={toggleEngine} checked />
    <span></span>
  </label>
  <slot />
</main>

<style lang="scss">
  :global(html) {
    font-family: "Gill Sans", "Gill Sans MT", Calibri, "Trebuchet MS",
      sans-serif;
  }
  :global(body) {
    background: linear-gradient(343deg, #07c090, #1971a9, #934646, #00ecf4);
    background-size: 800% 800%;
    animation: backgroundSwoop 74s ease infinite;
    overflow: scroll;
    margin: 0;
    padding: 0;
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
    position: relative;
    overflow: scroll;
  }

  .canvas-container {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    overflow: hidden; // Ensure overflow is hidden
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
  }

  .play-pause {
    position: absolute;
    top: 10px;
    right: 10px;
    width: 50px;
    height: 50px;
    cursor: pointer;
  }

  .play-pause input {
    display: none;
  }

  .play-pause span {
    display: block;
    width: 100%;
    height: 100%;
    background: no-repeat center/50%
      url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="white"><path d="M8 5v14l11-7z"/></svg>');
    transition: background 0.3s;
  }

  .play-pause input:checked + span {
    background: no-repeat center/50%
      url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="white"><path d="M6 19h4V5H6v14zm8-14v14h4V5h-4z"/></svg>');
  }
</style>
