<script>
  import { onMount } from "svelte";
  import Matter from "matter-js";

  let engine;
  let render;
  let world;

  onMount(() => {
    // Create Matter.js engine
    engine = Matter.Engine.create();
    world = engine.world;

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
      },
    });

    // Create walls around the viewport
    const wallThickness = 20;
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

    // Generate multiple droplets as soft bodies
    const numDroplets = 10;
    const createSoftBody = (x, y, size, color) => {
      const particleOptions = {
        friction: 0.05,
        frictionAir: 0.075,
        restitution: 0.1,
        render: { fillStyle: color },
      };
      const particles = Matter.Composites.stack(x, y, 3, 3, 10, 10, (x, y) =>
        Matter.Bodies.circle(x, y, size / 6, particleOptions)
      );
      const constraints = Matter.Composites.mesh(particles, 3, 3, false, {
        stiffness: 0.1,
        render: { visible: false },
      });
      return Matter.Composite.create({
        bodies: particles.bodies,
        constraints: constraints.constraints,
      });
    };

    const droplets = Array.from({ length: numDroplets }, () => {
      const size = Math.random() * 50 + 30;
      const color = `rgba(${Math.random() * 255}, ${Math.random() * 255}, ${Math.random() * 255}, 0.7)`;
      const droplet = createSoftBody(
        Math.random() * window.innerWidth,
        Math.random() * window.innerHeight,
        size,
        color
      );
      Matter.World.add(world, droplet);
      return droplet;
    });

    // Add random movement and wall repulsion to simulate floating
    Matter.Events.on(engine, "beforeUpdate", () => {
      droplets.forEach((droplet) => {
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
    });

    // Run the Matter.js engine and renderer
    Matter.Engine.run(engine);
    Matter.Render.run(render);

    // Adjust the canvas size on window resize
    window.addEventListener("resize", () => {
      Matter.Render.lookAt(render, {
        min: { x: 0, y: 0 },
        max: { x: window.innerWidth, y: window.innerHeight },
      });

      // Update wall positions and sizes on resize
      Matter.Body.setPosition(walls[0], {
        x: window.innerWidth / 2,
        y: -wallThickness / 2,
      });
      Matter.Body.setPosition(walls[1], {
        x: window.innerWidth / 2,
        y: window.innerHeight + wallThickness / 2,
      });
      Matter.Body.setPosition(walls[2], {
        x: -wallThickness / 2,
        y: window.innerHeight / 2,
      });
      Matter.Body.setPosition(walls[3], {
        x: window.innerWidth + wallThickness / 2,
        y: window.innerHeight / 2,
      });

      Matter.Body.setVertices(
        walls[0],
        Matter.Vertices.create(
          [
            { x: 0, y: -wallThickness / 2 },
            { x: window.innerWidth, y: -wallThickness / 2 },
            { x: window.innerWidth, y: wallThickness / 2 },
            { x: 0, y: wallThickness / 2 },
          ],
          walls[0]
        )
      );

      Matter.Body.setVertices(
        walls[1],
        Matter.Vertices.create(
          [
            { x: 0, y: window.innerHeight - wallThickness / 2 },
            { x: window.innerWidth, y: window.innerHeight - wallThickness / 2 },
            { x: window.innerWidth, y: window.innerHeight + wallThickness / 2 },
            { x: 0, y: window.innerHeight + wallThickness / 2 },
          ],
          walls[1]
        )
      );

      Matter.Body.setVertices(
        walls[2],
        Matter.Vertices.create(
          [
            { x: -wallThickness / 2, y: 0 },
            { x: wallThickness / 2, y: 0 },
            { x: wallThickness / 2, y: window.innerHeight },
            { x: -wallThickness / 2, y: window.innerHeight },
          ],
          walls[2]
        )
      );

      Matter.Body.setVertices(
        walls[3],
        Matter.Vertices.create(
          [
            { x: window.innerWidth - wallThickness / 2, y: 0 },
            { x: window.innerWidth + wallThickness / 2, y: 0 },
            { x: window.innerWidth + wallThickness / 2, y: window.innerHeight },
            { x: window.innerWidth - wallThickness / 2, y: window.innerHeight },
          ],
          walls[3]
        )
      );
    });
  });
</script>

<main style="height: 100vh">
  <div id="canvas-container"></div>
  <slot />
</main>

<style lang="scss">
  :global(html),
  :global(body) {
    background: linear-gradient(343deg, #07c090, #1971a9, #934646, #00ecf4);
    background-size: 800% 800%;
    animation: backgroundSwoop 74s ease infinite;
    overflow: hidden;
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
    overflow: hidden;
  }

  #canvas-container {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
  }

  canvas {
    display: block;
  }
</style>
