<script>
  import { onMount } from "svelte";
  import Matter from "matter-js";

  export let id;
  export let color;

  let canvas;

  onMount(() => {
    const { Engine, Render, World, Bodies, Body } = Matter;

    const engine = Engine.create();
    const render = Render.create({
      element: canvas,
      engine: engine,
      options: {
        width: 800,
        height: 600,
        wireframes: false,
        background: "transparent",
      },
    });

    const blob = Bodies.circle(400, 200, 50, {
      render: {
        fillStyle: color,
        strokeStyle: color,
        lineWidth: 1,
      },
    });

    Body.setVelocity(blob, { x: 0.5, y: 0.5 });
    World.add(engine.world, [blob]);

    Matter.Runner.run(engine);
    Render.run(render);
  });
</script>

<canvas bind:this={canvas}></canvas>

<style>
  canvas {
    display: block;
  }
</style>
