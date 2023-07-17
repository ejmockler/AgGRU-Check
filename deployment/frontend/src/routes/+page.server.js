import { API_URL } from "$env/static/private";

/** @type {import('./$types').Actions} */
export const actions = {
  predict: async ({ request }) => {
    console.log(await request.formData());
    // TODO use server-sent events over websocket for Sveltekit compatibility
  },
};
