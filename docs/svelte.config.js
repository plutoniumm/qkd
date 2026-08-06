import { vitePreprocess } from '@sveltejs/vite-plugin-svelte';

// Read automatically by @sveltejs/vite-plugin-svelte (Vite root is docs/). Lets
// the explainer .svelte files use <script lang="ts">.
export default { preprocess: vitePreprocess() };
