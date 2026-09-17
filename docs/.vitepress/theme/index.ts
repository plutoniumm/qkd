// Custom VitePress theme = the default theme + one global component, <SvelteIsland>,
// which lets any markdown route mount a Svelte 5 explainer (see ./SvelteIsland.vue).
import DefaultTheme from 'vitepress/theme';
import type { Theme } from 'vitepress';

import SvelteIsland from './SvelteIsland.vue';

export default {
  extends: DefaultTheme,
  enhanceApp({ app }) {
    app.component('SvelteIsland', SvelteIsland);
  },
} satisfies Theme;
