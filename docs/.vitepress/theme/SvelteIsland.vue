<script setup>
// Mounts a Svelte 5 component as an island inside a VitePress (Vue) page. The
// route imports the component and passes it here. Mount is synchronous in
// onMounted, which runs before the route's onVnodeMounted fires VitePress'
// content-updated callbacks, so the island's <h2>s are in the DOM when the "on
// this page" outline scans for them. `.qtut` on the target scopes an island
// stylesheet to this pane.
// No page mounts an island since the tutorial track was deleted.
import { ref, onMounted, onBeforeUnmount } from 'vue'
import { mount, unmount } from 'svelte'

const props = defineProps({
  component: { type: [Object, Function], required: true },
  props: { type: Object, default: () => ({}) },
})

const host = ref(null)
let app = null

onMounted(() => {
  app = mount(props.component, { target: host.value, props: props.props })
})

onBeforeUnmount(() => {
  if (app) {
    unmount(app)
    app = null
  }
})
</script>

<template>
  <div ref="host" class="qtut"></div>
</template>
