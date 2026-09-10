import { defineConfig } from 'vite';

export default defineConfig({
  build: {
    // NVIDIA's streaming SDK is one ~700 kB module; the local viewer loads nothing else.
    chunkSizeWarningLimit: 1024,
  },
});
