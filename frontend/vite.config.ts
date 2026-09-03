import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      // Static file uploads are served by the backend at /api/uploads/*, so
      // forward that path verbatim (no /api strip). Must come BEFORE the
      // generic /api rule below.
      "/api/uploads": {
        target: "http://localhost:8000",
        changeOrigin: true,
      },
      // Proxy /api/* to the FastAPI backend during development, stripping
      // the /api prefix so the backend (which does not mount its routes
      // under /api) sees the same paths in dev and prod.
      "/api": {
        target: "http://localhost:8000",
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ""),
      },
    },
  },
});
