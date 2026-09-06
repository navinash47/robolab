import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5173,
    proxy: {
      // Align with `make dev` / uvicorn (docs: API :8000). Was 8001 by mistake.
      "/api": "http://127.0.0.1:8000",
      "/health": "http://127.0.0.1:8000",
      "/spend": "http://127.0.0.1:8000",
      "/failures": "http://127.0.0.1:8000",
    },
  },
});
