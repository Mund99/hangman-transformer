import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

// Project page served at https://<user>.github.io/hangman-transformer/
export default defineConfig(({ command }) => ({
  base: command === 'build' ? '/hangman-transformer/' : '/',
  plugins: [react()],
  server: { port: 5175, strictPort: true },
  resolve: { alias: { '@': path.resolve(__dirname, './src') } },
}))
