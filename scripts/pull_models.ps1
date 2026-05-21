# Pull required Ollama models for the RAG chatbot.
# Run this once after installing Ollama: https://ollama.com

Write-Host "Pulling LLM model..."
ollama pull llama3.1:8b

Write-Host "Pulling embedding model..."
ollama pull nomic-embed-text

Write-Host "Done. Models ready."
Write-Host ""
Write-Host "If you have less than 8 GB VRAM, use llama3.2:3b instead:"
Write-Host "  ollama pull llama3.2:3b"
Write-Host "  Then set LLM_MODEL=llama3.2:3b in your .env file."
