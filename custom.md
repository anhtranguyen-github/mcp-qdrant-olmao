QDRANT_URL="http://localhost:6333" \
COLLECTION_NAME="my-collection" \
FASTMCP_PORT=1234 \
uv run python mcp_qdrant_ollama.py --transport sse