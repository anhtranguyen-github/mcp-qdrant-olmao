import asyncio
import logging

try:
    import ollama
except ImportError:
    ollama = None

from mcp_server_qdrant.embeddings.base import EmbeddingProvider

logger = logging.getLogger(__name__)

class OllamaEmbeddingProvider(EmbeddingProvider):
    """
    Ollama implementation of the embedding provider.
    :param model_name: The name of the Ollama model to use.
    :param host: The host URL of the Ollama server.
    """

    def __init__(self, model_name: str, host: str):
        if ollama is None:
            raise ImportError("Ollama python package is not installed. Please install it with `pip install ollama`.")
        
        self.model_name = model_name
        self.host = host
        self.client = ollama.Client(host=self.host)

    async def embed_documents(self, documents: list[str]) -> list[list[float]]:
        """Embed a list of documents into vectors."""
        # Ollama python client is synchronous, so we run it in an executor
        loop = asyncio.get_event_loop()
        
        def _embed_batch():
            embeddings = []
            for doc in documents:
                response = self.client.embeddings(model=self.model_name, prompt=doc)
                embeddings.append(response["embedding"])
            return embeddings

        return await loop.run_in_executor(None, _embed_batch)

    async def embed_query(self, query: str) -> list[float]:
        """Embed a query into a vector."""
        loop = asyncio.get_event_loop()
        
        def _embed_one():
            response = self.client.embeddings(model=self.model_name, prompt=query)
            return response["embedding"]

        return await loop.run_in_executor(None, _embed_one)

    def get_vector_name(self) -> str:
        """Return the name of the vector for the Qdrant collection."""
        # Sanitize model name to be safe for vector name
        safe_name = self.model_name.replace(":", "_").replace("/", "_")
        return f"ollama-{safe_name}"

    def get_vector_size(self) -> int:
        """Get the size of the vector for the Qdrant collection."""
        try:
            # We do a dummy embedding to get the size
            response = self.client.embeddings(model=self.model_name, prompt="test")
            return len(response["embedding"])
        except Exception as e:
            logger.error(f"Failed to get vector size for model {self.model_name}: {e}")
            raise
