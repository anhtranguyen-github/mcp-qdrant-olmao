import os
import sys
from dotenv import load_dotenv

# Ensure src is in the path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

# Load environment variables from .env file
load_dotenv()

# Import main from the package
from mcp_server_qdrant.main import main

if __name__ == "__main__":
    main()
