# Use a slim Python base image
FROM python:3.11-slim

LABEL maintainer="AI Assistant"
LABEL description="MCP service for interacting with a Python RAG server."

WORKDIR /app

# Copy project configuration and source code
COPY pyproject.toml ./
COPY README.md ./
COPY src ./src

# Install the project and its dependencies from pyproject.toml.
# This also makes the mcp-rag-server script available in PATH.
RUN pip install --no-cache-dir .

# Set default environment variables.
# These can be overridden at runtime with docker run -e ...
# For Linux, you might need to use --network=host or pass --add-host=host.docker.internal:host-gateway
ENV RAG_SERVER_URL="http://host.docker.internal:8000"

# The DB path should be mounted as a volume to persist data.
# This default path assumes the main RAG server's data volume is mounted here.
ENV DB_PATH="/data/rag.duckdb"

# The server communicates via stdio, so no port is exposed.

# The entrypoint is the script installed from pyproject.toml
CMD ["mcp-rag-server"] 