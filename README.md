# Multi-agent AI sample with Azure Cosmos DB

A sample personal shopping AI Chatbot that can help with product enquiries, making sales, and refunding orders by transferring to different agents for those tasks.

Features:
- **Multi-agent**: [LangGraph](https://www.langchain.com/langgraph) to orchestrate multi-agent interactions with [Azure OpenAI](https://learn.microsoft.com/azure/ai-services/openai/overview) API calls.
- **Transactional data management**: planet scale [Azure Cosmos DB database service](https://learn.microsoft.com/azure/cosmos-db/introduction) to store transactional user and product operational data.
- **Persistent Memory**: save chat memory and agent state using the native LangGraph [checkpoint implementation for Azure Cosmos DB](https://pypi.org/project/langgraph-checkpoint-cosmosdb/).
- **Multi-tenant session storage**: [Hierarchical Partitioning](https://learn.microsoft.com/azure/cosmos-db/hierarchical-partition-keys) is used to manage each user session (this can be adapted for multi-tenancy). 
- **Retrieval Augmented Generation (RAG)**: [vector search](https://learn.microsoft.com/azure/cosmos-db/nosql/vector-search) in Azure Cosmos DB with powerful [DiskANN index](https://www.microsoft.com/en-us/research/publication/diskann-fast-accurate-billion-point-nearest-neighbor-search-on-a-single-node/?msockid=091c323873cd6bd6392120ac72e46a98) to serve product enquiries from the same database.
- **Gradio UI**: [Gradio](https://www.gradio.app/) to provide a simple UI ChatBot for the end-user.

## Backend agent activity

Run as an interactive session to see the agent handoffs in action...

![Demo](./media/demo-cli.gif)

## Front-end AI chat bot

Run the AI chat bot for the end-user experience...

![Demo](./media/demo-chatbot.gif)

## Overview

The personal shopper example includes four main agents to handle various customer service requests:

1. **Triage Agent**: Determines the type of request and transfers to the appropriate agent.
2. **Product Agent**: Answers customer queries from the products container using [Retrieval Augmented Generation (RAG)](https://learn.microsoft.com/azure/cosmos-db/gen-ai/rag).
2. **Refund Agent**: Manages customer refunds, requiring both user ID and item ID to initiate a refund.
3. **Sales Agent**: Handles actions related to placing orders, requiring both user ID and product ID to complete a purchase.

## Prerequisites

- [Azure Cosmos DB account](https://learn.microsoft.com/azure/cosmos-db/create-cosmosdb-resources-portal) - ensure the [vector search](https://learn.microsoft.com/azure/cosmos-db/nosql/vector-search) feature is enabled.
- [Azure OpenAI API key](https://learn.microsoft.com/azure/ai-services/openai/overview) and endpoint.
- [Azure OpenAI Embedding Deployment ID](https://learn.microsoft.com/azure/ai-services/openai/overview) for the RAG model.

## Setup

Clone the repository:

```shell
git clone https://github.com/AzureCosmosDB/multi-agent-langgraph
cd multi-agent-langgraph
```

Install dependencies:

```shell
pip install -r src/app/requirements.txt 
```

Ensure you have the following environment variables set:
```shell
COSMOSDB_ENDPOINT=your_cosmosdb_account_uri
COSMOSDB_KEY=your_cosmosdb_account_key
AZURE_OPENAI_API_KEY=your_azure_openai_api_key
AZURE_OPENAI_ENDPOINT=your_azure_openai_endpoint
AZURE_OPENAI_EMBEDDINGDEPLOYMENTID=your_azure_openai_embeddingdeploymentid
```

Once you have installed dependencies, run azure_cosmos_db.py to create the required Cosmos DB containers and load the sample data.
**Note**: Ensure that you have enabled both vector search capability in your Cosmos DB account. Please wait at least 10 minutes after enabling this feature before running the sample code.
**Note**: The Products container is initially created with 10000 RUs for faster loading of 1000 products. If you are using a free tier account, you may need to reduce the RUs to avoid exceeding the free tier limit. You can also reduce the RUs after the initial data load to avoid excessive charges.

```shell
python src/app/azure_cosmos_db.py
```

Then run the chat bot code below and click on url provided in the output:

```shell
python src/app/ai_chat_bot.py
```

To see the agent transfers, you can also run as an interactive CLI session using:
    
```shell
python3 src/app/multi_agent_service.py
```

