# Uva AI Environment - API Reference

Uva AI environment uses LiteLLM, which is OpenAI-compatible. This means you can use the OpenAI SDK and most OpenAI compatible tools, you just need to point them to uva endpoint.

## 1. Your API Key

You've been provided with an API key, keep this key secure and never commit it to GitHub or share it publicly.

## 2. Base URL

Replace the standard OpenAI base URL with uva endpoint:

```
https://ai-research-proxy.azurewebsites.net/v1
```

## Installation

Install the OpenAI Python SDK:

```bash
pip install openai
```

## Example

```python
import openai
client = openai.OpenAI(
    api_key="your_api_key",
    base_url="https://ai-research-proxy.azurewebsites.net" # LiteLLM Proxy is OpenAI compatible, Read More: https://docs.litellm.ai/docs/proxy/user_keys
)

response = client.chat.completions.create(
    model="gpt-3.5-turbo", # model to send to the proxy
    messages = [
        {
            "role": "user",
            "content": "this is a test request, write a short poem"
        }
    ]
)

print(response)
```

### LangChain 

```bash
pip install langchain langchain-openai
```

```python
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.schema import HumanMessage, SystemMessage

chat = ChatOpenAI(
    openai_api_base="https://ai-research-proxy.azurewebsites.net",
    model = "gpt-3.5-turbo",
    temperature=0.1
)

messages = [
    SystemMessage(
        content="You are a helpful assistant that im using to make a test request to."
    ),
    HumanMessage(
        content="test from litellm. tell me why it's amazing in 1 sentence"
    ),
]
response = chat(messages)
print(response)
```

### LlamaIndex 

```bash
pip install llama-index
```

```python
import os, dotenv

from llama_index.llms import AzureOpenAI
from llama_index.embeddings import AzureOpenAIEmbedding
from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext

llm = AzureOpenAI(
    engine="azure-gpt-3.5",               # model_name on litellm proxy
    temperature=0.0,
    azure_endpoint="https://ai-research-proxy.azurewebsites.net", # litellm proxy endpoint
    api_key="sk-1234",                    # litellm proxy API Key
    api_version="2023-07-01-preview",
)
embed_model = AzureOpenAIEmbedding(
    deployment_name="azure-embedding-model",
    azure_endpoint="https://ai-research-proxy.azurewebsites.net",
    api_key="sk-1234",
    api_version="2023-07-01-preview",
)

documents = SimpleDirectoryReader("llama_index_data").load_data()
service_context = ServiceContext.from_defaults(llm=llm, embed_model=embed_model)
index = VectorStoreIndex.from_documents(documents, service_context=service_context)

query_engine = index.as_query_engine()
response = query_engine.query("What did the author do growing up?")
print(response)
```


- OpenAI SDK documentation: https://platform.openai.com/docs
- LangChain documentation: https://python.langchain.com/docs
- LlamaIndex documentation: https://docs.llamaindex.ai
- Litellm documentation: https://docs.litellm.ai/docs/

---
 