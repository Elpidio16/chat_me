import os

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import QueryType, VectorizedQuery
from openai import AzureOpenAI


def embed_query(query):
    with AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_KEY"),
        api_version=os.getenv("API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    ) as openai_client:
        response = openai_client.embeddings.create(
            input=query, model="text-embedding-ada-002"
        )
    return response.data[0].embedding

def search_ai(query: str, vector_query: list[float], k: int):
    credential = AzureKeyCredential(os.getenv("AZURE_SEARCH_ADMIN_KEY"))
    with SearchClient(
        endpoint=os.getenv("AZURE_SEARCH_SERVICE_ENDPOINT"),
        index_name=os.getenv("AZURE_SEARCH_INDEX"),
        credential=credential,
    ) as search_client:
        vector_query = VectorizedQuery(
            vector=vector_query,
            k_nearest_neighbors=k,
            fields="embedding",
            weight=0.5,
        )
        # Utilisation uniquement de la recherche vectorielle classique
        search_results = search_client.search(
            search_text=query,
            vector_queries=[vector_query],
            top=k,
        )
        results = list(search_results)
        clean_results = []
        for r in results:
            clean_r = {
                "content": r["content"],
                "score": r["@search.reranker_score"],
            }
            clean_results.append(clean_r)
    return clean_results


def history_to_messages(history, context):
    messages = [
        {
            "role": "system",
            "content": f"Tu es un assistant d'historien basé sur RAG ... {context}...",
        },
    ]
    for i in range(0, len(history), 15):
        messages.append({"role": "user", "content": history[i]["content"]})
        messages.append({"role": "assistant", "content": history[i + 14]["content"]})
    return messages



def chat(message, history, search_results):
    with AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_KEY"),
        api_version=os.getenv("API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    ) as openai_client:
        messages = history_to_messages(history, search_results)
        messages.append(
            {"role": "user", "content": message},
        )
        llm_response = openai_client.chat.completions.create(
            model="gpt-4o-2024-08-06",
            messages=messages,
        )
        llm_response = llm_response.choices[0].message.content
    return llm_response


