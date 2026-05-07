from llama_index.core import StorageContext, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

#  从本地加载索引
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-zh-v1.5")
storage_context = StorageContext.from_defaults(persist_dir="./llamaindex_index_store")
vectorStore = load_index_from_storage(storage_context, embed_model=embed_model)

# 使用检索器进行查询（无LLM生成）
retriever = vectorStore.as_retriever()
results = retriever.retrieve("LlamaIndex是什么?")
for result in results:
    print(result)