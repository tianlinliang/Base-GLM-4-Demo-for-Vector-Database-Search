import os
import json
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter

# 设置 NVIDIA API 密钥
os.environ["NVIDIA_API_KEY"] = "put_your_nvidia_ai_api_here"

# 加载嵌入模型
embedder = NVIDIAEmbeddings(model="ai-embed-qa-4")

# 使用相对路径，文件应该为和示例的doc1文档以及doc2文档一致，为一段无空行的文字内容。
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
file_paths = [
    os.path.join(base_path,  "tools", "doc1.txt"),
    os.path.join(base_path,  "tools", "doc2.txt")
]

# 确认生成的文件路径
print("文件路径：", file_paths)
# 初始化文本分割器
text_splitter = CharacterTextSplitter(chunk_size=400, separator=" ")

# 初始化存储分割后的文本和元数据的列表
docs = []
metadatas = []

# 读取文件内容并进行分割
for file_path in file_paths:
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
        splits = text_splitter.split_text(content)
        docs.extend(splits)
        metadatas.extend([{"source": file_path}] * len(splits))

# 打印分割后的文本块和元数据检查格式是否正确
for i, (doc, metadata) in enumerate(zip(docs, metadatas)):
    print(f"文本块 {i+1}:\n{doc}\n")
    print(f"元数据 {i+1}:\n{metadata}\n")

# 创建向量存储
store = FAISS.from_texts(docs, embedder, metadatas=metadatas)
store.save_local('./nv_embedding')

# 加载向量存储并进行查询
store = FAISS.load_local("./nv_embedding", embedder, allow_dangerous_deserialization=True)
retriever = store.as_retriever()

#测试向量数据库是否成功建立，可以根据自己的文本内容进行修改
query = "数据科学"
results = retriever.get_relevant_documents(query)

# 手动提取results中可序列化的部分
serializable_results = []
for result in results:
    serializable_results.append({
        'metadata': result.metadata,
        'page_content': result.page_content
    })

# 输出查询结果
print(json.dumps(serializable_results, ensure_ascii=False, indent=2))
