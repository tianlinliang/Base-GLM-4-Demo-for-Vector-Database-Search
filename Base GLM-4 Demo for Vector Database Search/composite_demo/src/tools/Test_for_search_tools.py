import json
from typing import List
import sys
import os

# 将父目录添加到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 现在可以导入工具注册模块
from tools.tool_registry import dispatch_tool, get_tools


'''
该函数为测试LLM工具是否正确注册成功，并且对实现的向量数据库的功能进行测试，返回其获取到的向量数据库的结果
'''

def test_vector_db_query():
    # Define a sample query
    query = "数据科学"

    # Format the query as a JSON string as expected by dispatch_tool
    query_json = json.dumps({"query": query})

    # Call the tool using dispatch_tool
    results = dispatch_tool("vector_db_query", query_json, "test_session")

    # Parse the results
    if results and results[0].content_type != "system_error":
        output = results[0].text
        parsed_results = json.loads(output)

        # Print the results for verification
        print(json.dumps(parsed_results, ensure_ascii=False, indent=2))
    else:
        print("Error: ", results[0].text)


def test_tool_registration():
    # Get the list of registered tools
    tools = get_tools()

    # Print the registered tools for verification
    print(json.dumps(tools, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    print("Testing tool registration...")
    test_tool_registration()

    print("\nTesting vector DB query...")
    test_vector_db_query()
