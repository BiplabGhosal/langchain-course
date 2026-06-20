import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_openai import AzureChatOpenAI

load_dotenv()
from langchain.agents import tool, create_tool_calling_agent, AgentExecutor
from tavily import TavilyClient


@tool
def search(query: str) -> str:
    """
    Tool that searches over the internet for a given query and returns the results
    Args:
        query: The query to search for
    Returns:
        The search results
    """
    client = TavilyClient(api_key=os.environ.get("TAVILY_API_KEY"))
    print(f"Searching for {query}...")
    results = client.search(query=query)
    return str(results)

llm = AzureChatOpenAI(api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
        temperature=0,
        api_version=os.environ.get("AZURE_OPENAI_API_VERSION"),
        azure_deployment=os.environ.get("AZURE_OPENAI_DEPLOYMENT"),)

tools = [search]

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that can search the web."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools)


def main():
    result = agent_executor.invoke({"input": "Search for the latest news on SpaceX"} )
    print(result["output"])




# Function for langchain calls
# def main1():
#     print("Hello from langchain-course!")
#     print(os.environ.get("AZURE_OPENAI_ENDPOINT"))
#     information = """Musk in 2019 as SpaceX CEO; the company's IPO in 2026 led to him becoming the first US$ trillionaire. Having been first listed on the Forbes Billionaires List in 2012, around 75% of Musk's wealth was derived from Tesla stock in November 2020, although he describes himself as "cash poor"
#     """
#     summary_template = """
#     given the information {information} about a person I want you to create:
#     1. A short summary
#     2. Interesting facts about the person
#     """

#     summary_prompt_template = PromptTemplate(
#         input_variables=["information"],
#         template=summary_template,
#     )

#     llm = AzureChatOpenAI(
#         api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
#         azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
#         temperature=0,
#         api_version="2024-08-01-preview",
#         azure_deployment=os.environ.get("AZURE_OPENAI_DEPLOYMENT"),
#     )
#     chain = summary_prompt_template | llm
#     response= chain.invoke(input={"information": information})
#     print(response.content)



if __name__ == "__main__":
    main()
