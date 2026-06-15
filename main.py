import os

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import AzureChatOpenAI

load_dotenv()


def main():
    print("Hello from langchain-course!")
    print(os.environ.get("AZURE_OPENAI_ENDPOINT"))
    information = """Musk in 2019 as SpaceX CEO; the company's IPO in 2026 led to him becoming the first US$ trillionaire. Having been first listed on the Forbes Billionaires List in 2012, around 75% of Musk's wealth was derived from Tesla stock in November 2020, although he describes himself as "cash poor"
    """
    summary_template = """
    given the information {information} about a person I want you to create:
    1. A short summary
    2. Interesting facts about the person
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"],
        template=summary_template,
    )

    llm = AzureChatOpenAI(
        api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
        temperature=0,
        api_version="2024-08-01-preview",
        azure_deployment=os.environ.get("AZURE_OPENAI_DEPLOYMENT"),
    )
    chain = summary_prompt_template | llm
    response= chain.invoke(input={"information": information})
    print(response.content)



if __name__ == "__main__":
    main()
