from playwright.async_api import async_playwright
from langchain.document_transformers import BeautifulSoupTransformer
from langchain.chains import create_extraction_chain
from langchain.chat_models import ChatOpenAI
from langchain_core.documents import Document
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
import asyncio
import nest_asyncio
import os
import openai

nest_asyncio.apply()

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

openai.api_key = os.environ['OPENAI_API_KEY']

import warnings
warnings.filterwarnings('ignore')


# Class to scrape result pages of quickcommerce sites
class QuickProductScraper:
    def __init__(self):
        self.llm = ChatOpenAI(
            temperature=0,
            model="gpt-4o-mini"
        )
        self.schema = {
            "properties": {
                "product_name": {"type": "string"},
                "price": {"type": "string"},
                "category": {"type": "string"}
            }
        }

    from playwright.async_api import async_playwright

    async def get_page_content(self, url):
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=False)
            context = await browser.new_context(
                geolocation={"latitude": 15.7218522, "longitude": 74.35479537},
                permissions=["geolocation"]
            )
            page = await context.new_page()
            try:
                await page.goto(url, wait_until="domcontentloaded")
                # Wait for critical elements that indicate page is ready
                await page.wait_for_load_state("domcontentloaded")
                await page.wait_for_load_state("load")
                
                # Wait for dynamic content )
  
                # await page.wait_for_selector("div.products", state="visible", timeout=10000)
                
                # Ensure network is idle
                await page.wait_for_load_state("networkidle")
                
                content = await page.content()
                return content
            finally:
                await page.close()
    
    async def search_products(self, product_name,url):
        search_url = f"{url}={product_name}"
        print(search_url)
        # Get page content with geolocation
        content = await self.get_page_content(search_url)
        
        # Transform HTML
        bs_transformer = BeautifulSoupTransformer()
        docs = bs_transformer.transform_documents([Document(page_content=content, metadata={})])
        
        # Extract product data
        chain = create_extraction_chain(self.schema, self.llm)
        products = []
        
        for doc in docs:
            data = chain.run(doc.page_content)
            if isinstance(data, list):
                products.extend(data)
            else:
                products.append(data)
                
        return products


async def compare_results(product_name):

    companies = {
                "blinkit" : "https://blinkit.com/s/?q", 
                "zepto" : "https://zeptonow.com/search?query"
                }
    
    scraper = QuickProductScraper()
    results = []
    company_names = list(companies.keys())

    for company,url in companies.items():
        results.append(await scraper.search_products(product_name, url))
    
    print(results)

    input_text = ""
    for i in range(len(results)):
        input_text += f"company name {company_names[i]}: Products: {results[i]} \n"
    print(input_text)
    
    model = ChatOpenAI(
                temperature=0,
                model="gpt-4o-mini"
            )
   
    compare_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are an expert product reviewer. You will take as input the prices scraped from company websites \
             and compare prices of the same product across these companies.  \
             You will indicate which company has the cheapest price for each product. \
             First show a summary of comparison and then provide a detailed comparison."),
            ("human", "Compare the products from companies {input_text}."),
        ]
    )
    compare_chain = compare_prompt | model | StrOutputParser()
    compare_output = await compare_chain.ainvoke({"input_text": input_text})
    return compare_output

async def main():
    product_name = input("Enter product name to search: ")
    cheapest = await compare_results(product_name)
    print(cheapest)

if __name__ == "__main__":
    asyncio.run(main())