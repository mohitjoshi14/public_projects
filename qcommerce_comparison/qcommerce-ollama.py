from langchain_community.document_loaders import AsyncChromiumLoader
from playwright.async_api import async_playwright
from langchain.document_transformers import BeautifulSoupTransformer
# from langchain.chains import create_extraction_chain
# from langchain_openai import ChatOpenAI
# from langchain_anthropic import ChatAnthropic
from langchain_ollama.chat_models import ChatOllama
from langchain_core.documents import Document
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
import asyncio
import nest_asyncio
import os
# import openai
# import anthropic
from functools import lru_cache
import logging
from asyncio import Semaphore
from typing import Dict, List
import json

nest_asyncio.apply()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

# openai.api_key = os.environ['OPENAI_API_KEY']
# anthropic.api_key = os.environ['ANTHROPIC_API_KEY']

import warnings
warnings.filterwarnings('ignore')

class QuickProductScraper:
    def __init__(self, max_concurrent_requests: int = 5):
        # self.llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")
        # self.llm = ChatAnthropic(temperature=0, model="claude-3.5-sonnet-20241022")
        self.llm = ChatOllama(temperature=0, model="llama3.2")
        self.browser = None
        self.context = None
        self.semaphore = Semaphore(max_concurrent_requests)
        self.schema = {
            "properties": {
                "product_name": {"type": "string"},
                "price": {"type": "string"},
                "category": {"type": "string"}
            }
        }

    async def initialize(self):
        if not self.browser:
            p = await async_playwright().start()
            self.browser = await p.chromium.launch(headless=True)
            self.context = await self.browser.new_context(
                geolocation={"latitude": float(os.getenv('MY_LAT')), "longitude": float(os.getenv('MY_LONG'))},
                # geolocation={"latitude": 15.7218522, "longitude": 74.35479537},
                permissions=["geolocation"],
                user_agent="Mozilla/5.0 (iPhone; CPU iPhone OS 14_4 like Mac OS X) AppleWebKit/605.1.15"
            )
            await self.context.grant_permissions(['geolocation'])

    async def cleanup(self):
        if self.browser:
            await self.browser.close()
            self.browser = None
            self.context = None

    @lru_cache(maxsize=100)
    async def get_page_content(self, url: str) -> str:
        if not self.browser:
            await self.initialize()
        
        page = await self.context.new_page()
        try:
            await page.goto(url, wait_until="domcontentloaded")
            await page.wait_for_load_state("domcontentloaded")
            await page.wait_for_load_state("load")
            await page.wait_for_load_state("networkidle")
            content = await page.content()
            return content
        except Exception as e:
            logger.error(f"Error fetching page content: {e}")
            raise
        finally:
            await page.close()
    
    async def search_products(self, product_name: str, url: str) -> List[Dict]:
        async with self.semaphore:
            try:
                search_url = f"{url}={product_name}"
                logger.info(f"Searching: {search_url}")
                content = await self.get_page_content(search_url)
                
                bs_transformer = BeautifulSoupTransformer()
                docs = bs_transformer.transform_documents(
                    [Document(page_content=content, metadata={})]
                )
                
                prompt = ChatPromptTemplate.from_messages([
                    ("system", """Extract product information in the following format:
                    [
                        {{
                            "product_name": "exact product name",
                            "price": "price in INR",
                            "category": "product category"
                        }}
                    ]
                    Only include products with clear prices and names."""),
                    ("human", "{text}")
                ])
                
                chain = prompt | self.llm | StrOutputParser()
                
                products = []
                for doc in docs:
                    try:
                        result = await chain.ainvoke({"text": doc.page_content})
                        parsed_data = json.loads(result)
                        if isinstance(parsed_data, list):
                            products.extend(parsed_data)
                        else:
                            products.append(parsed_data)
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse product data: {e}")
                        continue
                    
                return products
                
            except Exception as e:
                logger.error(f"Error searching products: {e}")
                return []



    # async def search_products(self, product_name: str, url: str) -> List[Dict]:
    #     async with self.semaphore:
    #         try:
    #             search_url = f"{url}={product_name}"
    #             logger.info(f"Searching: {search_url}")
                
    #             content = await self.get_page_content(search_url)
    #             bs_transformer = BeautifulSoupTransformer()
    #             docs = bs_transformer.transform_documents(
    #                 [Document(page_content=content, metadata={})]
    #             )

    #             chain = create_extraction_chain(self.schema, self.llm)
    #             products = []
                
    #             for doc in docs:
    #                 data = chain.run(doc.page_content)
    #                 if isinstance(data, list):
    #                     products.extend(data)
    #                 else:
    #                     products.append(data)
                
    #             return products
    #         except Exception as e:
    #             logger.error(f"Error searching products: {e}")
    #             return []

async def compare_results(product_name: str) -> str:
    companies = {
        "blinkit": "https://blinkit.com/s/?q",
        "zepto": "https://zeptonow.com/search?query"
    }
    
    scraper = QuickProductScraper()
    try:
        # Run all requests concurrently
        tasks = [
            scraper.search_products(product_name, url) 
            for company, url in companies.items()
        ]
        results = await asyncio.gather(*tasks)
        
        # Process results
        company_names = list(companies.keys())
        input_text = "\n".join(
            f"company name {company_names[i]}: Products: {result}" 
            for i, result in enumerate(results)
        )
        
        # Compare prices
        # model = ChatOpenAI(temperature=0, model="gpt-4o-mini")
        # model = ChatAnthropic(temperature=0, model="claude-3.5-sonnet-20241022")
        model = ChatOllama(temperature=0, model="llama3.2")
        compare_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a price comparison expert. Extract product names and prices, then: \
                1. Match identical products across companies \
                2. Convert all prices to numerical values \
                3. Create a price comparison table \
                4. Highlight the lowest price for each product \
                Ignore irrelevant products and invalid prices. Be concise and factual."),
            ("human", "Compare the products from companies {input_text}.")
        ])
        
        compare_chain = compare_prompt | model | StrOutputParser()
        return await compare_chain.ainvoke({"input_text": input_text})
    
    finally:
        await scraper.cleanup()

async def main():
    try:
        product_name = input("Enter product name to search: ")
        cheapest = await compare_results(product_name)
        print(cheapest)
    except Exception as e:
        logger.error(f"Error in main execution: {e}")

if __name__ == "__main__":
    asyncio.run(main())

