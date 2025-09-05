# Setup
import gradio as gr
import pandas as pd
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import CSVLoader
from langchain.chat_models import init_chat_model
from langchain_openai import OpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain.schema.output_parser import OutputParserException
from langchain_core.documents import Document
from pydantic import BaseModel
import os
from dotenv import load_dotenv

# TODO: Load environment variables and API Keys here

# Load and preprocess data
loader = CSVLoader(file_path="jiomart_products_database.csv", source_column="title")
documents_raw = loader.load()

# Convert page_content string to dict and build metadata
documents = []
for doc in documents_raw:
    try:
        row_data = dict(
            line.split(":", 1) for line in doc.page_content.split("\n") if ":" in line
        )
        row_data = {k.strip(): v.strip() for k, v in row_data.items()}

        page_text = f"Name: {row_data.get('title', '')} | Sub-type: {row_data.get('subType', '')} | Type: {row_data.get('type', '')} | Price: {row_data.get('discountedPrice', 0)} | Image: {row_data.get('filename', '')}"
        metadata = {
            "category": row_data.get("type", ""),
            "sub_category": row_data.get("subType", "")
        }

        documents.append(Document(page_content=page_text, metadata=metadata))

    except Exception as e:
        print("Skipping row due to error:", e)

# TODO: Create embeddings
# Use: OllamaEmbeddings with llama3.2:1b and pull the model beforehand

# TODO: Store in Chroma
# Use Chroma.from_documents() and .as_retriever() on only the first 100 docs

# TODO: Create Pydantic schema - Create two classes 1) GroceryItem with item_name, price, quantity, image_url and 2) GroceryOutput with reasoning and items. Pay attention to how you would create these classes

# Initialize parser
parser = PydanticOutputParser(pydantic_object=GroceryOutput)

# TODO: Prompt Template
# Use ChatPromptTemplate.from_template with fields: preferences, context, format_instructions

# TODO: Create a list of LLMs that will be used
# Use llama-3.3-70b-versatile and 2 other LLMs from Groq - make sure to check leaderboard to explore which ones to use

model_name_map = {
    "Llama-3.3": "llama-3.3-70b-versatile"
    #INSERT HERE
}

model_choices = [
    "Llama-3.3 (70B) (Groq)",
    #INSERT HERE
]
# TODO: Model selector
# Implement get_llm(model_choice, temperature) using init_chat_model
# def get_llm(model_choice, temperature):
    # if "Groq" in model_choice:
        # return init_chat_model(...)
        
# TODO: RAG pipeline
# Implement generate_cart(model_choice, user_input)
# Steps:
# 1. Get context from retriever using user_input["preferences"]
# 2. Format prompt with preferences, context, format instructions
# 3. Call LLM and parse output using parser
# 4. Handle OutputParserException and return fallback if needed

# def generate_cart(model_choice, user_input):
    # context_docs = 
    # relevant_text = 
    # llm = 
    # prompt = 
    # output =
    # try:
    #     result = parser.parse(output.content)
    # except OutputParserException:
    #     return "Could not parse output.", None
    # return result

# User Interface
def gradio_interface(preferences, model_choice, temperature):
    user_input = {
        "preferences": preferences,
        "model_choice": model_choice,
        "temperature": temperature,
    }
    result = generate_cart(model_choice, user_input)
    if not result or isinstance(result, str):
        return result, None
    if isinstance(result, tuple):
        explanation, _ = result
        return explanation, None

    # Build a gallery format: [(image_url, caption), ...]
    gallery_items = [
        (item.image_url, f"{item.item_name}\nQty.{item.quantity}\n₹{item.quantity*item.price}\n")
        for item in result.items
    ]
    return result.reasoning, gallery_items

demo = gr.Interface(
    fn=gradio_interface,
    inputs=[
        gr.Textbox(label="Describe your grocery needs (e.g., 'high protein, no besan or curd')"),
        gr.Dropdown(label="Model", choices=model_choices),
        gr.Slider(minimum=0.0, maximum=1.5, value=0.7, step=0.1, label="Temperature")
    ],
    outputs=[
        gr.Textbox(label="Considerations"),
        gr.Gallery(label="Shopping Cart", columns=3, height="auto")
    ],
    title="Smart Grocery Cart Assistant",
    description="Get a product list tailored to your dietary preferences."
)

# Launch UI
if __name__ == "__main__":
    demo.launch()
