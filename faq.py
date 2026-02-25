from typing import TypedDict, List, Literal, Annotated
import operator
from typing import TypedDict, Any, List, Literal, Annotated
from abc import ABC, abstractmethod
import os
import re
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from pydantic import BaseModel, Field
from typing import Literal, List, Optional
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
gemini_api_key = os.getenv("GEMINI_API_KEY") 
llm = ChatGoogleGenerativeAI(model='gemini-2.5-flash-lite',google_api_key=gemini_api_key)
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
parser=StrOutputParser()
import os
import uuid
import time
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# --- 1. INITIALIZE COMPONENTS ---
# Initialize the embedding model (384 dimensions)
dense_embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Initialize Pinecone Client
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = "agriculture"


# Your specific agricultural schema
CROPS = [
    "wheat", "rice", "maize", "barley", "jowar", "ragi", 
    "moong", "horsegram", "soyabean", "mustard", "sesame", 
    "sunflower", "cotton", "jute", "onion", "potato", 
    "sweet_potato", "tapioca", "banana", "coriander", 
    "garlic", "turmeric", "black_pepper", "cashewnuts", "arecanut"
    ]
TYPES = ["weed_control", "nutrient_management", "pest_control", "disease_symptoms", "pest_insects"]

class SearchTask(BaseModel):
    query: str = Field(description="The semantic search string")
    crop: str = Field(description="The specific crop filter")
    type: str = Field(description="The category filter")

class AgentState(TypedDict):
    question: str
    tasks: List[SearchTask] # List of split queries + filters
    queries:List[str]
    context: Annotated[List[str], operator.add] # Aggregates all results
    answer: str
    
from pydantic import BaseModel, Field
from typing import List

# 1. Your existing model
class SearchTask(BaseModel):
    crop: str = Field(description="The crop name")
    type: str = Field(description="The information category")

# 2. THE FIX: Create a wrapper class
class SearchTaskList(BaseModel):
    """A collection of search tasks identified from the user query."""
    tasks: List[SearchTask]

# 3. Update your analyzer_node
def analyzer_node(state: AgentState):
    print("🧠 Analyzing query for crops and types...")
    
    prompt = f"""
    Analyze the question: "{state['question']}"
    Identify all crops from this list: {CROPS}
    Identify all types from this list: {TYPES}
    
    If the user mentions multiple crops or types, create a separate SearchTask for EACH combination.
    Example: "Wheat and Rice pests" -> [Task(Wheat, Pest), Task(Rice, Pest)]
    """
    
    # ✅ Update: Use the wrapper class here
    structured_llm = llm.with_structured_output(SearchTaskList)
    
    # This will now return a SearchTaskList object
    result = structured_llm.invoke(prompt)
    
    # Access the list inside the object
    raw_tasks = result.tasks 
    
    refined_tasks = []
    for task in raw_tasks:
        t_type = task.type.lower()
        t_crop = task.crop

        if "weed" in t_type:
            print(f"🔄 Routing {t_crop} weed task to 'common' database...")
            task.crop = "common"
        
        refined_tasks.append(task)

    return {"tasks": refined_tasks}    
    
from pydantic import BaseModel, Field
from typing import List

# 1. Define the schema for our individual query dictionaries
class SubQuery(BaseModel):
    text: str = Field(description="The enhanced semantic search query string")
    crop: str = Field(description="The crop this query is focused on")
    type: str = Field(description="The specific information category (e.g., pest_control)")

# 2. Define the wrapper for the LLM output
class MultiQueryOutput(BaseModel):
    queries: List[SubQuery] = Field(description="List of structured search tasks")

def multiquery_node(state: AgentState):
    """
    Transforms tasks into a list of dictionaries for high-precision RAG.
    """
    print("🔄 Generating Structured Multi-Queries...")
    
    # We use .with_structured_output to force the LLM to follow our schema
    structured_llm = llm.with_structured_output(MultiQueryOutput)
    
    # If tasks were already identified, we ask the LLM to polish them.
    # If not, we ask it to expand the original question.
    prompt = f"""
    The user is asking: "{state['question']}"
    Current identified tasks: {state.get('tasks', [])}
    ### KNOWLEDGE RULE:(only APPLICABLE FOR WEED TYPE)
    - WEED MANAGEMENT: For all tasks where type is 'weed_control', treat the advice as UNIVERSAL/COMMON. 
    - Even if the user asked about a specific crop, weed control strategies (like Parthenium management) apply to all.
    - Use 'common' as the crop identifier for all weed-related queries.
    Apply these expansion rules:
    1. TECHNICAL: Use scientific names and formal agricultural terminology.
    2. SYMPTOMATIC: Focus on how a farmer identifies the problem in the field.
    3. PROCEDURAL: Focus on step-by-step management or 'how-to' instructions.
    For each task, create a highly descriptive search query. 
    If only one task exists, create 3 different semantic versions of the query.
    
    Every query MUST be returned as a dictionary with 'text', 'crop', and 'type'.
    - 'text': A professional, standalone search query.
    - 'crop': The specific crop name.
    - 'type': The specific category.
    """
    
    result = structured_llm.invoke(prompt)
    
    # Returning a list of dictionaries as requested
    # [{'text': '...', 'crop': '...', 'type': '...'}, ...]
    return {"queries": [q.dict() for q in result.queries]}    
def retrieve_node(state: AgentState):
    """
    Executes semantic search for each expanded query dictionary with strict metadata filtering.
    """
    index = pc.Index("agriculture")
    all_docs = []
    seen_ids = set() # To prevent duplicate chunks from multiple queries
    
    # We now iterate over the 'queries' list of dicts created in multiquery_node
    for q_item in state["queries"]:
        query_text = q_item["text"]
        target_crop = q_item["crop"]
        target_type = q_item["type"]
        
        print(f"🔍 Semantic Search: '{query_text[:50]}...' | Filter: {target_crop} + {target_type}")
        
        # 1. Generate vector for the expanded text
        query_vector = dense_embeddings.embed_query(query_text)
        
        # 2. Run Query with strict Metadata Filtering ($eq)
        # top_k=3 is usually ideal when using 3+ expanded queries
        res = index.query(
            vector=query_vector,
            top_k=5, 
            filter={
                "crop": {"$eq": target_crop},
                "type": {"$eq": target_type}
            },
            include_metadata=True
        )
        
        # 3. Process and Deduplicate
        for match in res['matches']:
            doc_id = match['id']
            if doc_id not in seen_ids:
                # Add source info so the LLM knows exactly which crop/type this is for
                content = f"DATABASE_RECORD (Crop: {target_crop}, Category: {target_type}): {match['metadata']['text']}"
                all_docs.append(content)
                seen_ids.add(doc_id)
    
    print(f"✅ Retrieval Complete. Total unique context chunks captured: {len(all_docs)}")
    
    # Return the aggregated context to the state
    return {"context": all_docs}
def generate_answer_node(state: AgentState):
    """
    Synthesizes the multi-crop, multi-type context into a structured agricultural report.
    """
    print("✍️ Synthesizing final research report...")
    
    # 1. Prepare the context string
    # We join the chunks with clear separators so the LLM sees the source tags
    context_str = "\n\n---\n\n".join(state["context"])
    
    # 2. Define the Mentor Prompt
    # We explicitly tell the LLM to use the metadata tags (Crop/Category) found in the context
    prompt = f"""
    You are an Expert Agricultural Researcher and Mentor. 
    Your goal is to provide a structured answer based on the provided database records to the user query.

    ORIGINAL QUESTION: {state['question']}

    DATABASE RECORDS:
    {context_str}
    ### KNOWLEDGE RULE:(only APPLICABLE FOR WEED TYPE)
    - WEED MANAGEMENT: For all tasks where type is 'weed_control', treat the advice as UNIVERSAL/COMMON. 
    - Even if the user asked about a specific crop, weed control strategies (like Parthenium management) apply to all.
    - Use 'common' as the crop identifier for all weed-related queries.
    INSTRUCTIONS:
    1. Organize the answer by CROP first. 
    2. If the context is empty or insufficient for a specific request, clearly state that no records were found in the database currently.
    3. Maintain a professional yet helpful mentor tone.
    4. At the end of each ans, if a 'source' URL was provided in the context, cite it.
     *****NOTE
     AT THE END OF ANSWER GIVE THE SOURCE OF THE DATA FROM THE METADATA PROVIDED TO YOU .THIS WILL BE THE LINK OF THE WEBSITE THE DATA IS GATHERED FROM.
    :"""

    # 3. Call Gemini
    response = llm.invoke(prompt)
    
    # Return the f inal answer to the state
    return {"answer": response.content}  

workflow = StateGraph(AgentState)

workflow.add_node("analyzer", analyzer_node)
workflow.add_node("retriever", retrieve_node)
workflow.add_node("multiquery_node", multiquery_node)

workflow.add_node("generator", generate_answer_node) # Your previous generation logic

workflow.add_edge(START, "analyzer")
workflow.add_edge("analyzer", "multiquery_node")
workflow.add_edge("multiquery_node","retriever")

workflow.add_edge("retriever", "generator")
workflow.add_edge("generator", END)

research_app = workflow.compile()  
