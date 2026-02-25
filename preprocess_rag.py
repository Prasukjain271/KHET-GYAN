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

# --- 2. ENSURE INDEX EXISTS ---
if not pc.has_index(index_name):
    print(f"🏗️ Creating index: {index_name}...")
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="dotproduct",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    # Wait for the serverless index to be ready
    while not pc.describe_index(index_name).status['ready']:
        time.sleep(2)

# --- 3. THE IMPROVED INGESTOR FUNCTION ---
def add_document_to_researcher(file_path, metadata_tags, chunk_size=600, chunk_overlap=100):
    """
    Reads a .txt file, tags it with metadata, splits it, and uploads to Pinecone.
    """
    if not os.path.exists(file_path):
        print(f"❌ Error: {file_path} not found.")
        return
    
    # Simple file handling
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Tagging with agricultural metadata
    doc = [Document(page_content=content, metadata=metadata_tags)]
    
    # Splitting into chunks to stay within context limits
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_documents(doc)
    print(f"✅ Created {len(chunks)} chunks for {metadata_tags.get('crop', 'Unknown')}")

    # Embed documents (Math conversion)
    texts = [c.page_content for c in chunks]
    embeddings = dense_embeddings.embed_documents(texts)
    
    # Connect to index and upsert vectors
    index = pc.Index(index_name)
    vectors = []
    for i, (chunk, vector) in enumerate(zip(chunks, embeddings)):
        final_meta = chunk.metadata.copy()
        final_meta["text"] = chunk.page_content # Essential for RAG
        
        vectors.append({
            "id": str(uuid.uuid4()),
            "values": vector,
            "metadata": final_meta
        })
    
    # Upsert in one go for smaller files
    index.upsert(vectors=vectors)
    print(f"🚀 Successfully uploaded {len(vectors)} vectors to '{index_name}' index.")

# --- EXECUTION ---
# Now you can easily ingest your data
#add_document_to_researcher("weed.txt", {"crop": "Common", "type": "weed_control"})
# add_document_to_researcher("maize_tips.txt", {"crop": "Maize", "type": "nutrient_management"})
#add_document_to_researcher("wheat_pest.txt", {"crop": "wheat", "type": "pest_control","source":"https://agriculture.vikaspedia.in/viewcontent/agriculture/crop-production/integrated-pest-managment/ipm-for-cerels/ipm-strategies-for-wheat/wheat-crop-stage-wise-ipm?lgn=en"})
#add_document_to_researcher("sasme_pest.txt", {"crop": "seasme", "type": "pest_control","source":"https://agriculture.vikaspedia.in/viewcontent/agriculture/crop-production/integrated-pest-managment/ipm-for-oilseeds/ipm-strategies-for-sesame/seasme-crop-stage-wise-ipm?lgn=en"})
#add_document_to_researcher("sunflower_pest.txt", {"crop": "sunflower", "type": "pest_control","source":"https://agriculture.vikaspedia.in/viewcontent/agriculture/crop-production/integrated-pest-managment/ipm-for-oilseeds/ipm-strategies-for-sunflower/ipm-strategies-of-sunflower?lgn=en"})
#add_document_to_researcher("turmeric_pest.txt", {"crop": "turmeric", "type": "pest_control","source":"https://agriculture.vikaspedia.in/viewcontent/agriculture/crop-production/integrated-pest-managment/ipm-for-spice-crops/ipm-strategies-for-turmeric/crop-stage-wise-ipm?lgn=en"})
#add_document_to_researcher("wheat_disease_syn.txt", {"crop": "wheat", "type": "disease_symptoms","source":"https://agriculture.vikaspedia.in/viewcontent/agriculture/crop-production/integrated-pest-managment/ipm-for-cerels/ipm-strategies-for-wheat/wheat-diseases-and-symptoms?lgn=en"})
#add_document_to_researcher("wheat_insects.txt", {"crop": "wheat", "type": "pest_insects","source":"https://agriculture.vikaspedia.in/viewcontent/agriculture/crop-production/integrated-pest-managment/ipm-for-cerels/ipm-strategies-for-wheat/wheat-insect-mites-and-nematode-pests-management?lgn=en"})
