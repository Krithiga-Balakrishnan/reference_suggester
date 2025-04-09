from dotenv import load_dotenv
load_dotenv()
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Import your route handlers
from app.citation_api import router as citation_router
from app.manual_citation_api import router as manual_router
from app.semantic_search_api import router as semantic_router

# Create FastAPI app
app = FastAPI(title="Citation & Semantic Search API")

# Allow frontend access (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can replace with ["http://localhost:3000"] for frontend only
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register your API routes
app.include_router(citation_router, prefix="/citation")
app.include_router(manual_router, prefix="/manual")
app.include_router(semantic_router, prefix="/semantic")

@app.get("/")
def root():
    return {"message": "🚀 Welcome to the Citation & Semantic Search API"}
