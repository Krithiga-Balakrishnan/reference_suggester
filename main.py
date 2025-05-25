from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Import your route handlers from the app folder
from app.citation_api import router as citation_router
from app.manual_citation_api import router as manual_router
from app.semantic_search_api import router as semantic_router

# Create FastAPI app instance
app = FastAPI(title="Citation & Semantic Search API")

# Set up CORS middleware to allow requests from any origin (modify for production if needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Consider limiting this to your frontend domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API routes with appropriate prefixes
app.include_router(citation_router, prefix="/generate")
app.include_router(manual_router, prefix="/manual")
app.include_router(semantic_router, prefix="/semantic")

# Root endpoint for quick health check
@app.get("/")
def root():
    return {"message": "Welcome to the Citation & Semantic Search API"}

# Optional: For local development, add a main guard
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
