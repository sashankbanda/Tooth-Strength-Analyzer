from fastapi import FastAPI
from api.routes import router
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load models on startup
    print("Startup: Initializing models...")
    from api.routes import get_segmentors
    get_segmentors() # Trigger loading
    yield
    print("Shutdown: Cleaning up...")

app = FastAPI(
    title="Automated Tooth Strength Analysis API",
    description="API for analyzing panoramic X-rays to determine tooth strength.",
    version="1.0.0",
    lifespan=lifespan
)

app.include_router(router, prefix="/api/v1")

@app.get("/")
def health_check():
    return {"status": "healthy", "service": "tooth-strength-analysis"}
