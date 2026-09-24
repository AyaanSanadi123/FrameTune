# api/server.py

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# 1. Import the API Router (The HTTP Department)
from api.routes.video_routes import router as video_router

# 2. Import the ML Engine (The Heavy Lifting)
from video_pipeline.config import settings
from video_pipeline.core import vifm_engine
from video_pipeline.core.vifm_engine import initialize_vifm
from video_pipeline.core.coordinate import UniversalCoordinateSpace

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Executes on server boot to load massive ML models into GPU VRAM exactly once.
    Executes on server shutdown to clear memory safely.
    """
    print("[Server Boot] Initializing FrameTune Infrastructure...")
    
    # Step A: Load the heavy 3D Transformer model into VRAM
    print("[Server Boot] Loading Video Foundation Model (LanguageBind)...")
    initialize_vifm()
    
    # Step B: Instantiate the coordinate grid and calculate the text anchors
    print("[Server Boot] Building Universal Coordinate Space...")
    
    # Step C: Inject the warm object into the global app state
    # This makes it instantly available to any API route without reloading
    app.state.coord_space = UniversalCoordinateSpace(
        model=vifm_engine._model,
        processor=vifm_engine._processor,
        device=vifm_engine._device,
        anchors_config=settings
    )
    
    print("[Server Boot] VRAM loaded successfully. API is live and accepting traffic.")
    
    yield  # --- Server pauses here and listens for HTTP requests ---
    
    print("[Server Shutdown] Clearing models from VRAM...")
    app.state.coord_space = None
    vifm_engine._model = None
    vifm_engine._processor = None

# Initialize the FastAPI motherboard
app = FastAPI(
    title="FrameTune Backend API",
    description="Spatiotemporal video analysis and audio matching backend.",
    version="1.0.0",
    lifespan=lifespan
)

# Security: Configure Cross-Origin Resource Sharing (CORS)
# This allows your Next.js frontend to securely send files to this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update this to your frontend URL (e.g., http://localhost:3000) in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Plug the video department into the main infrastructure
app.include_router(video_router, prefix="/api/v1/video", tags=["Video Pipeline"])

# Health check endpoint for deployment monitoring
@app.get("/health", tags=["System"])
def health_check():
    """Simple ping to verify the server is running and the GPU is warm."""
    is_warm = hasattr(app.state, "coord_space") and app.state.coord_space is not None
    return {"status": "online", "gpu_warmed": is_warm}