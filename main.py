from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.perturbed_image import router as perturbed_image_router

# To run: uvicorn main:app --reload
# To run: uvicorn main:app --host 0.0.0.0 --port 8000

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://protectures.vercel.app/"],  # Replace "*" with specific allowed origins for production
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, OPTIONS, etc.)
    allow_headers=["*"],  # Allow all headers
)

# Include the router
app.include_router(perturbed_image_router)