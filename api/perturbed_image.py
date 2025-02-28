from fastapi import APIRouter

router = APIRouter()

@router.post("/api/perturbed-image")
async def perturbed_image():
    return {"message": "This is a test route"}