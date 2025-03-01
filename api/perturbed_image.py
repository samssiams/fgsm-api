import os
import io
import uuid
import logging
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import numpy as np
from fastapi import APIRouter, HTTPException, File, UploadFile, Form
from fastapi.responses import JSONResponse
from supabase import create_client, Client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

# Supabase configuration (URL and key from environment variables)
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
BUCKET_NAME = "protecture"
FOLDER_PATH = "post-image"

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

model = models.resnet50(pretrained=True)
model.eval()
logger.info("Pre-trained model loaded and set to eval mode.")

def preprocess_image(image_data: bytes):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    image = Image.open(io.BytesIO(image_data)).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)
    logger.info("Image preprocessed successfully.")
    return image_tensor

def fgsm_attack(model, image, target_class, epsilon, iterations, alpha, threshold):
    perturbed_image = image.clone().detach()
    perturbed_image.requires_grad = True
    prev_confidence = 0
    for i in range(iterations):
        output = model(perturbed_image)
        probs = torch.nn.functional.softmax(output, dim=1)
        confidence = probs[0, target_class].item()
        logger.info(f"Iteration {i+1}: Confidence for target class: {confidence:.4f}")
        if confidence >= threshold:
            logger.info("Confidence threshold reached; stopping iterations.")
            break
        if confidence == 0.0 and prev_confidence == 0.0:
            logger.info("Confidence stuck at 0.0; returning current perturbed image.")
            return perturbed_image
        prev_confidence = confidence
        loss = nn.CrossEntropyLoss()(output, torch.tensor([target_class]))
        model.zero_grad()
        loss.backward()
        data_grad = perturbed_image.grad.data
        perturbed_image = perturbed_image + alpha * data_grad.sign()
        perturbed_image = torch.clamp(perturbed_image, image - epsilon, image + epsilon)
        perturbed_image = torch.clamp(perturbed_image, 0, 1)
        perturbed_image = perturbed_image.clone().detach().requires_grad_(True)
    return perturbed_image

def tensor_to_image(tensor):
    image_np = tensor.squeeze().detach().numpy()
    image_np = np.transpose(image_np, (1, 2, 0))
    image_np = (image_np * 255).astype(np.uint8)
    logger.info("Tensor converted back to PIL image.")
    return Image.fromarray(image_np)

def upload_file_to_supabase(file_bytes: bytes, original_file_name: str) -> str:
    # Generate a unique file name using UUID and preserve file extension.
    file_extension = os.path.splitext(original_file_name)[1] or ".jpg"
    unique_file_name = f"perturbed_{uuid.uuid4()}{file_extension}"
    path = f"{FOLDER_PATH}/{unique_file_name}"
    
    # Upload the file to Supabase.
    res = supabase.storage.from_(BUCKET_NAME).upload(path, file_bytes)
    # Assuming a successful HTTP request (200 OK) means the file is uploaded.
    public_url = supabase.storage.from_(BUCKET_NAME).get_public_url(path)
    logger.info("Supabase public URL: %s", public_url)
    return public_url

@router.post("/api/perturbed-image")
async def create_perturbed_image(
    file: UploadFile = File(...),
    perturbation_level: str = Form(...),
    description: str = Form(...),
    category_id: str = Form(...),
    user_id: int = Form(...),
    community_id: str = Form(None)
):
    logger.info("Received request for perturbed image post creation.")
    logger.info("Received field - perturbation_level: %s", perturbation_level)
    logger.info("Received field - description: %s", description)
    logger.info("Received field - category_id: %s", category_id)
    logger.info("Received field - user_id: %s", user_id)
    logger.info("Received field - community_id: %s", community_id)
    logger.info("Received file - filename: %s, content_type: %s", file.filename, file.content_type)
    
    try:
        image_data = await file.read()
        logger.info("File read successfully. Size: %d bytes", len(image_data))
    except Exception as e:
        logger.error("Error reading file: %s", e)
        raise HTTPException(status_code=400, detail="Error reading file")
    
    image_tensor = preprocess_image(image_data)
    level = perturbation_level.upper()
    if level == "LOW":
        epsilon = 0.20
        iterations = 5
        alpha = 0.06
        threshold = 0.50
    elif level == "MEDIUM":
        epsilon = 0.25
        iterations = 5
        alpha = 0.10
        threshold = 0.50
    elif level == "HIGH":
        epsilon = 0.30
        iterations = 5
        alpha = 0.15
        threshold = 0.50
    else:
        logger.error("Invalid perturbation level: %s", perturbation_level)
        raise HTTPException(status_code=400, detail="Invalid perturbation level. Choose LOW, MEDIUM, or HIGH.")
    
    target_class = 425
    logger.info("Starting FGSM attack with perturbation level: %s", level)
    perturbed_tensor = fgsm_attack(model, image_tensor, target_class, epsilon, iterations, alpha, threshold)
    logger.info("FGSM attack completed, converting tensor to image.")
    perturbed_img = tensor_to_image(perturbed_tensor)
    
    buf = io.BytesIO()
    perturbed_img.save(buf, format="PNG")
    buf.seek(0)
    file_bytes = buf.getvalue()
    
    try:
        public_url = upload_file_to_supabase(file_bytes, file.filename)
        logger.info("Uploaded perturbed image to Supabase. Image URL: %s", public_url)
    except Exception as e:
        logger.error("Error uploading to Supabase: %s", e)
        raise HTTPException(status_code=500, detail="Error uploading file to storage")
    
    # Return the perturbed image URL immediately.
    return JSONResponse(status_code=201, content={"image_url": public_url})
