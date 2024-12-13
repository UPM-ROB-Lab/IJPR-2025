import cv2
import matplotlib.pyplot as plt
import torch
import numpy as np
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import sys
sys.path.append("..")
from segment_anything import sam_model_registry, SamPredictor
import os

'''Prompt'''
# Define the parsing function
def parse_prompt(prompt):
    # Split the string by semicolon
    parts = prompt.split('; ')
    
    # Ensure there are enough parts
    if len(parts) != 3:
        raise ValueError("Prompt format is incorrect, it should contain three parts")

    # Parse the Customer Need part
    customer_need = parts[0].strip()
    need_parts = customer_need[1:-1].split(', ')
    need_dict = {
        "Modified Position": need_parts[0].strip(),
        "Score": need_parts[1].strip()
    }

    # Parse the Functional Requirement part
    functional_requirement = parts[1].strip()

    # Parse the Design Evaluation and Market Trends part
    evaluation_and_trends = parts[2].strip().split(' + ')
    evaluation_parts = evaluation_and_trends[0][1:-1].split(', ')
    trends_parts = evaluation_and_trends[1][1:-1].split(', ')
    evaluation_dict = {
        "Modify automotive interior serial number": evaluation_parts[0].strip(),
        "Score": evaluation_parts[1].strip(),
        "Modify requirements": [trend.strip() for trend in trends_parts]
    }

    # Create the final result dictionary
    result = {
        "Customer Need": need_dict,
        "Functional Requirement": functional_requirement,
        "Design Evaluation and Market Trends": evaluation_dict
    }

    return result

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    

# Read the prompt content from the txt file
with open('prompt.txt', 'r') as file:
    prompt_text = file.read().strip()

# Parse the prompt
parsed_result = parse_prompt(prompt_text)
# Output the result
print(parsed_result)


scheme_name= parsed_result["Design Evaluation and Market Trends"]["Modify automotive interior serial number"]
print(scheme_name)
parsed_result = parse_prompt(prompt_text)
requirements = parsed_result["Design Evaluation and Market Trends"]["Modify requirements"]
modified_positions = parsed_result["Customer Need"]["Modified Positions"]

if requirements:
    prompt_extraction = ", ".join(requirements)  # Use the comma-separated nouns as prompt
else:
    prompt_extraction = "" # If nothing is extracted, set the prompt as an empty string to avoid errors
print(prompt_extraction) # Output: Ergonomics, Aesthetics, Automatic transmission

'''SAM'''
# Build the image path
image_folder = 'D:\\CIRP2025\\experiment\\scheme'
image_path = os.path.join(image_folder, f"{scheme_name}.jpg") 

image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

sam_checkpoint = "D:\\CIRP2025\\segment-anything\\packages\\sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

mask_generator = SamAutomaticMaskGenerator(sam)
masks = mask_generator.generate(image)
# Output the number of generated masks
num_masks = len(masks)
print(f"Number of masks generated: {num_masks}")

# Display the image and masks
plt.figure(figsize=(10, 10))
plt.imshow(image)

# Loop through all generated masks and overlay them
for mask_data in masks:
    mask = mask_data['segmentation']  # Get the actual mask
    color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0) # Generate random color with transparency
    h, w = mask.shape
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    plt.imshow(mask_image)

plt.axis('off')  # Turn off the axis
plt.show()


# Another clearer way to visualize masks (display bounding boxes and scores of each mask)
def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = img[m] * (1 - 0.35) + color_mask * 0.35
    ax.imshow(img)

plt.figure(figsize=(10,10))
plt.imshow(image)
show_anns(masks)
plt.axis('off')
plt.show()


predictor = SamPredictor(sam)
predictor.set_image(image)
input_boxes_data = {
    "The control button": [0, 0, 268, 923],
    "The seat": [0, 780, 1350, 920],
    "The steering wheel": [200, 300, 580, 630],
    "The centrol console": [580, 450, 840, 650],
    "The gearshift lever": [600, 630, 820, 920],
    "The control button (right)": [1150, 0, 1388, 923],
}

# Create torch.tensor
input_boxes = torch.tensor(list(input_boxes_data.values()), device=predictor.device)

transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, image.shape[:2])
masks, _, _ = predictor.predict_torch(
    point_coords=None,
    point_labels=None,
    boxes=transformed_boxes,
    multimask_output=False,
)

print(masks.shape)  # (batch_size) x (num_predicted_masks_per_input) x H x W | output: torch.Size([4, 1, 600, 900])

plt.figure(figsize=(10, 10))
plt.imshow(image)
for mask in masks:
    show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
for box in input_boxes:
    show_box(box.cpu().numpy(), plt.gca())
plt.axis('off')
plt.show()

'''DM'''
import torch
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image

# Check if CUDA device is available
device = "cuda" if torch.cuda.is_available() else "cpu"

# Use SD 1.5 model
model_id = "D:\\CIRP2025\\stable-diffusion\\model\\stable-diffusion-v1-5"

# Load the pipeline, set torch_dtype based on the device type
try:
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, torch_dtype=torch.float16 if device == "cuda" else torch.float32, use_auth_token=True)
except Exception as e:
    print(f"Error loading the model: {e}")
    exit()

# Move the pipeline to the specified device
pipe = pipe.to(device)

image2_folder = 'D:\\CIRP2025\\experiment\\scheme'
image2_path = os.path.join(image_folder, f"{modified_positions}.jpg") 
# Load customer need modified position image and resize
try:
    init_image = Image.open(image2_path).convert("RGB").resize((600,450))
except FileNotFoundError:
    print("The specified image file cannot be found. Please check the path.")
    exit()
except Exception as e:
    print(f"Error opening the image file: {e}")
    exit()

# Set the prompt, strength, and other parameters
prompt = prompt_extraction
strength = 0.7
guidance_scale = 7.5
num_inference_steps = 50
generator = torch.Generator(device=device).manual_seed(1024)  # Set random seed for reproducibility

# Generate the image with the generator
try:
    images = pipe(prompt=prompt, init_image=init_image, strength=strength, guidance_scale=guidance_scale, num_inference_steps=num_inference_steps, generator=generator).images
except Exception as e:
    print(f"Error generating the image: {e}")
    exit()

# Save the generated image
try:
    images[0].save("D:\\CIRP2025\\experiment\\scheme\\generated_image.jpg")
    print("Image has been saved.")
except Exception as e:
    print(f"Error saving the image: {e}")
