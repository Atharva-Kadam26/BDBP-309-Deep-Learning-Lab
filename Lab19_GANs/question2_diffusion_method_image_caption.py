
import os
import torch
from PIL import Image

from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    CLIPProcessor,
    CLIPModel
)

from diffusers import StableDiffusionPipeline


# --------------------------------------------------
# 1. Device configuration
# --------------------------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Device:", device)


# --------------------------------------------------
# 2. Load pretrained BLIP model
# --------------------------------------------------

print("\nLoading BLIP model...")

blip_processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)

blip_model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
).to(device)

print("BLIP model loaded!")


# --------------------------------------------------
# 3. Load Stable Diffusion model
# --------------------------------------------------

print("\nLoading Stable Diffusion model...")

sd_model = "stable-diffusion-v1-5/stable-diffusion-v1-5"

pipe = StableDiffusionPipeline.from_pretrained(
    sd_model,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    safety_checker=None
)

pipe = pipe.to(device)

# Reduce memory usage
pipe.enable_attention_slicing()

print("Stable Diffusion loaded!")


# --------------------------------------------------
# 4. Load CLIP model for Approach 2
# --------------------------------------------------

print("\nLoading CLIP model...")

clip_processor = CLIPProcessor.from_pretrained(
    "openai/clip-vit-base-patch32"
)

clip_model = CLIPModel.from_pretrained(
    "openai/clip-vit-base-patch32"
).to(device)

print("CLIP model loaded!")


# --------------------------------------------------
# 5. Select input image
# --------------------------------------------------

image_folder = "images"

image_files = [
    f for f in os.listdir(image_folder)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

if len(image_files) == 0:
    raise FileNotFoundError("No images found in images folder")

image_path = os.path.join(image_folder, image_files[0])

original_image = Image.open(image_path).convert("RGB")

print("\nInput image:", image_path)


# --------------------------------------------------
# APPROACH 1:
# BLIP + Stable Diffusion
# --------------------------------------------------

print("\n========== APPROACH 1 ==========")


def generate_caption(image):

    inputs = blip_processor(
        image,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():

        output = blip_model.generate(
            **inputs,
            max_new_tokens=50
        )

    caption = blip_processor.decode(
        output[0],
        skip_special_tokens=True
    )

    return caption


caption = generate_caption(original_image)

print("\nGenerated Caption:")
print(caption)


# Generate an image from the caption
print("\nGenerating image using Stable Diffusion...")

generated_image = pipe(
    prompt=caption,
    num_inference_steps=10,
    guidance_scale=7.5,
    height=256,
    width=256
).images[0]

generated_image.save("approach1_generated.png")

print("Generated image saved as approach1_generated.png")


# --------------------------------------------------
# APPROACH 2:
# Stable Diffusion-guided caption selection
# --------------------------------------------------
#
# print("\n========== APPROACH 2 ==========")
#
#
# def generate_candidate_captions(image):
#
#     inputs = blip_processor(
#         image,
#         return_tensors="pt"
#     ).to(device)
#
#     with torch.no_grad():
#
#         outputs = blip_model.generate(
#             **inputs,
#             max_new_tokens=50,
#             num_beams=5,
#             num_return_sequences=2,
#             do_sample=True,
#             top_p=0.9
#         )
#
#     captions = []
#
#     for output in outputs:
#
#         text = blip_processor.decode(
#             output,
#             skip_special_tokens=True
#         )
#
#         if text not in captions:
#             captions.append(text)
#
#     return captions
#
#
# # Generate multiple captions
# candidate_captions = generate_candidate_captions(
#     original_image
# )
#
# print("\nCandidate captions:")
#
# for i, text in enumerate(candidate_captions):
#     print(i + 1, ":", text)
#
#
# # Generate an image for each candidate caption
# generated_images = []
#
# for i, text in enumerate(candidate_captions):
#
#     print("\nGenerating image for candidate", i + 1)
#
#     result = pipe(
#         prompt=text,
#         num_inference_steps=10,
#         guidance_scale=7.5,
#         height=256,
#         width=256
#     )
#
#     generated = result.images[0]
#
#     generated.save(
#         "candidate_" + str(i + 1) + ".png"
#     )
#
#     generated_images.append(generated)
#
#
# # --------------------------------------------------
# # 6. Compare images using CLIP
# # --------------------------------------------------
#
# print("\nCalculating CLIP similarity scores...")
#
#
# def calculate_similarity(original, generated):
#
#     inputs = clip_processor(
#         text=["An image"],
#         images=[original, generated],
#         return_tensors="pt",
#         padding=True
#     ).to(device)
#
#     with torch.no_grad():
#
#         image_features = clip_model.get_image_features(
#             pixel_values=inputs["pixel_values"]
#         )
#
#         image_features = image_features / image_features.norm(
#             dim=-1, keepdim=True
#         )
#
#         similarity = torch.sum(
#             image_features[0] * image_features[1]
#         )
#
#     return similarity.item()
#
#
# scores = []
#
# for i, generated in enumerate(generated_images):
#
#     score = calculate_similarity(
#         original_image,
#         generated
#     )
#
#     scores.append(score)
#
#     print(
#         "Caption:", candidate_captions[i],
#         "| CLIP similarity:", round(score, 4)
#     )
#
#
# # Select caption with highest similarity
# best_index = scores.index(max(scores))
#
# best_caption = candidate_captions[best_index]
#
# print("\nBest matching caption:")
# print(best_caption)
#
# print("\nBest CLIP similarity:", round(scores[best_index], 4))
#
#
# # --------------------------------------------------
# # 7. Save results
# # --------------------------------------------------
#
# with open("question2_results.txt", "w") as file:
#
#     file.write("APPROACH 1\n")
#     file.write("Caption: " + caption + "\n\n")
#
#     file.write("APPROACH 2\n")
#
#     for i, text in enumerate(candidate_captions):
#
#         file.write(
#             "Candidate " + str(i + 1) + ": " + text + "\n"
#         )
#
#         file.write(
#             "CLIP Score: " + str(scores[i]) + "\n"
#         )
#
#     file.write("\nBest Caption: " + best_caption)
#
#
# print("\nAll results saved successfully!")
