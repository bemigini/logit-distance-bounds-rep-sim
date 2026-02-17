"""
Get and save the DINO embeddings for SUB dataset
"""


import os
import torch
from datasets import load_dataset
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
from tqdm import tqdm


def get_and_save_SUB_dino_embs():
    # ======================
    # Configuration
    # ======================
    DATASET_NAME = "Jessica-bader/SUB"          # Hugging Face dataset name
    SPLIT = "test"               # change if needed
    IMAGE_COLUMN = "image"        # change if different
    MODEL_NAME = "facebook/dinov2-base"
    IMAGE_SIZE = 512
    BATCH_SIZE = 16               # adjust based on GPU memory
    OUTPUT_FILE = "data/SUB_dinov2_embeddings.pt"

    # ======================
    # Device
    # ======================
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # ======================
    # Load dataset
    # ======================
    dataset = load_dataset(DATASET_NAME)
    # print(dataset['test'])
    # quit()

    # ======================
    # Load DINOv2
    # ======================
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.to(device)
    model.eval()

    # ======================
    # Helper functions
    # ======================
    def preprocess_images(images):
        processed = []
        for img in images:
            if img.mode != "RGB":
                img = img.convert("RGB")
            img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
            processed.append(img)
        return processed


    def extract_batch_embeddings(images):
        inputs = processor(images=images, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        # CLS token embeddings
        embeddings = outputs.last_hidden_state[:, 0, :]
        return embeddings.cpu()


    # ======================
    # Main loop
    # ======================
    all_embeddings = []
    all_labels = []
    all_attributes = []

    # collect 10 images per class and save in a folder in data
    for i in tqdm(range(0, len(dataset['test']), BATCH_SIZE)):
        batch = dataset['test'][i : i + BATCH_SIZE]
        img, label, att = batch['image'], batch['bird_label'], batch['attr_label']
        # print(len(img), len(label), len(att))
        # print(label)
        # quit()
        images = preprocess_images(img)
        batch_embeddings = extract_batch_embeddings(images)
        all_embeddings.append(batch_embeddings)
        all_labels.extend(label)
        all_attributes.extend(att)

        # save image if it is among the first 10 of its class
        for j in range(len(images)):
            class_dir = os.path.join("data/SUB_images", str(att[j]))
            os.makedirs(class_dir, exist_ok=True)
            existing_images = os.listdir(class_dir)
            if len(existing_images) < 100:
                img_path = os.path.join(class_dir, f"{len(existing_images)}.png")
                images[j].save(img_path)
            print(f"Saved images to data/SUB_images/")

    #quit()
    # ======================
    # Save as single tensor
    # ======================
    all_embeddings = torch.cat(all_embeddings, dim=0)
    all_labels = torch.tensor(all_labels)
    all_attributes = torch.tensor(all_attributes)

    print("Final embedding shape:", all_embeddings.shape)
    torch.save(
        {
            "embeddings": all_embeddings,
            "labels": all_labels,
            "attributes": all_attributes,
            "model": MODEL_NAME,
            "image_size": IMAGE_SIZE,
            "dataset": DATASET_NAME,
            "split": SPLIT,
        },
        OUTPUT_FILE,
    )

    print(f"Saved embeddings to: {OUTPUT_FILE}")
