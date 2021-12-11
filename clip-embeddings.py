import os
from PIL import Image
import csv, shutil, torch, pandas as pd, numpy as np
import clip

#import dataroot?

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

clip.available_models()

model, preprocess = clip.load("ViT-B/32")
model.to(device).eval()
# Get working directory
PATH = os.getcwd()

base_path = DATAROOT
images = []
original_images = []

# Get images
# Check os.listdir(base_path)
all_files = os.listdir(base_path)

for file in all_files:
    file_path = os.path.join(base_path, file)
    image = Image.open(file_path).convert("RGB")
    original_images.append(image)
    images.append(preprocess(image))
    print("#", end=" ")

image_input = torch.tensor(np.stack(images)).cuda()

# Replace this with a variable - DOMAIN?
domain_names = ['infograph', 'quickdraw', 'real', 'clipart', 'quickdraw', 'sketch']


def get_clip_domain_embed(domain_names):
    """
    This function computed the embeddings from a given domain description
    :return: pandas dataframe with domains embeddings of each description extracted from clip.
             Each row is one embedding
    """
    domain_embed = {}
    # Text Embedding
    # text_inputs = torch.cat([clip.tokenize(f"This is a photo of a {c}") for c in domain_names]).to(device)
    text_inputs = torch.cat([clip.tokenize(f"This is a {c} data") for c in domain_names]).to(device)

    # Calculate features
    with torch.no_grad():
        image_features = model.encode_image(image_input).float()
        text_features = model.encode_text(text_inputs).float()

    # Pick the top (1) most similar labels for the image
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    top_probs, top_labels = text_probs.cpu().topk(1, dim=-1)

    #---

    # Print the result
    pred_domain = []
    for i, image in enumerate(original_images):
        pred = [domain_names[index] for index in top_labels[i].numpy()][0]
        pred_domain.append(pred)

    pred_df = pd.DataFrame(pred_domain)
    pred_df.to_csv('pred/pred_domain.csv', index=False, header=False)
    return pred_df
