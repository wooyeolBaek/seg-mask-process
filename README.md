# Post Process for Segmentation Masks
## Usage
Remove pixel areas below the specific threshold using Breadth First Search

## Initialize
```shell
git clone 
cd seg-mask-process
python -m venv .seg_venv
mkdir masks
```
Put mask images inside the `masks` directory

## (Optional)Normalize Mask
- Visualizing mask images
```shell
python norm.py
```

## Process
```shell
python process.py
```

## (Optional)CLIP Seg
- Quick Start: Segmentation Mask
```python
from PIL import Image
import requests
from transformers import AutoProcessor, CLIPSegModel

processor = AutoProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
model = CLIPSegModel.from_pretrained("CIDAS/clipseg-rd64-refined")

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(
    text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True
)

outputs = model(**inputs)
logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
```

## References
- https://github.com/switchablenorms/CelebAMask-HQ
- https://huggingface.co/docs/transformers/model_doc/clipseg
- https://huggingface.co/CIDAS/clipseg-rd64-refined