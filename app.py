import torch
from torchvision import transforms
from PIL import Image
import gradio as gr

from src.models import get_model
from src.utils import device


dev = device()
print("Device used:", dev)


MODEL_INFOS = {
    "resnet50": {
        "path": "model_resnet50.pth",
        "label": "ResNet-50 (Transfer Learning)"
    },
    "efficientnet_b0": {
        "path": "model_efficientnet_b0.pth",
        "label": "EfficientNet-B0"
    },
    "cnn_custom": {
        "path": "model_cnn_custom.pth",
        "label": "Custom CNN"
    },
}


THRESHOLDS = {
    "cnn_custom": 0.531,
    "efficientnet_b0": 0.656,
    "resnet50": 0.487,
}


MODELS = {}

for name, info in MODEL_INFOS.items():
    try:
        print(f"Loading model '{name}' from {info['path']} ...")
        model = get_model(name, num_classes=2, pretrained=False).to(dev)
        state = torch.load(info["path"], map_location=dev)
        model.load_state_dict(state)
        model.eval()
        MODELS[name] = model
        print("  -> Loaded successfully")
    except Exception as e:
        print(f"  !! Failed to load {name}: {e}")


IMG_SIZE = 128

val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

CLASSES = ["benign (non-cancer)", "malignant (cancer)"]

def predict(image: Image.Image, model_name: str):


    if image is None:
        return "Please upload an image first."

    if model_name not in MODELS:
        return f"Model '{model_name}' is not available."

    model = MODELS[model_name]
    threshold = THRESHOLDS.get(model_name, 0.5)

    x = val_tf(image).unsqueeze(0).to(dev)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

    p_benign = float(probs[0])
    p_malignant = float(probs[1])

    has_cancer = p_malignant >= threshold

    if has_cancer:
        diagnosis = "⚠️ Suspicion of SKIN CANCER (malignant)"
    else:
        diagnosis = "✅ Likely benign lesion (non-cancer)"

    model_label = MODEL_INFOS[model_name]["label"]

    text = (
        f"Model used: {model_label}\n"
        f"{diagnosis}\n\n"
        f"Benign probability :  {p_benign:.3f}\n"
        f"Malignant probability : {p_malignant:.3f}\n"
        f"Threshold for 'cancer' : {threshold:.3f}\n\n"
        f"⚠️ This tool is an assistant and does NOT replace a dermatologist."
    )

    return text


model_dropdown = gr.Dropdown(
    choices=[
        ("ResNet-50 (Transfer Learning)", "resnet50"),
        ("EfficientNet-B0", "efficientnet_b0"),
        ("Custom CNN", "cnn_custom"),
    ],
    value="resnet50",
    label="Choose model",
)

iface = gr.Interface(
    fn=predict,
    inputs=[
        gr.Image(type="pil", label="Skin lesion image"),
        model_dropdown,
    ],
    outputs=gr.Textbox(label="Model result"),
    title="Skin Cancer Detection — Diagnostic Assistant",
    description=(
        "Upload a skin lesion image and choose a model "

    ),
)


if __name__ == "__main__":
    iface.launch()
