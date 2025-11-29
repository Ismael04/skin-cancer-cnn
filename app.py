import torch
from torchvision import transforms
from PIL import Image
import gradio as gr

from src.models import get_model
from src.utils import device

dev = device()
print("Device utilisé :", dev)

MODEL_NAME = "resnet50"
MODEL_PATH = "model_resnet50.pth"

model = get_model(MODEL_NAME, num_classes=2, pretrained=False).to(dev)
model.load_state_dict(torch.load(MODEL_PATH, map_location=dev))
model.eval()

IMG_SIZE = 128

val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

CLASSES = ["benign (non-cancer)", "malignant (cancer)"]

THRESHOLD = 0.48


def predict(image: Image.Image):

    x = val_tf(image).unsqueeze(0).to(dev)   # (1, 3, H, W)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

    p_benign = float(probs[0])
    p_malignant = float(probs[1])

    has_cancer = p_malignant >= THRESHOLD

    if has_cancer:
        diagnosis = "⚠️ Suspicion of skin cancer"
    else:
        diagnosis = "✅ Probably benign"

    # 4) Texte lisible pour le médecin
    text = (
        f"{diagnosis}\n\n"
        f"benign probs :  {p_benign:.3f}\n"
        f"malignant probs : {p_malignant:.3f}\n"
        f"Threshold use for 'cancer' : {THRESHOLD:.2f}\n\n"
        f"⚠️ Attention : This model is diagnostic aid tool, "
        f" This does not replace a dermatologist opinion."
    )

    return text



iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Image"),
    outputs=gr.Textbox(label="Model result"),
    title="Skin Cancer Detection - Diagnostic",
    description=(
        "Put your image"
    )
)

if __name__ == "__main__":
    iface.launch()
