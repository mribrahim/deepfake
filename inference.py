import os
import sys
import json
import argparse
import importlib.util
from pathlib import Path
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import timm
from tabulate import tabulate

# --- THE PURE DOWNSTREAM LAYER CLASSIFIER ---
class LinearProbe(nn.Module):
    """Simple linear classifier matching training checkpoint topology"""
    def __init__(self, input_dim, num_classes=1):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)
        
    def forward(self, x):
        # Expects x of shape [batch_size, input_dim]
        return torch.sigmoid(self.fc(x))

# --- ISOLATED FEATURE EXTRACTORS ---
class FeatureExtractor:
    def __init__(self, backbone_name, device):
        self.backbone_name = backbone_name
        self.device = device
        
        if backbone_name == "radio":
            print("\n" + "="*60)
            print("LOADING NVIDIA C-RADIOv4-H BACKBONE VIA VIRTUAL PACKAGE")
            print("="*60) 
            os.environ["HF_TRUST_REMOTE_CODE"] = "True"
            model_folder_path = "./checkpoint_radiov4"
            abs_path = Path(model_folder_path).resolve()
            
            if str(abs_path.parent) not in sys.path:
                sys.path.insert(0, str(abs_path.parent))

            package_name = "checkpoint_radiov4"
            init_spec = importlib.util.spec_from_file_location(package_name, str(abs_path / "hf_model.py"))
            package_module = importlib.util.module_from_spec(init_spec)
            package_module.__path__ = [str(abs_path)]  
            sys.modules[package_name] = package_module

            hf_model_spec = importlib.util.spec_from_file_location(f"{package_name}.hf_model", str(abs_path / "hf_model.py"))
            hf_model_module = importlib.util.module_from_spec(hf_model_spec)
            sys.modules[f"{package_name}.hf_model"] = hf_model_module
            hf_model_spec.loader.exec_module(hf_model_module)

            RADIOConfig = hf_model_module.RADIOConfig
            RADIOModel = hf_model_module.RADIOModel

            with open(abs_path / "config.json", "r", encoding="utf-8") as f:
                config_dict = json.load(f)
            if "auto_map" in config_dict:
                del config_dict["auto_map"]

            config = RADIOConfig.from_dict(config_dict)
            self.model = RADIOModel.from_pretrained(model_folder_path, config=config, local_files_only=True).to(self.device)
            
        elif backbone_name == "dinov3":
            print("[*] Loading Meta DINOv3 Backbone...")
            self.model = timm.create_model("vit_large_patch16_dinov3.lvd1689m", pretrained=True, num_classes=0).to(self.device)
        else: 
            print("[*] Loading Supervised RoPE-ViT Backbone...")
            self.model = timm.create_model("vit_base_patch16_rope_mixed_ape_224.naver_in1k", pretrained=True, num_classes=0).to(self.device)
            
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def extract(self, x):
        with torch.no_grad():
            if self.backbone_name == "radio":
                # Standard summary outputs for RADIOv4 return a class token and patch feature tuple
                out = self.model(x)[1]
            else:
                out = self.model.forward_features(x)
                
            # Safely average across spatial token dimension [Batch, Tokens, Features]
            features = out.mean(dim=1) if len(out.shape) == 3 else out.mean(dim=[2, 3])
            return features
            
def get_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def process_image(img_path, extractors, probes, device, transform):
    try:
        img = Image.open(img_path).convert('RGB')
        tensor = transform(img).unsqueeze(0).to(device)
    except Exception as e:
        print(f"[-] Error loading image {img_path}: {e}")
        return None

    results = {}
    for name, extractor in extractors.items():
        probe = probes[name]
        with torch.no_grad():
            feats = extractor.extract(tensor)
            prob = probe(feats).item()
        
        # CORRECTED: Output >= 0.5 indicates target label 1 (REAL)
        pred_label = "REAL" if prob >= 0.5 else "FAKE"
        results[name] = f"{pred_label} (Score: {prob:.4f})"
    return results

def eval_folder_mode(target_dir, extractors, probes, device, transform):
    print(f"[*] Initializing evaluation dashboard on structural directories...")
    metrics = {name: {"TP": 0, "FP": 0, "TN": 0, "FN": 0} for name in extractors.keys()}
    
    # Target Map Configuration: Fake = 0, Real = 1
    classes = {"fake": 0, "real": 1}
    
    for class_name, true_label in classes.items():
        subfolder = os.path.join(target_dir, class_name)
        if not os.path.exists(subfolder):
            continue
            
        img_files = [os.path.join(subfolder, f) for f in os.listdir(subfolder) 
                     if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
        
        print(f"[*] Parsing class folder '{class_name}' ({len(img_files)} images found)...")
        for img_path in img_files:
            try:
                img = Image.open(img_path).convert('RGB')
                tensor = transform(img).unsqueeze(0).to(device)
            except:
                continue
            
            for name, extractor in extractors.items():
                probe = probes[name]
                with torch.no_grad():
                    feats = extractor.extract(tensor)
                    prob = probe(feats).item()
                
                pred_label = 1 if prob >= 0.5 else 0
                
                # CORRECTED: Standard Matrix Mapping considering Class 1 as Positive (REAL)
                if true_label == 1 and pred_label == 1:       metrics[name]["TP"] += 1 # True Real
                elif true_label == 0 and pred_label == 1:     metrics[name]["FP"] += 1 # False Real (Fake evaluated as Real)
                elif true_label == 0 and pred_label == 0:     metrics[name]["TN"] += 1 # True Fake
                elif true_label == 1 and pred_label == 0:     metrics[name]["FN"] += 1 # False Fake (Real evaluated as Fake)

    summary_data = []
    for name, m in metrics.items():
        total = m["TP"] + m["FP"] + m["TN"] + m["FN"]
        if total == 0: continue
        acc = (m["TP"] + m["TN"]) / total
        p = m["TP"] / (m["TP"] + m["FP"]) if (m["TP"] + m["FP"]) > 0 else 0
        r = m["TP"] / (m["TP"] + m["FN"]) if (m["TP"] + m["FN"]) > 0 else 0
        f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0
        summary_data.append([name, f"{acc:.4f}", f"{p:.4f}", f"{r:.4f}", f"{f1:.4f}"])
        
    print("\n[+] Structured Dataset Processing Complete (Positive Class: REAL):")
    print(tabulate(summary_data, headers=["Model LP", "Accuracy", "Precision", "Recall", "F1-Score"], tablefmt="grid"))

def main():
    parser = argparse.ArgumentParser(description="Inference Pipeline for VFM Deepfake Linear Probes")
    parser.add_argument("--input", type=str, required=True, help="Path to image file OR directory with real/fake folders")
    parser.add_argument("--weights_vit", type=str, default="model-vit.pth")
    parser.add_argument("--weights_dino", type=str, default="model-dino.pth")
    parser.add_argument("--weights_radio", type=str, default="model-radiov4.pth")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = get_transforms()

    configs = {
        "Supervised_ViT_LP": {"backbone": "vit", "dim": 768, "path": args.weights_vit},
        "DINOv3_LP": {"backbone": "dinov3", "dim": 1024, "path": args.weights_dino},
        "RADIOv4_LP": {"backbone": "radio", "dim": 1280, "path": args.weights_radio}
    }

    extractors = {}
    probes = {}
    
    for name, cfg in configs.items():
        if os.path.exists(cfg["path"]):
            # 1. Load Extractor
            extractors[name] = FeatureExtractor(cfg["backbone"], device)
            
            # 2. Instantiate and clean LinearProbe directly
            probe = LinearProbe(input_dim=cfg["dim"])
            state_dict = torch.load(cfg["path"], map_location=device, weights_only=True)
            
            cleaned_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("linear_probe."):
                    cleaned_state_dict[k.replace("linear_probe.", "fc.")] = v
                elif k.startswith("fc."):
                    cleaned_state_dict[k] = v
                else:
                    cleaned_state_dict[k] = v
                    
            probe.load_state_dict(cleaned_state_dict, strict=False)
            probe.to(device).eval()
            probes[name] = probe
            print(f"[+] Successfully mapped and isolated weights for: {name}")

    if not extractors:
        print("[-] Error: No checkpoint configurations initialized.")
        return

    if os.path.isdir(args.input):
        eval_folder_mode(args.input, extractors, probes, device, transform)
    else:
        res = process_image(args.input, extractors, probes, device, transform)
        if res:
            print(f"\n[+] Single File Classification Vector Output for: {os.path.basename(args.input)}")
            print(tabulate(res.items(), headers=["Evaluated Model", "Prediction Output (Fake=0, Real=1)"], tablefmt="presto"))

if __name__ == "__main__":
    main()