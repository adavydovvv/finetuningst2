# experiments/export_onnx.py
import torch
import timm
import argparse
import os
import onnx

def export(weights_path, out_path, model_name="resnet18", image_size=224, num_classes=3):
    device = torch.device("cpu")

    # Создаём модель и загружаем веса
    model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    # ДUMMY вход для ONNX
    dummy = torch.randn(1, 3, image_size, image_size, device=device)

    # Экспортируем модель в ONNX opset 18
    torch.onnx.export(
        model,
        dummy,
        out_path,
        opset_version=18,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
        verbose=False
    )

    # Проверка ONNX
    onnx_model = onnx.load(out_path)
    onnx.checker.check_model(onnx_model)
    print(f"✅ Exported and checked ONNX model: {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True, help="Path to PyTorch weights (.pth)")
    parser.add_argument("--out", default="models/model.onnx", help="Output ONNX file path")
    parser.add_argument("--model_name", default="resnet18", help="Model architecture")
    parser.add_argument("--image_size", type=int, default=224, help="Input image size")
    parser.add_argument("--num_classes", type=int, default=3, help="Number of classes")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    export(args.weights, args.out, args.model_name, args.image_size, args.num_classes)
