import os
from pathlib import Path
from PIL import Image, UnidentifiedImageError

def clean_images_in_path(path: Path):
    print(f"\n🧹 Limpando imagens na pasta: {path}")
    total = 0
    removidos = 0
    for img_path in path.rglob("*"):
        if img_path.is_file():
            total += 1
            if img_path.name.startswith("._"):
                print(f"⛔ Arquivo oculto removido: {img_path}")
                img_path.unlink()
                removidos += 1
                continue
            try:
                with Image.open(img_path) as img:
                    img.verify()
            except (UnidentifiedImageError, OSError) as e:
                print(f"❌ Imagem inválida removida: {img_path} ({e})")
                img_path.unlink()
                removidos += 1
    print(f"\n✅ Verificação concluída: {total} arquivos verificados, {removidos} removidos.\n")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Remove arquivos inválidos de imagem.")
    parser.add_argument("dataset_folder", type=str, help="datasets/cars")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_folder)
    if not dataset_dir.exists():
        print(f"❌ Caminho não encontrado: {dataset_dir}")
    else:
        clean_images_in_path(dataset_dir)
