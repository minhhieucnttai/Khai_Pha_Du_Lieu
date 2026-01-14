# main.py
# Script chinh de chay chuong trinh
# Tac gia: Minh Hieu

import os
import sys
import argparse

# Them thu muc hien tai vao path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from preprocessing import CLASSES, CATEGORY_MAP, get_data_paths, create_data_generators, count_images
from preprocessing import get_csv_paths, get_csv_distribution


def show_info():
    """Hien thi thong tin he thong"""
    print("="*50)
    print("PHAN LOAI BENH LA DAU (Bean Leaf Disease)")
    print("="*50)
    
    print("\nCac loai benh:")
    print("  1. Healthy - La khoe manh")
    print("  2. Angular Leaf Spot - Dom la goc")
    print("  3. Bean Rust - Gi sat dau")
    
    print("\nMapping tu CSV:")
    for k, v in CATEGORY_MAP.items():
        print(f"  {k} -> {v}")
    
    # Thong tin du lieu
    train_path, val_path = get_data_paths()
    if os.path.exists(train_path):
        train_counts = count_images(train_path)
        val_counts = count_images(val_path)
        
        print("\nDu lieu tu folder:")
        print(f"  Training: {sum(train_counts.values())} anh")
        print(f"  Validation: {sum(val_counts.values())} anh")
    
    # CSV
    train_csv, val_csv = get_csv_paths()
    if os.path.exists(train_csv):
        csv_train = get_csv_distribution(train_csv)
        csv_val = get_csv_distribution(val_csv)
        print("\nDu lieu tu CSV:")
        print(f"  train.csv: {sum(csv_train.values())} anh")
        print(f"  val.csv: {sum(csv_val.values())} anh")
    
    print("\n" + "="*50)


def run_eda():
    """Chay EDA"""
    from eda import run_eda as eda_run
    train_path, val_path = get_data_paths()
    eda_run(train_path, val_path)


def run_train(model_type='cnn', epochs=30):
    """Huan luyen mo hinh"""
    from model_minhhieu import build_cnn_model, build_mobilenet_model, get_callbacks, train_model, save_model
    from evaluation import full_evaluation
    
    print("="*50)
    print("HUAN LUYEN MO HINH")
    print("="*50)
    
    # Load data
    train_path, val_path = get_data_paths()
    train_gen, val_gen = create_data_generators(train_path, val_path)
    
    print(f"\nTraining: {train_gen.samples} anh")
    print(f"Validation: {val_gen.samples} anh")
    
    # Xay dung model
    if model_type == 'mobilenet':
        print("\nDang xay dung MobileNetV2...")
        model = build_mobilenet_model()
    else:
        print("\nDang xay dung CNN...")
        model = build_cnn_model()
    
    model.summary()
    
    # Train
    print("\nBat dau training...")
    callbacks = get_callbacks(model_type)
    history = train_model(model, train_gen, val_gen, epochs, callbacks)
    
    # Luu model
    save_model(model, f'../models/{model_type}_final.keras')
    
    # Danh gia
    print("\nDanh gia mo hinh...")
    full_evaluation(model, val_gen, history)
    
    print("\nHoan thanh!")


def run_predict(model_path, image_path):
    """Du doan 1 anh"""
    from model_minhhieu import load_model, predict_image
    
    print("="*50)
    print("DU DOAN ANH")
    print("="*50)
    
    print(f"\nModel: {model_path}")
    print(f"Anh: {image_path}")
    
    model = load_model(model_path)
    cls, conf = predict_image(model, image_path)
    
    print(f"\nKet qua:")
    print(f"  Loai: {cls}")
    print(f"  Do tin cay: {conf:.2%}")
    
    # Mo ta tieng Viet
    desc = {
        'healthy': 'La khoe manh',
        'angular_leaf_spot': 'Benh dom la goc',
        'bean_rust': 'Benh gi sat'
    }
    print(f"  Mo ta: {desc.get(cls, cls)}")


def main():
    parser = argparse.ArgumentParser(description='Phan loai benh la dau')
    
    subparsers = parser.add_subparsers(dest='cmd', help='Cac lenh')
    
    # info
    subparsers.add_parser('info', help='Hien thi thong tin')
    
    # eda
    subparsers.add_parser('eda', help='Phan tich du lieu')
    
    # train
    train_p = subparsers.add_parser('train', help='Huan luyen mo hinh')
    train_p.add_argument('--model', default='cnn', choices=['cnn', 'mobilenet'])
    train_p.add_argument('--epochs', type=int, default=30)
    
    # predict
    pred_p = subparsers.add_parser('predict', help='Du doan anh')
    pred_p.add_argument('model', help='Duong dan model')
    pred_p.add_argument('image', help='Duong dan anh')
    
    args = parser.parse_args()
    
    if args.cmd == 'info':
        show_info()
    elif args.cmd == 'eda':
        run_eda()
    elif args.cmd == 'train':
        run_train(args.model, args.epochs)
    elif args.cmd == 'predict':
        run_predict(args.model, args.image)
    else:
        show_info()


if __name__ == '__main__':
    main()
