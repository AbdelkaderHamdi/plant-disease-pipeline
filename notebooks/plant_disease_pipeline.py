import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')

import json
import joblib
import argparse
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

import warnings
warnings.filterwarnings("ignore")

# Configuration des chemins de base
BASE_DIR = "D:\\projects\\plant-disease-pipeline"
DATA_DIR = "D:\\projects\\plant-disease-pipeline\\data\\sampled-plants-diseases"
MODELS_DIR = os.path.join(BASE_DIR, "models")
NOTEBOOKS_DIR = os.path.join(BASE_DIR, "notebooks")
DEPLOYMENT_DIR = os.path.join(BASE_DIR, "deployment")

# Création des répertoires s'ils n'existent pas
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(os.path.join(MODELS_DIR, "without_segmentation"), exist_ok=True)
os.makedirs(os.path.join(MODELS_DIR, "with_segmentation"), exist_ok=True)
os.makedirs(NOTEBOOKS_DIR, exist_ok=True)
os.makedirs(DEPLOYMENT_DIR, exist_ok=True)


def define_paths(data_dir):
    """Génère les chemins des fichiers et les étiquettes à partir de la structure des répertoires"""
    filepaths = []
    labels = []

    folds = os.listdir(data_dir)
    for fold in folds:
        foldpath = os.path.join(data_dir, fold)
        if not os.path.isdir(foldpath):
            continue
            
        filelist = os.listdir(foldpath)
        for file in filelist:
            fpath = os.path.join(foldpath, file)
            filepaths.append(fpath)
            labels.append(fold)

    return filepaths, labels


def define_df(files, classes):
    """Concatène les chemins des fichiers avec les étiquettes dans un dataframe"""
    Fseries = pd.Series(files, name='filepaths')
    Lseries = pd.Series(classes, name='labels')
    return pd.concat([Fseries, Lseries], axis=1)


def split_data(data_dir):
    """Divise le dataframe en ensembles d'entraînement, de validation et de test"""
    files, classes = define_paths(data_dir)
    df = define_df(files, classes)
    strat = df['labels']
    train_df, dummy_df = train_test_split(df, train_size=0.8, shuffle=True, random_state=123, stratify=strat)

    strat = dummy_df['labels']
    valid_df, test_df = train_test_split(dummy_df, train_size=0.5, shuffle=True, random_state=123, stratify=strat)

    return train_df, valid_df, test_df


def segment_image(image, method='otsu'):
    """
    Segmenter l'image pour isoler la plante du fond
    Methods: 'otsu', 'adaptive', 'kmeans'
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    if method == 'otsu':
        _, segmented = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif method == 'adaptive':
        segmented = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                         cv2.THRESH_BINARY, 11, 2)
    elif method == 'kmeans':
        # Reshape l'image pour K-means
        pixel_values = gray.reshape((-1, 1))
        pixel_values = np.float32(pixel_values)
        
        # Critères d'arrêt
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        
        # Nombre de clusters (K)
        k = 2
        
        # Application de K-means
        _, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        # Conversion des valeurs en uint8
        centers = np.uint8(centers)
        
        # Aplatir les labels
        labels = labels.flatten()
        
        # Conversion tous les pixels en couleur en fonction des labels
        segmented = centers[labels].reshape(gray.shape)
        
        # Binarisation
        _, segmented = cv2.threshold(segmented, 127, 255, cv2.THRESH_BINARY)
    else:
        raise ValueError(f"Méthode de segmentation non supportée: {method}")
    
    return segmented


def preprocess_image(img_path, target_size=(64, 64), with_segmentation=False, segmentation_method='otsu'):
    """Prétraitement d'une image avec ou sans segmentation"""
    try:
        # Vérifier que le fichier existe
        if not os.path.exists(img_path):
            print(f"Fichier non trouvé: {img_path}")
            return None
            
        # Charger l'image
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"Impossible de lire l'image: {img_path}")
            return None
        
        # Appliquer la segmentation si demandé
        if with_segmentation:
            segmented = segment_image(img, method=segmentation_method)
            # Appliquer le masque à l'image originale
            if len(img.shape) == 3:
                mask = cv2.merge([segmented, segmented, segmented])
                img = cv2.bitwise_and(img, mask)
            else:
                img = cv2.bitwise_and(img, segmented)
        
        # Redimensionner
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
        
        # Convertir en niveaux de gris
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Normalisation
        img = img.astype(np.float32) / 255.0
        
        # Aplatir l'image pour Random Forest/XGBoost
        img_flat = img.flatten()
        
        return img_flat
        
    except Exception as e:
        print(f"Erreur lors du traitement de {img_path}: {str(e)}")
        return None


def create_dataset(df, dataset_name, target_size=(64, 64), with_segmentation=False, segmentation_method='otsu'):
    """Prétraite un dataset complet"""
    print(f"\nTraitement du dataset {dataset_name} ({len(df)} images)...")
    
    filepaths = df['filepaths'].tolist()
    all_images = []
    
    for i, filepath in enumerate(filepaths):
        if i % 1000 == 0:  # Nettoyage périodique
            import gc
            gc.collect()
            
        img = preprocess_image(filepath, target_size, with_segmentation, segmentation_method)
        if img is not None:
            all_images.append(img)
    
    print(f"Dataset {dataset_name}: {len(all_images)}/{len(df)} images traitées avec succès")
    
    if len(all_images) == 0:
        raise ValueError(f"Aucune image n'a pu être traitée pour {dataset_name}")
    
    return np.array(all_images, dtype=np.float32)


def train_model(X_train, y_train, model_type='random_forest', params=None):
    """Entraîne un modèle (RandomForest ou XGBoost)"""
    if model_type == 'random_forest':
        model = RandomForestClassifier(**params if params else {})
    elif model_type == 'xgboost':
        model = XGBClassifier(**params if params else {})
    else:
        raise ValueError(f"Type de modèle non supporté: {model_type}")
    
    # Encoder les étiquettes si elles sont des chaînes
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    
    model.fit(X_train, y_train_encoded)
    
    return model, le


def evaluate_model(model, X_test, y_test, le):
    """Évalue un modèle et retourne les métriques"""
    y_pred = model.predict(X_test)
    y_test_encoded = le.transform(y_test)
    
    accuracy = accuracy_score(y_test_encoded, y_pred)
    report = classification_report(y_test_encoded, y_pred, output_dict=True)
    confusion_mat = confusion_matrix(y_test_encoded, y_pred)
    
    return {
        'accuracy': accuracy,
        'classification_report': report,
        'confusion_matrix': confusion_mat.tolist()  # Convertir en liste pour la sérialisation JSON
    }


def save_model_and_metrics(model, le, metrics, plant_name, model_type, with_segmentation=False):
    """Sauvegarde le modèle et les métriques"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Déterminer le répertoire de sauvegarde
    segment_dir = "with_segmentation" if with_segmentation else "without_segmentation"
    save_dir = os.path.join(MODELS_DIR, segment_dir)
    os.makedirs(save_dir, exist_ok=True)
    
    # Sauvegarder le modèle
    model_filename = f"{plant_name}_{model_type}_{timestamp}.joblib"
    model_path = os.path.join(save_dir, model_filename)
    joblib.dump(model, model_path)
    
    # Sauvegarder l'encodeur d'étiquettes
    le_filename = f"{plant_name}_{model_type}_le_{timestamp}.joblib"
    le_path = os.path.join(save_dir, le_filename)
    joblib.dump(le, le_path)
    
    # Sauvegarder les métriques
    metrics_filename = f"{plant_name}_{model_type}_metrics_{timestamp}.json"
    metrics_path = os.path.join(save_dir, metrics_filename)
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    print(f"Modèle sauvegardé dans: {model_path}")
    print(f"Métriques sauvegardées dans: {metrics_path}")
    
    return model_path, metrics_path


def show_sample_images(df, num_samples=25, target_size=(224, 224)):
    """Affiche des exemples d'images du dataframe"""
    sample_df = df.sample(n=min(num_samples, len(df)), random_state=42)
    
    plt.figure(figsize=(20, 20))
    
    for i, (idx, row) in enumerate(sample_df.iterrows()):
        if i >= 25:  # Limiter à 25 images
            break
            
        plt.subplot(5, 5, i + 1)
        
        # Charger et afficher l'image
        img = cv2.imread(row['filepaths'])
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convertir BGR en RGB pour matplotlib
            img = cv2.resize(img, target_size)
            plt.imshow(img)
            plt.title(row['labels'], color='blue', fontsize=12)
        else:
            plt.text(0.5, 0.5, 'Image non trouvée', ha='center', va='center')
            plt.title(row['labels'], color='red', fontsize=12)
            
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()



def main():
    """Fonction principale pour exécuter le pipeline"""
    parser = argparse.ArgumentParser(description='Pipeline de détection de maladies des plantes')
    parser.add_argument('--plant_name', type=str, required=True, help='Nom de la plante (ex: apple, tomato)')
    parser.add_argument('--model_type', type=str, choices=['random_forest', 'xgboost'], 
                        default='random_forest', help='Type de modèle à entraîner')
    parser.add_argument('--with_segmentation', action='store_true', 
                        help='Utiliser la segmentation dans le prétraitement')
    parser.add_argument('--segmentation_method', type=str, choices=['otsu', 'adaptive', 'kmeans'], 
                        default='otsu', help='Méthode de segmentation à utiliser')
    parser.add_argument('--target_size', type=int, nargs=2, default=[64, 64], 
                        help='Taille cible pour le redimensionnement des images (hauteur largeur)')
    parser.add_argument('--show_samples', action='store_true', 
                        help='Afficher des exemples d\'images du dataset')
    
    args = parser.parse_args()
    
    print(f"Détection de maladies pour: {args.plant_name}")
    print(f"Modèle: {args.model_type}")
    print(f"Segmentation: {'Activée' if args.with_segmentation else 'Désactivée'}")
    if args.with_segmentation:
        print(f"Méthode de segmentation: {args.segmentation_method}")
    
    # Définir le répertoire des données
    data_dir = os.path.join(DATA_DIR, f"{args.plant_name}")
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Répertoire de données non trouvé: {data_dir}")
    
    # Charger et diviser les données
    print("Chargement et division des données...")
    train_df, valid_df, test_df = split_data(data_dir)
    print("Chargement et division des données terminés.")
    
    # Afficher la distribution des classes
    print("\nDistribution des classes dans l'ensemble d'entraînement:")
    print(train_df['labels'].value_counts())
    
    # Afficher des exemples d'images si demandé
    if args.show_samples:
        print("\nAffichage d'exemples d'images de l'ensemble d'entraînement...")
        show_sample_images(train_df)
    
    # Prétraiter les images
    target_size = tuple(args.target_size)
    print("\nPrétraitement des images...")
    train_images = create_dataset(train_df, "TRAIN", target_size, args.with_segmentation, args.segmentation_method)
    valid_images = create_dataset(valid_df, "VALIDATION", target_size, args.with_segmentation, args.segmentation_method)
    test_images = create_dataset(test_df, "TEST", target_size, args.with_segmentation, args.segmentation_method)
    
    train_labels = train_df['labels'].values
    valid_labels = valid_df['labels'].values
    test_labels = test_df['labels'].values
    
    # Entraîner le modèle
    print("\nEntraînement du modèle...")
    model, le = train_model(train_images, train_labels, args.model_type)
    print("Entraînement terminé.")
    
    # Évaluer le modèle
    print("\nÉvaluation du modèle...")
    metrics = evaluate_model(model, test_images, test_labels, le)
    print(f"Précision: {metrics['accuracy']:.4f}")
    
    # Sauvegarder le modèle et les métriques
    print("\nSauvegarde du modèle et des métriques...")
    model_path, metrics_path = save_model_and_metrics(
        model, le, metrics, args.plant_name, args.model_type, args.with_segmentation
    )
    
    print("\nPipeline terminé avec succès!")

if __name__ == "__main__":
    main()