import os
import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image
from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory
from werkzeug.utils import secure_filename
import numpy as np

app = Flask(__name__)
app.secret_key = 'supersecretkey'  # Changez ceci pour une clé sécurisée

# Configuration
UPLOAD_FOLDER = 'static/uploads/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Détection du device (CPU/GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transformations pour les images
data_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4823, 0.4823, 0.4823], std=[0.2412, 0.2412, 0.2412])
])

# Classes
class_names = ['NORMAL', 'PNEUMONIA']

# Charger les modèles
models_dict = {
    'resnet50': models.resnet50(weights=None)
}

# Charger les poids des modèles
for model_name, model in models_dict.items():
    num_ftrs = model.fc.in_features  # Pour ResNet50
    model.fc = nn.Linear(num_ftrs, 2)
    model_path = 'best_model.pth'
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Le fichier {model_path} est introuvable. Placez best_model.pth dans le répertoire du projet.")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    model.to(device)

# Vérifier les extensions des fichiers
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Fonction de prédiction
def predict_image(image_path, model):
    try:
        model.eval()
        image = Image.open(image_path).convert('RGB')
        image = data_transforms(image).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(image)
            _, pred = torch.max(outputs, 1)
        return class_names[pred.item()]
    except Exception as e:
        return f"Erreur: {str(e)}"

# Routes
@app.route('/')
@app.route('/home')
def home():
    return send_from_directory('static', 'home.html')

@app.route('/model/<model_name>')
def model_details(model_name):
    if model_name not in models_dict:
        return "Modèle non trouvé", 404
    model_info = {
        'resnet50': {'desc': 'ResNet50 avec 50 couches et blocs résiduels.', 'link': 'https://github.com/votre-repo/resnet50', 'accuracy': '95.2%'},
        'vgg16': {'desc': 'Modèle VGG16 avec 16 couches, utilisant le transfert learning.', 'link': 'https://github.com/votre-repo/vgg16', 'accuracy': '94.5%'},
        'cnn': {'desc': 'CNN personnalisé avec 2 couches convolutives.', 'link': 'https://github.com/votre-repo/cnn', 'accuracy': '92.8%'}
    }.get(model_name, {'desc': 'Détails non disponibles', 'link': '#', 'accuracy': 'N/A'})
    return render_template('model.html', model_name=model_name, model_info=model_info)

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('Aucun fichier sélectionné.')
            return redirect(request.url)
        file = request.files['file']
        model_name = request.form['model']
        if file.filename == '':
            flash('Aucun fichier sélectionné.')
            return redirect(request.url)
        if file and allowed_file(file.filename) and model_name in models_dict:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            prediction = predict_image(file_path, models_dict[model_name])
            return render_template('result.html', filename=filename, prediction=prediction, model_name=model_name)
        else:
            flash('Type de fichier non autorisé ou modèle invalide.')
            return redirect(request.url)
    return render_template('upload.html', models=models_dict.keys())

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/models')
def models_page():
    return render_template('models.html')

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)