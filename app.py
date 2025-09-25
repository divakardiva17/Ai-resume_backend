from flask import Flask, request, jsonify
import pickle
import os
from werkzeug.utils import secure_filename
from src.feature_extraction import extract_keywords
from src.model import load_model

app = Flask(__name__)

# Load the pre-trained model
model = load_model('models/trained_model.pkl')

# Allowed file extensions
ALLOWED_EXTENSIONS = {'pdf', 'txt', 'docx'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/predict', methods=['POST'])
def predict():
    if 'resume' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['resume']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join('uploads', filename)
        file.save(file_path)

        # Extract text from the file (use your custom extraction method here)
        text = extract_text_from_pdf(file_path)  # Or handle DOCX and TXT

        # Extract keywords and make prediction
        features = extract_keywords(text)
        prediction = model.predict([features])

        return jsonify({"prediction": prediction[0]})

    return jsonify({"error": "Invalid file format"}), 400

if __name__ == '__main__':
    app.run(debug=True)
