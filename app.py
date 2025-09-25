from flask import Flask, request, jsonify
import pickle
import os
from werkzeug.utils import secure_filename
from pdfminer.high_level import extract_text
from src.feature_extraction import extract_keywords
from src.model import load_model

app = Flask(__name__)

# Load the pre-trained model
model = load_model('models/trained_model.pkl')

# Allowed file extensions for resume
ALLOWED_EXTENSIONS = {'pdf', 'txt', 'docx'}

# Check if file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to extract text from PDF file
def extract_text_from_pdf(pdf_path):
    return extract_text(pdf_path)

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
        
        # Extract text from the uploaded file (Handle different file formats here)
        if filename.endswith('.pdf'):
            text = extract_text_from_pdf(file_path)
        else:
            text = file.read().decode('utf-8')

        # Process the extracted text to get features
        features = extract_keywords(text)

        # Make a prediction
        prediction = model.predict([features])

        # Return the prediction result
        return jsonify({"prediction": prediction[0]})

    return jsonify({"error": "Invalid file format"}), 400

if __name__ == '__main__':
    # Make sure to create the 'uploads' directory for file saving
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    
    app.run(debug=True)
