🩺 Pneumonia Detection Using CNN + LSTM Hybrid Model (Flask Web App)
This Flask-based web application allows users to upload pediatric chest X-ray images and receive a prediction indicating whether the patient is Normal or has Pneumonia. It leverages a hybrid deep learning model combining Convolutional Neural Networks (CNNs) and Long Short-Term Memory (LSTM) networks trained using TensorFlow/Keras.

📁 Dataset
We use the Pediatric Chest X-ray Dataset, categorized into:

Normal

Pneumonia

Folder Structure:
css
Copy
Edit
Pediatric Chest X-ray Pneumonia/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── test/
│   ├── NORMAL/
│   └── PNEUMONIA/
📦 Dataset Source:
Kaggle - Chest X-Ray Images (Pneumonia)

🧠 Model Architecture: CNN + LSTM Hybrid
This architecture combines the best of both worlds:

CNN extracts spatial features from X-ray images.

Reshape Layer converts CNN features into sequences.

LSTM processes these sequences to learn temporal/spatial dependencies.

Dense Layers output final classification: Normal or Pneumonia.

🔧 Model Highlights:
Component	Purpose
CNN	Local feature extraction from images
LSTM	Capturing dependencies across feature regions
Reshape	Convert CNN output to a sequence for LSTM
Dense	Classification layer
Softmax	Output probabilities

🌐 Web App (Flask)
Features:
Upload .jpg, .jpeg, or .png chest X-ray image

View prediction results with confidence score

Displays uploaded image and output class

🚀 How to Run the Project
1. Clone the Repository
bash
Copy
Edit
git clone https://github.com/your-repo/pneumonia-flask-app.git
cd pneumonia-flask-app
2. Install Required Dependencies
bash
Copy
Edit
pip install -r requirements.txt
📌 Or install manually:

bash
Copy
Edit
pip install flask tensorflow keras numpy pillow
3. Prepare Folder Structure
Make sure you have the following:

templates/index.html – the frontend template

static/uploaded_images/ – folder to store uploaded images

pneumonia_model.h5 – the trained CNN+LSTM model file in the project root

4. Run the Application
bash
Copy
Edit
python app.py
5. Open in Browser
Visit: http://127.0.0.1:5000

📂 Project Structure
php
Copy
Edit
pneumonia-flask-app/
├── app.py                     # Main Flask application
├── pneumonia_model.h5         # Trained CNN+LSTM model
├── templates/
│   └── index.html             # Frontend HTML
├── static/
│   └── uploaded_images/       # Stores uploaded X-ray images
└── README.md                  # Project documentation
🧪 Evaluation Metrics
Accuracy

Confusion Matrix

Precision / Recall / F1-Score

Evaluated using scikit-learn and visualized using matplotlib and seaborn.

📊 Visualization Tools
matplotlib.pyplot

seaborn.heatmap – for visualizing confusion matrices

🖼️ Sample Output (Frontend)
Image preview of the uploaded chest X-ray

Classification result: Normal or Pneumonia

Model confidence score (e.g., 92.65%)

🧾 Summary
Concept	Description
CNN	Extract spatial features from chest X-ray images
LSTM	Understand region-to-region spatial dependencies
Reshape Layer	Convert 3D CNN output to sequential input for LSTM
Softmax Activation	Final classification output
Flask	Lightweight web framework for deployment