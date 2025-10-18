import os
import pickle
import numpy as np
import parselmouth
from parselmouth.praat import call
from flask import Flask, request, render_template, jsonify, redirect, url_for, flash, session
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash

# --- App Initialization ---
app = Flask(__name__)
app.config['SECRET_KEY'] = 'a_very_secret_key_that_should_be_changed'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# --- Database Model ---
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    first_name = db.Column(db.String(150), nullable=False)
    last_name = db.Column(db.String(150), nullable=False)
    dob = db.Column(db.String(50), nullable=False)
    age = db.Column(db.Integer, nullable=False)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password_hash = db.Column(db.String(150), nullable=False)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

# --- Load ML Model ---
try:
    with open('parkinsons_model_simple.pkl', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    print("ERROR: Model file 'parkinsons_model_simple.pkl' not found.")
    model = None # Handle case where model is not found

# --- Feature Extraction Logic ---
def extract_features(file_path):
    try:
        sound = parselmouth.Sound(file_path)
        point_process = call(sound, "To PointProcess (periodic, cc)", 60, 600)
        mean_f0_val = call(sound.to_pitch(), "Get mean", 0, 0, "Hertz")
        f0_max = call(sound.to_pitch(), "Get maximum", 0, 0, "Hertz", "Parabolic")
        f0_min = call(sound.to_pitch(), "Get minimum", 0, 0, "Hertz", "Parabolic")
        local_jitter = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        local_absolute_jitter = call(point_process, "Get jitter (local, absolute)", 0, 0, 0.0001, 0.02, 1.3)
        rap_jitter = call(point_process, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
        ppq5_jitter = call(point_process, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3)
        ddp_jitter = call(point_process, "Get jitter (ddp)", 0, 0, 0.0001, 0.02, 1.3)
        local_shimmer = call([sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        local_db_shimmer = local_shimmer
        apq3_shimmer = call([sound, point_process], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        apq5_shimmer = call([sound, point_process], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        apq11_shimmer = call([sound, point_process], "Get shimmer (apq11)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        dda_shimmer = call([sound, point_process], "Get shimmer (dda)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        harmonics = call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
        hnr = call(harmonics, "Get mean", 0, 0)
        nhr = 1 / (10**(hnr/10))
        return [
            mean_f0_val, f0_max, f0_min, local_jitter, local_absolute_jitter,
            rap_jitter, ppq5_jitter, ddp_jitter, local_shimmer, local_db_shimmer,
            apq3_shimmer, apq5_shimmer, apq11_shimmer, dda_shimmer, nhr, hnr
        ]
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None

# --- Main Application Routes ---
@app.route('/')
def home():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('index.html')

# --- Find this function in your app.py and replace it ---
@app.route('/predict', methods=['POST'])
def predict():
    if 'user_id' not in session:
        return jsonify({'error': 'User not logged in'}), 401
    if model is None:
        return jsonify({'error': 'Model is not loaded'}), 500
    if 'file' not in request.files:
        return jsonify({'error': 'No audio file found'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file:
        temp_dir = 'temp_audio'
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        temp_path = os.path.join(temp_dir, file.filename)
        file.save(temp_path)
        
        raw_features = extract_features(temp_path)
        os.remove(temp_path)
        
        if raw_features is None:
            return jsonify({'error': 'Feature extraction failed.'}), 400
            
        features_array = np.array(raw_features).reshape(1, -1)
        prediction = model.predict(features_array)
        probabilities = model.predict_proba(features_array)[0]
        confidence = probabilities[int(prediction[0])]
        
        # --- NEW: Package key features for the dashboard ---
        # Indices from the extract_features function list
        # local_jitter = index 3, local_shimmer = index 8, hnr = index 15
        feature_values = {
            'jitter': round(raw_features[3] * 100, 4), # Convert to percentage
            'shimmer': round(raw_features[8] * 100, 4), # Convert to percentage
            'hnr': round(raw_features[15], 2) # Decibels
        }
        
        result = {
            'prediction': int(prediction[0]), 
            'confidence': float(confidence),
            'features': feature_values # Add the new feature dictionary
        }
        
        return jsonify(result)

# --- Authentication Routes ---
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = User.query.filter_by(username=username).first()
        if user and user.check_password(password):
            session['user_id'] = user.id
            session['first_name'] = user.first_name
            return redirect(url_for('home'))
        else:
            flash('Invalid username or password.', 'error')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        user = User(
            first_name=request.form.get('first_name'),
            last_name=request.form.get('last_name'),
            dob=request.form.get('dob'),
            age=request.form.get('age'),
            username=request.form.get('username')
        )
        user.set_password(request.form.get('password'))
        db.session.add(user)
        db.session.commit()
        flash('Registration successful! Please log in.', 'success')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    session.pop('first_name', None)
    return redirect(url_for('login'))

@app.route('/profile')
def profile():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    user = User.query.get(session['user_id'])
    return render_template('profile.html', user=user)

# --- ADD THIS NEW ROUTE TO YOUR app.py FILE ---

@app.route('/change_password', methods=['POST'])
def change_password():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    user = User.query.get(session['user_id'])
    current_password = request.form.get('current_password')
    new_password = request.form.get('new_password')
    confirm_password = request.form.get('confirm_password')

    # Validate the form data
    if not user or not user.check_password(current_password):
        flash('Your current password was incorrect.', 'error')
        return redirect(url_for('profile'))
    
    if not new_password or len(new_password) < 6:
        flash('New password must be at least 6 characters long.', 'error')
        return redirect(url_for('profile'))

    if new_password != confirm_password:
        flash('New passwords do not match.', 'error')
        return redirect(url_for('profile'))

    # If all checks pass, update the password
    user.set_password(new_password)
    db.session.commit()
    flash('Your password has been updated successfully!', 'success')
    
    return redirect(url_for('profile'))

if __name__ == '__main__':
    with app.app_context():
        db.create_all() # This will create the database file and table
    app.run(debug=True)