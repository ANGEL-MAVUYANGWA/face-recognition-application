"""
Face Recognition Web Application
Author: ANGEL MAVUYANGWA
Description: A web app that detects faces in images and provides AI analysis
"""

# Import required libraries
import os  # For file operations
import cv2  # For image processing
import numpy as np  # For numerical operations
import sqlite3  # For database operations
import json  # For handling JSON data
from datetime import datetime  # For timestamps

# Flask imports for web framework
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user

# Password hashing for security
import bcrypt

# For secure file naming
from werkzeug.utils import secure_filename

# Import our custom models
from models.face_model import FaceRecognitionModel
from models.image_analyzer import ImageAnalyzer

# ============= CONFIGURATION =============
# Create the Flask app
app = Flask(__name__)

# Secret key for session security (in production, use environment variable)
app.config['SECRET_KEY'] = 'your-super-secret-key-change-this'

# Folder where uploaded images will be stored
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Maximum file size (16MB)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Allowed image file extensions
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ============= INITIALIZE AI MODELS =============
# Create instance of our face recognition model
face_model = FaceRecognitionModel()

# Create instance of our image analyzer
image_analyzer = ImageAnalyzer()

# ============= DATABASE SETUP =============
def get_db():
    """
    Get a connection to the SQLite database
    Returns: Database connection object
    """
    conn = sqlite3.connect('database/users.db')
    conn.row_factory = sqlite3.Row  # This allows us to access columns by name
    return conn

def init_db():
    """
    Initialize the database by creating tables if they don't exist
    This is called when the app starts
    """
    # Create database directory if it doesn't exist
    os.makedirs('database', exist_ok=True)
    
    conn = get_db()
    
    # Create users table
    conn.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            is_admin INTEGER DEFAULT 0
        )
    ''')
    
    # Create analysis history table
    conn.execute('''
        CREATE TABLE IF NOT EXISTS analysis_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            image_path TEXT NOT NULL,
            analysis_result TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    conn.commit()
    conn.close()
    print("Database initialized successfully!")

# Call this when app starts
init_db()

# ============= FLASK-LOGIN SETUP =============
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'  # Where to redirect if not logged in
login_manager.login_message = 'Please log in to access this page.'

class User(UserMixin):
    """
    User class for Flask-Login
    This represents a logged-in user
    """
    def __init__(self, id, username, email, is_admin=0):
        self.id = id
        self.username = username
        self.email = email
        self.is_admin = is_admin

@login_manager.user_loader
def load_user(user_id):
    """
    Load a user from the database by ID
    Flask-Login calls this to get user objects
    """
    conn = get_db()
    user = conn.execute('SELECT * FROM users WHERE id = ?', (user_id,)).fetchone()
    conn.close()
    
    if user:
        return User(user['id'], user['username'], user['email'], user['is_admin'])
    return None

# ============= HELPER FUNCTIONS =============
def allowed_file(filename):
    """
    Check if a file has an allowed extension
    Args:
        filename: The name of the file to check
    Returns: True if allowed, False otherwise
    """
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# ============= ROUTES (Web Pages) =============
@app.route('/')
def index():
    """
    Home page - accessible to everyone
    """
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    """
    User registration page
    GET: Show registration form
    POST: Process registration form
    """
    if request.method == 'POST':
        # Get form data
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        
        # Check if passwords match
        if password != confirm_password:
            flash('Passwords do not match!', 'danger')
            return redirect(url_for('register'))
        
        # Hash the password for security (never store plain text passwords!)
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        
        # Try to insert into database
        conn = get_db()
        try:
            conn.execute(
                'INSERT INTO users (username, email, password) VALUES (?, ?, ?)',
                (username, email, hashed_password)
            )
            conn.commit()
            flash('Registration successful! Please log in.', 'success')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            # This happens if username or email already exists
            flash('Username or email already exists!', 'danger')
        finally:
            conn.close()
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    """
    User login page
    GET: Show login form
    POST: Process login
    """
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        # Look up user in database
        conn = get_db()
        user = conn.execute('SELECT * FROM users WHERE username = ?', (username,)).fetchone()
        conn.close()
        
        # Check if user exists and password is correct
        if user and bcrypt.checkpw(password.encode('utf-8'), user['password']):
            # Create user object and log them in
            user_obj = User(user['id'], user['username'], user['email'], user['is_admin'])
            login_user(user_obj)
            flash(f'Welcome back, {username}!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password!', 'danger')
    
    return render_template('login.html')

@app.route('/logout')
@login_required  # This decorator ensures only logged-in users can access
def logout():
    """
    Log out the current user
    """
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('index'))

@app.route('/dashboard')
@login_required
def dashboard():
    """
    User dashboard - shows analysis history
    """
    conn = get_db()
    # Get last 10 analyses for this user
    history = conn.execute(
        'SELECT * FROM analysis_history WHERE user_id = ? ORDER BY created_at DESC LIMIT 10',
        (current_user.id,)
    ).fetchall()
    conn.close()
    return render_template('dashboard.html', history=history)

@app.route('/analyze', methods=['GET', 'POST'])
@login_required
def analyze():
    """
    Image analysis page
    GET: Show upload form
    POST: Process uploaded image and return analysis
    """
    if request.method == 'POST':
        # Check if an image was uploaded
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Check if file type is allowed
        if file and allowed_file(file.filename):
            # Create a unique filename using timestamp and user ID
            filename = secure_filename(f"{current_user.id}_{datetime.now().timestamp()}_{file.filename}")
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Perform face detection
            print("Analyzing face...")
            face_result = face_model.analyze_face(filepath)
            print(f"Face result: {face_result}")
            
            # Perform AI image analysis
            print("Analyzing image...")
            analysis_result = image_analyzer.analyze_image(filepath)
            print("Image analysis complete")
            
            # Combine results
            result = {
                'face_detection': face_result,
                'image_analysis': analysis_result,
                'image_path': f'/static/uploads/{filename}'
            }
            
            # Save to history for this user
            conn = get_db()
            conn.execute(
                'INSERT INTO analysis_history (user_id, image_path, analysis_result) VALUES (?, ?, ?)',
                (current_user.id, filename, json.dumps(result))
            )
            conn.commit()
            conn.close()
            
            return jsonify(result)
        
        return jsonify({'error': 'Invalid file type'}), 400
    
    return render_template('analyze.html')

# ============= API ENDPOINTS =============
@app.route('/api/history/<int:history_id>')
@login_required
def get_history(history_id):
    """
    Get a specific analysis result by ID
    This is called by JavaScript to show detailed results
    """
    conn = get_db()
    history = conn.execute(
        'SELECT * FROM analysis_history WHERE id = ? AND user_id = ?',
        (history_id, current_user.id)
    ).fetchone()
    conn.close()
    
    if history:
        result = json.loads(history['analysis_result'])
        return jsonify(result)
    return jsonify({'error': 'History not found'}), 404

# ============= RUN THE APP =============
if __name__ == '__main__':
    print("=" * 50)
    print("Face Recognition App Starting...")
    print(f"Upload folder: {app.config['UPLOAD_FOLDER']}")
    print("Open http://localhost:5000 in your browser")
    print("=" * 50)
    app.run(debug=True, host='0.0.0.0', port=5000)