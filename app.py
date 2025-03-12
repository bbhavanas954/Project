import os
import fitz  # PyMuPDF
import google.generativeai as genai
import speech_recognition as sr
from flask import Flask, render_template, request, redirect, url_for, session, send_file, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from gtts import gTTS
import traceback
import pyttsx3
import textwrap  # ✅ Add this line
import cv2
import mediapipe as mp
import numpy as np
import threading
import datetime
from flask_socketio import SocketIO, emit
import json
import time
import tensorflow as tf 

app = Flask(__name__, template_folder='templates')
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///exam_system.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = 'c'
app.config['SESSION_PERMANENT'] = False  # Ensure sessions are not permanent
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

db = SQLAlchemy(app)
migrate = Migrate(app, db)

GEMINI_API_KEY = "A"
genai.configure(api_key=GEMINI_API_KEY)


# Database Models
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), nullable=False, unique=True)
    email = db.Column(db.String(100), nullable=False, unique=True)
    password_hash = db.Column(db.String(255), nullable=False)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)  # ✅ Fix


class VoiceAnswer(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    answer_text = db.Column(db.Text, nullable=False)


# User authentication
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        user = User.query.filter_by(email=email).first()
        if user and user.check_password(password):
            session['user_id'] = user.id
            session['user'] = user.username  # Track username in session
            return redirect(url_for('dashboard'))
        return "Invalid credentials, please try again."
    
    return render_template('login.html')


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        
        existing_user = User.query.filter_by(email=email).first()
        if existing_user:
            return "User already exists. Please log in."

        new_user = User(username=username, email=email)
        new_user.set_password(password)
        db.session.add(new_user)
        db.session.commit()
        return redirect(url_for('login'))
    
    return render_template('signup.html')


# Question bank generation
def extract_text_from_pdf(pdf_path):
    """Extract text from PDF using PyMuPDF."""
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text


def generate_questions(text):
    prompt = f"Generate 30 descriptive exam questions based on the following syllabus unit wise:\n\n{text}"
    model = genai.GenerativeModel("gemini-1.5-pro")
    response = model.generate_content(prompt)
    
    if response.text:
        questions = response.text.split("\n")  # Split output into a list
        return [q.strip() for q in questions if q.strip()]
    else:
        return ["Error: No questions generated. Check API key and response."]

def save_to_pdf(questions, filename="Generated_Questions.pdf"):
    c = canvas.Canvas(filename, pagesize=letter)
    c.setFont("Helvetica", 12)

    y_position = 750  
    line_height = 18  
    max_line_width = 100  

    for question in questions:
        if question.startswith("UNIT"):  
            c.setFont("Helvetica-Bold", 14)
            y_position -= 10  
        else:
            c.setFont("Helvetica", 12)
        
        wrapped_lines = textwrap.wrap(question, max_line_width)

        for line in wrapped_lines:
            c.drawString(50, y_position, line)
            y_position -= line_height  

            if y_position < 50:  
                c.showPage()
                c.setFont("Helvetica", 12)  
                y_position = 750

    c.save()
    return filename

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file uploaded.", 400
        
        file = request.files['file']
        if file.filename == '':
            return "No selected file.", 400
        
        if file and file.filename.endswith('.pdf'):
            pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
            file.save(pdf_path)
            
            extracted_text = extract_text_from_pdf(pdf_path)
            questions = generate_questions(extracted_text)  
            output_pdf = os.path.join(app.config['UPLOAD_FOLDER'], 'question_bank.pdf')

            save_to_pdf(questions, output_pdf)  

            session['generated_questions'] = questions
            session['output_pdf'] = output_pdf
            
            return render_template('upload.html', questions=questions)

    return render_template('upload.html', questions=None)


@app.route('/download')
def download():
    if 'output_pdf' not in session:
        return redirect(url_for('upload'))
    
    return send_file(session['output_pdf'], as_attachment=True)

# Voice answering

answers = []

@app.route("/exam")
def exam_page():
    return render_template("exam.html")

@app.route("/submit_exam", methods=["POST"])
def submit_exam():
    data = request.json  # Get JSON data from frontend
    answers.append(data)  # Store the answers
    print("📜 Exam Submitted. Answers:", answers)  # Log the submitted answers
    return jsonify({"status": "success", "message": "Exam Submitted Successfully!"})


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/dashboard')
def dashboard():
    if not session.get('user'):
        return redirect(url_for('login'))
    return render_template('dashboard.html')

@app.route('/cheating_prevention')
def cheating_prevention():
    return render_template("cheating_prevention.html")

@app.route('/exam')
def exam():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('exam.html')

@app.route('/logout')
def logout():
    """Logs out the user by clearing the session and redirecting to login."""
    session.pop('user_id', None)  # Remove user from session
    return redirect(url_for('home'))


if __name__ == '__main__':
    app.run(debug=True)
