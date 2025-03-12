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
app.config['SECRET_KEY'] = ''
app.config['SESSION_PERMANENT'] = False  # Ensure sessions are not permanent
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

db = SQLAlchemy(app)
migrate = Migrate(app, db)

GEMINI_API_KEY = ""
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
questions = []
answers = []
user_responses = []
current_question_index = 0


# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    """Extract text from the uploaded PDF"""
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text("text")
    return text.strip()


# Function to generate exam questions and answers
def generate_exam_questions(pdf_text):
    """Generate short-answer questions using Gemini API"""
    global questions, answers, user_responses, current_question_index
    questions, answers, user_responses = [], [], []

    prompt = (
        "Generate five one-mark questions based on the text. Each question should have a related answer.\n\n"
        f"Text:\n{pdf_text}\n\n"
        "Format:\n1. What is regression? Answer: Predictive modeling\n"
        "2. Define entropy. Answer: Measure of randomness\n"
        "3. What is overfitting? Answer: Poor generalization\n"
    )

    model = genai.GenerativeModel("gemini-1.5-pro")
    response = model.generate_content(prompt)

    if hasattr(response, 'text'):
        lines = response.text.strip().split("\n")
        for line in lines:
            parts = line.split("Answer:")
            if len(parts) == 2:
                question = parts[0].strip().strip("12345. ")
                answer = parts[1].strip()
                if question and answer:
                    questions.append(question)
                    answers.append(answer.lower())  # Store answers in lowercase for easy matching
                    user_responses.append(None)

        if questions:
            current_question_index = 0
            return questions
        else:
            return ["Error: Could not generate valid questions"]

    return ["Error: AI response missing questions"]


# API to upload PDF, generate questions, and start the quiz
@app.route("/upload_pdf", methods=["POST"])
def upload_pdf():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400

        file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(file_path)

        pdf_text = extract_text_from_pdf(file_path)
        questions = generate_exam_questions(pdf_text)

        return jsonify({"questions": questions})

    except Exception as e:
        print("❌ ERROR:", traceback.format_exc())
        return jsonify({"error": str(e)}), 500


# API to get the next question
@app.route("/next_question", methods=["GET"])
def next_question():
    global current_question_index

    if current_question_index < len(questions):
        question = questions[current_question_index]
        return jsonify({"question": question})
    
    return jsonify({"question": "All questions completed."})


# API to submit an answer
@app.route("/submit_answer", methods=["POST"])
def submit_answer():
    global current_question_index

    try:
        if not questions or not answers:
            return jsonify({"error": "No questions available"}), 400

        data = request.json
        user_answer = data.get("answer", "").strip().lower()

        if not user_answer:
            return jsonify({"error": "Answer cannot be empty"}), 400

        if current_question_index < len(questions):
            correct_answer = answers[current_question_index].strip().lower()
            user_responses[current_question_index] = user_answer

            is_correct = user_answer == correct_answer
            current_question_index += 1  # Move to next question

            if current_question_index < len(questions):
                return jsonify({"next_question": questions[current_question_index], "correct": is_correct})

            # Calculate Final Score
            correct_count = sum(
                1 for i in range(len(questions)) if user_responses[i] and user_responses[i] == answers[i].strip().lower()
            )
            score = f"{correct_count} / {len(questions)}"

            return jsonify({"message": "All questions answered.", "final_score": score})

        return jsonify({"error": "No more questions"}), 400

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/voice_exam')
def voice_exam():
    return render_template('voice_exam.html')

#cheating_prevention
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

socketio = SocketIO(app, cors_allowed_origins="*")

# ✅ Exam State
exam_active = False
face_tracking_strikes = 0
tab_switch_strikes = 0
phone_detection_strikes = 0
multiple_people_strikes = 0

MAX_FACE_VIOLATIONS = 5
MAX_TAB_SWITCHES = 3
MAX_PHONE_VIOLATIONS = 2
MAX_PEOPLE_VIOLATIONS = 2

# ✅ Face Detection (MediaPipe)
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)

# ✅ Log File Setup
LOG_FILE = "cheating_logs.json"
cheating_logs = []

def save_logs():
    """Save logs to file"""
    with open(LOG_FILE, "w") as file:
        json.dump(cheating_logs, file, indent=4)

def log_cheating(event):
    """Log cheating events and send alerts"""
    global face_tracking_strikes, tab_switch_strikes, phone_detection_strikes, multiple_people_strikes, exam_active

    if not exam_active:
        return

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = {"timestamp": timestamp, "event": event}
    cheating_logs.append(log_entry)
    save_logs()
    socketio.emit("cheating_alert", log_entry)

    # 🚨 Auto Redirect to Dashboard if violations exceed limits
    if event in ["Looking Left", "Looking Right", "No face detected!"]:
        face_tracking_strikes += 1
        if face_tracking_strikes >= MAX_FACE_VIOLATIONS:
            socketio.emit("redirect_dashboard")
            log_cheating("🚨 Student Redirected due to Face Tracking Violations")
            return

    if event == "User switched tab":
        tab_switch_strikes += 1
        if tab_switch_strikes >= MAX_TAB_SWITCHES:
            socketio.emit("redirect_dashboard")
            log_cheating("🚨 Student Redirected due to Tab Switches")
            return

    if event == "Mobile phone detected":
        phone_detection_strikes += 1
        if phone_detection_strikes >= MAX_PHONE_VIOLATIONS:
            socketio.emit("redirect_dashboard")
            log_cheating("🚨 Student Redirected due to Mobile Phone Usage")
            return

    if event == "Multiple people detected":
        multiple_people_strikes += 1
        if multiple_people_strikes >= MAX_PEOPLE_VIOLATIONS:
            socketio.emit("redirect_dashboard")
            log_cheating("🚨 Student Redirected due to Unauthorized Person in Room")
            return

def detect_phone_or_multiple_people(frame):
    """Detect mobile phones or multiple people in the frame"""
    global exam_active
    if not exam_active:
        return None

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detection.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if faces.detections and len(faces.detections) > 1:
        return "Multiple people detected"

    if len(faces.detections) == 1:
        return None  # Normal case, one person detected

    return "No face detected!"

def monitor_face_tracking():
    """AI-based cheating detection (Face, Gaze, and Phone Detection)"""
    global exam_active
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret or not exam_active:
            continue

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        additional_check = detect_phone_or_multiple_people(frame)

        if additional_check:
            log_cheating(additional_check)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                landmarks = [(lm.x * frame.shape[1], lm.y * frame.shape[0]) for lm in face_landmarks.landmark]
                
                left_eye = np.mean([landmarks[i] for i in [33, 133, 159, 145]], axis=0)
                right_eye = np.mean([landmarks[i] for i in [362, 263, 386, 374]], axis=0)

                if left_eye[0] < right_eye[0] - 10:
                    log_cheating("Looking Left")
                elif left_eye[0] > right_eye[0] + 10:
                    log_cheating("Looking Right")
        else:
            log_cheating("No face detected!")

        cv2.imshow("AI Exam Monitoring", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

@app.route("/start_exam", methods=["POST"])
def start_exam():
    """Start the exam and enable AI monitoring"""
    global exam_active, face_tracking_strikes, tab_switch_strikes, phone_detection_strikes, multiple_people_strikes
    exam_active = True
    face_tracking_strikes = 0
    tab_switch_strikes = 0
    phone_detection_strikes = 0
    multiple_people_strikes = 0
    return jsonify({"message": "Exam started, AI monitoring active."})

@app.route("/submit_exam", methods=["POST"])
def submit_exam():
    """End the exam and disable AI monitoring"""
    global exam_active
    exam_active = False
    return jsonify({"message": "Exam submitted, AI monitoring stopped."})

@socketio.on("exam_started")
def handle_exam_start():
    """Start exam (WebSocket)"""
    global exam_active
    exam_active = True
    socketio.emit("exam_status", {"status": "Exam started, monitoring enabled"})

@socketio.on("exam_ended")
def handle_exam_end():
    """End exam (WebSocket)"""
    global exam_active
    exam_active = False
    socketio.emit("exam_status", {"status": "Exam ended, monitoring disabled"})

@socketio.on("tab_switch_detected")
def handle_tab_switch():
    """Detect tab switching"""
    log_cheating("User switched tab")

# ✅ Start AI Monitoring Thread (Only when exam starts)
threading.Thread(target=monitor_face_tracking, daemon=True).start()

#routers
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
