from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    session,
    flash,
    jsonify
)

from werkzeug.utils import secure_filename
from database import init_db, get_db_connection
from werkzeug.security import generate_password_hash, check_password_hash
import tensorflow as tf
import numpy as np
import cv2
import os
import uuid

app = Flask(__name__)

app.secret_key = "your-secret-key"


# =========================
# UPLOAD CONFIGURATION
# =========================

UPLOAD_FOLDER = os.path.join(
    app.root_path,
    "static",
    "uploads"
)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = {
    "png",
    "jpg",
    "jpeg",
    "jfif",
    "webp"
}

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)

def allowed_file(filename):

    return (
        "." in filename
        and
        filename.rsplit(".", 1)[1].lower()
        in ALLOWED_EXTENSIONS
    )

# Initialize database
init_db()

# =========================
# LOAD AI MODEL
# =========================

MODEL_PATH = "model/best_efficientnetb0.keras"

model = tf.keras.models.load_model(MODEL_PATH)

classes = [
    "cordana",
    "healthy",
    "panama",
    "sigatoka"] 


IMAGE_SIZE = (224, 224)
# =========================
# HOME
# =========================

@app.route("/")
def home():

    if "user_id" in session:
        return redirect(url_for("profile"))

    return redirect(url_for("login"))


# =========================
# REGISTER
# =========================

@app.route("/register", methods=["GET", "POST"])
def register():

    if request.method == "POST":

        name = request.form["name"].strip()
        email = request.form["email"].strip().lower()
        password = request.form["password"]

        # Basic validation
        if not name or not email or not password:
            flash("All fields are required.")
            return redirect(url_for("register"))

        if len(password) < 6:
            flash("Password must be at least 6 characters.")
            return redirect(url_for("register"))

        conn = get_db_connection()

        # Check if email already exists
        existing_user = conn.execute(
            "SELECT id FROM users WHERE email = ?",
            (email,)
        ).fetchone()

        if existing_user:
            conn.close()
            flash("An account with this email already exists.")
            return redirect(url_for("register"))

        # Hash password
        password_hash = generate_password_hash(password)

        # Create user
        conn.execute(
            """
            INSERT INTO users (name, email, password_hash)
            VALUES (?, ?, ?)
            """,
            (name, email, password_hash)
        )

        conn.commit()
        conn.close()

        flash("Registration successful! Please log in.")
        return redirect(url_for("login"))

    return render_template("register.html")


# =========================
# LOGIN
# =========================

@app.route("/login", methods=["GET", "POST"])
def login():

    if request.method == "POST":

        email = request.form["email"].strip().lower()
        password = request.form["password"]

        conn = get_db_connection()

        user = conn.execute(
            """
            SELECT *
            FROM users
            WHERE email = ?
            """,
            (email,)
        ).fetchone()

        conn.close()

        # Check user and password
        if user and check_password_hash(
            user["password_hash"],
            password
        ):

            # Store user information in session
            session["user_id"] = user["id"]
            session["user_name"] = user["name"]
            session["user_email"] = user["email"]

            return redirect(url_for("profile"))

        flash("Invalid email or password.")
        return redirect(url_for("login"))

    return render_template("login.html")


# =========================
# LOGOUT
# =========================

@app.route("/logout")
def logout():

    session.clear()

    flash("You have been logged out.")

    return redirect(url_for("login"))



  # =========================
# PROFILE / DASHBOARD
# =========================

@app.route("/profile")
def profile():

    if "user_id" not in session:
        return redirect(url_for("login"))

    user_id = session["user_id"]

    conn = get_db_connection()

    # Get user information
    user = conn.execute(
        """
        SELECT *
        FROM users
        WHERE id = ?
        """,
        (user_id,)
    ).fetchone()

    # Total predictions
    total_predictions = conn.execute(
        """
        SELECT COUNT(*)
        FROM predictions
        WHERE user_id = ?
        """,
        (user_id,)
    ).fetchone()[0]

    # Healthy count
    healthy_count = conn.execute(
        """
        SELECT COUNT(*)
        FROM predictions
        WHERE user_id = ?
        AND prediction = 'healthy'
        """,
        (user_id,)
    ).fetchone()[0]

    # Cordana count
    cordana_count = conn.execute(
        """
        SELECT COUNT(*)
        FROM predictions
        WHERE user_id = ?
        AND prediction = 'cordana'
        """,
        (user_id,)
    ).fetchone()[0]

    # Panama count
    panama_count = conn.execute(
        """
        SELECT COUNT(*)
        FROM predictions
        WHERE user_id = ?
        AND prediction = 'panama'
        """,
        (user_id,)
    ).fetchone()[0]
    
    # Sigatoka count
    sigatoka_count = conn.execute(
        """
        SELECT COUNT(*)
        FROM predictions
        WHERE user_id = ?
        AND prediction = 'sigatoka'
        """,
        (user_id,)
    ).fetchone()[0]

    # Diseased leaves
    diseased_count = (
        cordana_count
        + panama_count
        + sigatoka_count
    )

    # Average confidence
    average_confidence = conn.execute(
        """
        SELECT AVG(confidence)
        FROM predictions
        WHERE user_id = ?
        """,
        (user_id,)
    ).fetchone()[0]

    if average_confidence is None:
        average_confidence = 0

    # Most common prediction
    most_common = conn.execute(
        """
        SELECT prediction, COUNT(*) AS count
        FROM predictions
        WHERE user_id = ?
        GROUP BY prediction
        ORDER BY count DESC
        LIMIT 1
        """,
        (user_id,)
    ).fetchone()

    if most_common:
        most_common_prediction = most_common["prediction"]
    else:
        most_common_prediction = "None"

    conn.close()

    return render_template(
        "profile.html",
        user=user,
        total_predictions=total_predictions,
        healthy_count=healthy_count,
        diseased_count=diseased_count,
        cordana_count=cordana_count,
        panama_count=panama_count,
        sigatoka_count=sigatoka_count,
        average_confidence=average_confidence,
        most_common_prediction=most_common_prediction
    )

# =========================
# PREDICTION
# =========================

@app.route("/predict", methods=["GET", "POST"])
def predict():

    if "user_id" not in session:
        return redirect(url_for("login"))

    # -----------------------------
    # SHOW PREDICTION PAGE
    # -----------------------------
    if request.method == "GET":
        return render_template("predict.html")


    # -----------------------------
    # CHECK FILE
    # -----------------------------
    if "image" not in request.files:
        return jsonify({
            "success": False,
            "error": "No image was uploaded."
        }), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({
            "success": False,
            "error": "Please select an image."
        }), 400


    # -----------------------------
    # CHECK EXTENSION
    # -----------------------------
    extension = file.filename.rsplit(".", 1)[-1].lower()

    if extension not in ALLOWED_EXTENSIONS:
        return jsonify({
            "success": False,
            "error": "Invalid image format. Use JPG, JPEG, PNG, jfif or WEBP."
        }), 400


    # -----------------------------
    # CREATE UNIQUE FILENAME
    # -----------------------------
    original_name = secure_filename(file.filename)

    if not original_name:
        return jsonify({
            "success": False,
            "error": "Invalid filename."
        }), 400

    unique_filename = (
        uuid.uuid4().hex + "." + extension
    )

    save_path = os.path.join(
        app.config["UPLOAD_FOLDER"],
        unique_filename
    )


    # -----------------------------
    # SAVE IMAGE
    # -----------------------------
    file.save(save_path)


    # -----------------------------
    # READ IMAGE WITH OPENCV
    # -----------------------------
    image = cv2.imread(save_path)

    if image is None:
        return jsonify({
            "success": False,
            "error": "The uploaded image could not be read."
        }), 400


    # -----------------------------
    # BGR -> RGB
    # -----------------------------
    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )


    # -----------------------------
    # RESIZE
    # SAME AS YOUR MODEL INPUT
    # -----------------------------
    image = cv2.resize(
        image,
        (224, 224),
        interpolation=cv2.INTER_LINEAR
    )


    # -----------------------------
    # NORMALIZE
    # -----------------------------
    image = image.astype(np.float32) 


    # -----------------------------
    # ADD BATCH DIMENSION
    # -----------------------------
    image = np.expand_dims(image, axis=0)


    # -----------------------------
    # PREDICTION
    # -----------------------------
    predictions = model.predict(
        image,
        verbose=0
    )

    probabilities = predictions[0]

    predicted_index = int(
        np.argmax(probabilities)
    )

    predicted_class = classes[predicted_index]

    confidence = float(
        probabilities[predicted_index] * 100
    )


    # -----------------------------
    # ALL PROBABILITIES
    # -----------------------------
    probability_data = {
        "Cordana": float(probabilities[0] * 100),
        "Healthy": float(probabilities[1] * 100),
        "Panama": float(probabilities[2] * 100),
        "Sigatoka": float(probabilities[3] * 100)
    }
    # -----------------------------
    # AMHARIC DISEASE INFORMATION
    # -----------------------------
    amharic_info = {
        "cordana": {
            "name": "ኮርዳና",
            "description": "ይህ በሽታ በሙዝ ቅጠል ላይ ቡናማ ወይም ጥቁር ነጠብጣቦች እንዲታዩ ሊያደርግ ይችላል።",
            "advice": "የተጠቁ ቅጠሎችን ያስወግዱ፣ የእርሻውን ንጽህና ይጠብቁ እና ቅጠሎች ለረጅም ጊዜ እርጥብ እንዳይሆኑ ያድርጉ።"
        },

        "healthy": {
            "name": "ጤናማ",
            "description": "የተመረመረው የሙዝ ቅጠል ጤናማ የሆነ ቅጠል ይመስላል።",
            "advice": "ተክሉን በመደበኛነት ይከታተሉ፣ በቂ ውሃ እና አስፈላጊ ንጥረ ነገሮች ያቅርቡ።"
        },

        "panama": {
            "name": "ፓናማ በሽታ",
            "description": "ፓናማ በሽታ የሙዝ ተክልን ሊጎዳ የሚችል ከባድ በሽታ ነው። ቅጠሎች ሊደርቁ እና ተክሉ ሊደክም ይችላል።",
            "advice": "የተጠቁ ተክሎችን ከጤናማ ተክሎች ይለዩ። የተበከለ አፈር ወደ ሌሎች የሙዝ እርሻዎች እንዳይሰራጭ ጥንቃቄ ያድርጉ።"
        },

        "sigatoka": {
            "name": "ሲጋቶካ",
            "description": "ሲጋቶካ በሽታ በሙዝ ቅጠል ላይ ትንንሽ ቡናማ ወይም ጥቁር ነጠብጣቦች እና ቁስሎች እንዲፈጠሩ ሊያደርግ ይችላል።",
            "advice": "የተጠቁ ቅጠሎችን ያስወግዱ፣ ተክሎችን በቅርብ ይከታተሉ እና ለትክክለኛ የሕክምና ምክር የእርሻ ባለሙያን ያማክሩ።"
        }
    }

    selected_amharic_info = amharic_info.get(
        predicted_class,
        {
            "name": predicted_class,
            "description": "የበሽታው መረጃ አልተገኘም።",
            "advice": "እባክዎ የእርሻ ባለሙያን ያማክሩ።"
        }
    )

    english_info = {
        "cordana": {
            "name": "Cordana",
            "description": "Cordana can cause brown or black spots on banana leaves.",
            "advice": "Remove infected leaves, keep the farm clean, and avoid keeping leaves wet for long periods."
        },
        "healthy": {
            "name": "Healthy",
            "description": "The analyzed banana leaf appears to be healthy.",
            "advice": "Continue regular monitoring and provide the plant with enough water and nutrients."
        },
        "panama": {
            "name": "Panama Disease",
            "description": "Panama disease is a serious disease that can damage banana plants and cause leaves to dry.",
            "advice": "Separate infected plants from healthy plants and prevent contaminated soil from spreading to other banana fields."
        },
        "sigatoka": {
            "name": "Sigatoka",
            "description": "Sigatoka can cause small brown or black spots and lesions on banana leaves.",
            "advice": "Remove infected leaves, monitor the plants closely, and consult an agricultural professional for treatment advice."
        }
    }

    selected_english_info = english_info.get(
        predicted_class,
        {
            "name": predicted_class.title(),
            "description": "Information about this disease is not available.",
            "advice": "Please consult an agricultural professional."
        }
    )

    # -----------------------------
    # SAVE TO DATABASE
    # -----------------------------
    conn = get_db_connection()

    conn.execute(
        """
        INSERT INTO predictions (
            user_id,
            image_name,
            prediction,
            confidence,
            cordana_probability,
            healthy_probability,
            panama_probability,
            sigatoka_probability
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            session["user_id"],
            unique_filename,
            predicted_class,
            confidence,
            probability_data["Cordana"],
            probability_data["Healthy"],
            probability_data["Panama"],
            probability_data["Sigatoka"]
        )
    )

    conn.commit()
    conn.close()

    return render_template(
        "predict.html",
        prediction=predicted_class,
        confidence=confidence,
        image_filename=unique_filename,
        probabilities=probability_data,
        amharic_info=selected_amharic_info,
        english_info=selected_english_info
    )
# =========================
# PREDICTION HISTORY
# =========================

@app.route("/history")
def history():

    if "user_id" not in session:
        return redirect(url_for("login"))

    conn = get_db_connection()

    predictions = conn.execute(
        """
        SELECT *
        FROM predictions
        WHERE user_id = ?
        ORDER BY created_at DESC
        """,
        (session["user_id"],)
    ).fetchall()

    conn.close()

    return render_template(
        "history.html",
        predictions=predictions
    )

# =========================
# PREDICTION DETAILS
# =========================

@app.route("/prediction/<int:prediction_id>")
def prediction_details(prediction_id):

    if "user_id" not in session:
        return redirect(url_for("login"))

    conn = get_db_connection()

    prediction = conn.execute(
        """
        SELECT *
        FROM predictions
        WHERE id = ?
        AND user_id = ?
        """,
        (
            prediction_id,
            session["user_id"]
        )
    ).fetchone()

    conn.close()

    if prediction is None:
        flash("Prediction not found.")
        return redirect(url_for("history"))

    return render_template(
        "prediction_detials.html",
        prediction=prediction
    )

# =========================
# DELETE PREDICTION
# =========================

@app.route("/delete_prediction/<int:prediction_id>", methods=["POST"])
def delete_prediction(prediction_id):

    if "user_id" not in session:
        return redirect(url_for("login"))
    conn = get_db_connection()

    # First get the prediction and make sure it belongs
    # to the currently logged-in user
    prediction = conn.execute(
        """
        SELECT image_name
        FROM predictions
        WHERE id = ?
        AND user_id = ?
        """,
        (
            prediction_id,
            session["user_id"]
        )
    ).fetchone()

    if prediction is None:
        conn.close()
        flash("Prediction not found.")
        return redirect(url_for("history"))

    # Delete database record
    conn.execute(
        """
        DELETE FROM predictions
        WHERE id = ?
        AND user_id = ?
        """,
        (
            prediction_id,
            session["user_id"]
        )
    )

    conn.commit()
    conn.close()

    flash("Prediction deleted successfully.")

    return redirect(url_for("history"))

# =========================
# RUN APPLICATION
# =========================

if __name__ == "__main__":
    app.run(debug=True)