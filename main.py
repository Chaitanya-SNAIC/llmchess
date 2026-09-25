from engine.engine import ChessEngine, log_openai_usage
import openai
from flask import Flask, render_template, request, session, jsonify
from flask_cors import CORS
from datetime import datetime
import logging
import uuid
import pickle
import os
import sys

# Checked before any setup below, so a direct run doesn't create a log file.
if __name__ == "__main__":
    sys.exit("Don't run main.py directly. Start the app with:\n\n    flask --app main run\n\nAdd --debug for debug mode.")

# --- Local in-memory storage for demo ---
instances = {}

def store_instance(session_id, engine_instance):
    """
    Serialize and store the engine instance locally.
    Stockfish will be recreated automatically on unpickle.
    """
    try:
        instances[session_id] = pickle.dumps(engine_instance)
    except Exception as e:
        print(f"[Error] Failed to store session {session_id}: {e}")

def get_instance(session_id):
    """Retrieve and deserialize the engine instance."""
    data = instances.get(session_id)
    if data is None:
        return None
    try:
        return pickle.loads(data)
    except Exception as e:
        print(f"[Error] Failed to load session {session_id}: {e}")
        return None

def delete_instance(session_id):
    """Delete a stored engine instance."""
    if session_id in instances:
        del instances[session_id]

# --- Flask setup ---
app = Flask(__name__, template_folder=".")
app.secret_key = "sf43d5f4s394jfe2dm903"
CORS(app)

# --- Logging ---
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
logger = logging.getLogger("llmchess")


def setup_logging(debug):
    """Send the "llmchess" loggers to logs/YYYY-MM-DD_HH-MM-SS.log.

    Returns True in the process that owns the run's log file. Under
    `flask run --debug` the reloader imports this module once in a watcher
    process and again in every server process it spawns; the first import picks
    the file name and passes it down through the environment, so one run writes
    one file and its startup lines are logged once.
    """
    log_file = os.environ.get("LLMCHESS_LOG_FILE")
    owner = log_file is None
    if owner:
        name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".log"
        log_file = os.path.join(LOG_DIR, name)
        os.environ["LLMCHESS_LOG_FILE"] = log_file

    try:
        os.makedirs(LOG_DIR, exist_ok=True)
        handler = logging.FileHandler(log_file, encoding="utf-8")
    except OSError:
        # App Engine's filesystem is read-only outside /tmp. Log to stderr there
        # so the app still starts and the lines reach Cloud Logging.
        handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s [%(name)s]",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG if debug else logging.INFO)
    # Keep these lines (the API key among them) out of any handlers something
    # else attaches to the root logger.
    logger.propagate = False
    return owner


DEBUG = app.debug

if setup_logging(DEBUG):
    logger.info("Server starting (debug=%s)", DEBUG)
    logger.debug(
        "Using OpenAI API key: %s", os.environ.get("OPENAI_API_KEY", "<not set>")
    )

# Sessions whose end is already logged, so a game that ends in checkmate and is
# then reset or closed isn't logged as ending twice.
ended_sessions = set()


def log_game_end(session_id, reason):
    if session_id in ended_sessions:
        return
    ended_sessions.add(session_id)
    logger.info("Game ended: session=%s reason=%s", session_id, reason)


@app.route("/new-session")
def new_session():
    session_id = str(uuid.uuid4())
    session["session_id"] = session_id
    api_key = session.get("api_key")
    model = session.get("model")
    store_instance(session_id, ChessEngine(api_key, model, session_id))
    logger.info("Game started: session=%s model=%s", session_id, model)
    return {"session_id": session_id}


@app.route("/end-game", methods=["POST"])
def end_game():
    session_id = session.get("session_id")
    if session_id not in instances:
        return jsonify({"error": "Invalid session"}), 400
    log_game_end(session_id, request.form.get("reason", "unknown"))
    return {"status": "success"}


# POST is for navigator.sendBeacon, which the page sends when its tab closes.
@app.route("/delete-session", methods=["GET", "POST"])
def delete_session():
    session_id = session.get("session_id")
    if get_instance(session_id) is not None:
        log_game_end(session_id, request.args.get("reason", "session deleted"))
        ended_sessions.discard(session_id)
        delete_instance(session_id)
        session.clear()
        return {"status": "success"}
    else:
        print("invalid session in delete-session")
        return {"error": "Invalid session"}


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/move", methods=["POST"])
def move():
    session_id = session.get("session_id")
    engine_instance = get_instance(session_id)
    if engine_instance is None:
        return jsonify({"error": "Invalid session"}), 400

    move_from = request.form.get("from")
    move_to = request.form.get("to")
    promotion = request.form.get("promotion", None)
    status = request.form.get("status", None)
    pgn_data = request.form.get("pgn", "")
    san = request.form.get("san")

    try:
        result, accumulative, absolute = engine_instance.process_move(
            move_from, move_to, promotion, status, pgn_data, san
        )
        store_instance(session_id, engine_instance)
        return jsonify({"move": result, "accumulative": accumulative, 'absolute': absolute})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/set-api-key", methods=["POST"])
def set_api_key():
    # hardcoded API key for testing
    api_key = os.environ["OPENAI_API_KEY"]
    # api_key = request.form.get("api_key")
    model = request.form.get("model")
    session["api_key"] = api_key
    session["model"] = model
    return {"status": "success"}


# Models already confirmed to work with the server's key. The key comes from the
# environment and can't change while the process runs, so one successful check
# per model is enough; repeating it made every Start wait on an OpenAI call.
validated_models = set()


@app.route("/check-api-key", methods=["POST"])
def check_api_key():
    api_key = os.environ["OPENAI_API_KEY"]
    # api_key = request.form.get("api_key")
    model = request.form.get("model")
    openai.api_key = api_key
    if model in validated_models:
        return {"status": "success"}
    print(f"Checking API key for model: {model}")  # debug
    try:
        response = openai.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": "Hello"}],
            max_completion_tokens=5,
        )
        print("API key valid, response:", response)  # debug
        log_openai_usage(response, "key check")
        validated_models.add(model)
        return {"status": "success"}
    except Exception as e:
        print("API key check failed:", e)  # debug
        return {"status": "failure", "message": str(e)}

# @app.route("/check-api-key", methods=["POST"])
# def check_api_key():
#     api_key = request.form.get("api_key")
#     openai.api_key = api_key
#     try:
#         openai.Engine.list()
#         return {"status": "success"}
#     except openai.error.AuthenticationError:
#         return {"status": "failure"}
