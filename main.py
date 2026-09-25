from engine.engine import ChessEngine
import openai
from flask import Flask, render_template, request, session, jsonify
from flask_cors import CORS
import uuid
import pickle
import os

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


@app.route("/new-session")
def new_session():
    session_id = str(uuid.uuid4())
    session["session_id"] = session_id
    api_key = session.get("api_key")
    model = session.get("model")
    store_instance(session_id, ChessEngine(api_key, model, session_id))
    print(f"New session created: {session_id}, model: {model}")
    return {"session_id": session_id}


@app.route("/delete-session")
def delete_session():
    session_id = session.get("session_id")
    if get_instance(session_id) is not None:
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


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=81, debug=True)
