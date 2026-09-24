import openai
import chess
import chess.pgn
import io
from stockfish import Stockfish
import os
import platform


class ChessEngine:
    project_root = os.path.dirname(os.path.abspath(__file__))
    stockfish_exec_file = "stockfish-windows-x86-64-avx2.exe"if platform.system() == "Windows" else "stockfish-linux-amd64"
    stockfish_path = os.path.join(project_root, "..", stockfish_exec_file)

    def __init__(self, api_key=None, model=None, session_id=None, stockfish_path=stockfish_path):
        self.move_count = 1
        self.board = chess.Board()
        self.messages = [
            {
                "role": "system",
                "content": (
                    "We are playing a chess game. At every turn, repeat all the moves that have already been made. "
                    "Find the best response for Black. I'm White and the game starts with 1.{first_move}\n\n"
                    "Output format should always be:\n\n"
                    "PGN of game so far: ...\n\n"
                    "Best move: ...\n\n"
                    "Do not include move numbers."
                ),
            }
        ]
        self.api_key = api_key
        self.model = model
        self.session_id = session_id
        self.retry_count = 0
        self._init_stockfish(stockfish_path)

        if platform.system() == "Windows":
            exec_file_name = "stockfish-windows-x86-64-avx2.exe"
        else:
            exec_file_name = "stockfish-linux-amd64"

        self.stockfish_path = stockfish_path or os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            exec_file_name,
        )
        self._init_stockfish(self.stockfish_path)
        self.white_eval = 0.0  # cumulative score from White’s perspective
        self.last_eval = 0.0

    def _init_stockfish(self, stockfish_path):
        self.stockfish = Stockfish(path=stockfish_path, depth=15)
        self.stockfish.set_skill_level(20)

    def __getstate__(self):
        state = self.__dict__.copy()
        if "stockfish" in state:
            del state["stockfish"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._init_stockfish(self.stockfish_path)

    def set_api_key(self):
        openai.api_key = self.api_key

    def update_first_move_message(self, first_move):
        self.messages[0]["content"] = self.messages[0]["content"].format(first_move=first_move)

    def extract_move(self, response):
        try:
            move = response.split("Best move:")[1]
        except IndexError:
            move = response
        move = move.replace("...", "").replace("1.", "").replace(".", "").replace(" ", "")
        return move[:10]

    def evaluate_move_quality(self, san_move):
        temp_board = self.board.copy()

        # Apply the move on a copy
        try:
            move_obj = temp_board.parse_san(san_move)
            temp_board.push(move_obj)
        except Exception:
            # If move invalid, return previous values
            return round(self.white_eval, 2), round(self.last_eval, 2)

        # Eval new position (always from White's perspective)
        self.stockfish.set_fen_position(temp_board.fen())
        eval_after = self.stockfish.get_evaluation()

        # Convert Stockfish evaluation to a positive number
        if eval_after["type"] == "cp":
            absolute = abs(eval_after["value"] / 100.0)
        elif eval_after["type"] == "mate":
            absolute = 100.0  # assign high score for mate
        else:
            absolute = 0.0

        # --- Only accumulate for White moves ---
        if self.board.turn == chess.BLACK:  # True means White just moved
            self.white_eval += absolute

        # Update last_eval for debugging/logging
        self.last_eval = absolute

        # Apply move to real board
        try:
            move_obj_real = self.board.parse_san(san_move)
            self.board.push(move_obj_real)
        except Exception:
            pass

        print(f"[DEBUG] SAN={san_move}, WhiteEval={self.white_eval}, Absolute={absolute}")

        return round(self.white_eval, 2), round(absolute, 2)

    def process_move(self, move_from, move_to, promotion, status, pgn, san):
        # Rebuild the board from PGN
        pgn_game = chess.pgn.read_game(io.StringIO(pgn))
        if pgn_game:
            self.board = pgn_game.end().board()
        else:
            self.board = chess.Board()  # fallback

        self.move_count += 1

        # --- Eval-only handler (for white move) ---
        if status == "eval-only":
            accumulative, absolute = self.evaluate_move_quality(san)
            print(f"[DEBUG] Eval-only SAN={san}, WhiteEval={accumulative}, Absolute={absolute}")
            return san, accumulative, absolute

        # --- Handle illegal move retry ---
        if status == "repeat":
            if self.retry_count >= 3:
                # Stockfish fallback
                self.stockfish.set_fen_position(self.board.fen())
                uci_move = self.stockfish.get_best_move()
                if uci_move:
                    move_obj = chess.Move.from_uci(uci_move)
                    san_move = self.board.san(move_obj)
                    self.board.push(move_obj)
                    self.messages.append({"role": "assistant", "content": f"Best move: {san_move}"})
                    self.retry_count = 0
                    accumulative, absolute = self.evaluate_move_quality(san_move)
                    return san_move, accumulative, absolute
                else:
                    raise ValueError("Stockfish failed to return a move")
            self.retry_count += 1
            self.messages.append({
                "role": "system",
                "content": "The move you suggested was illegal on the current board. Please suggest a legal move."
            })
            response = self.get_gpt_response(self.messages)
            self.messages.append({"role": "assistant", "content": response})
            move_san = self.extract_move(response)
            accumulative, absolute = self.evaluate_move_quality(move_san)
            # self.retry_count = 0
            return move_san, accumulative, absolute

        # --- Normal move processing ---
        self.messages.append({"role": "user", "content": san})
        response = self.get_gpt_response(self.messages)
        self.messages.append({"role": "assistant", "content": response})
        move_san = self.extract_move(response)
        accumulative, absolute = self.evaluate_move_quality(move_san)

        # Handle first move special message
        if self.move_count == 2 and pgn_game and pgn_game.variations:
            first_move_obj = pgn_game.variations[0].move
            initial_board = chess.Board()
            first_san = initial_board.san(first_move_obj)
            self.update_first_move_message(first_san)

        self.retry_count = 0
        return move_san, accumulative, absolute

    def get_gpt_response(self, messages):
        completion = openai.chat.completions.create(model=self.model, messages=messages)
        return completion.choices[0].message.content.strip()

    def evaluate_position(self):
        """Return Stockfish evaluation always from White's perspective (static eval)."""
        self.stockfish.set_fen_position(self.board.fen())
        eval_data = self.stockfish.get_evaluation()

        if eval_data["type"] == "cp":
            score = eval_data["value"] / 100.0
            # if self.board.turn == chess.BLACK:
            #     score = -score
            return round(score, 2)

        elif eval_data["type"] == "mate":
            mate_score = eval_data["value"]
            # if self.board.turn == chess.BLACK:
            #     mate_score = -mate_score
            return f"Mate in {mate_score}"

        return "N/A"
