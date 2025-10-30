import numpy as np
from chess import Board

# gjør om et brett til en tredimensjonal matrise med informasjon om lovlige trekk, hvor brikkene står osv... 
def board_to_matrix(board: Board):
    # 8x8 er størrelsen på sjakkbrettet
    # 12 = antall unike brikker.
    # 13 = brett for lovlig etrekk
    # 14 = brett for der vi er
    matrix = np.zeros((13, 8, 8))
    piece_map = board.piece_map()

    # Lager 12 8x8 brett hvor hvert brett har kun en type, feks hvite bønder, svarte konger osv..
    for square, piece in piece_map.items():
        row, col = divmod(square, 8)
        piece_type = piece.piece_type - 1
        piece_color = 0 if piece.color else 6
        matrix[piece_type + piece_color, row, col] = 1

    # Lag lovlige trekk brettet
    legal_moves = board.legal_moves
    for move in legal_moves:
        to_square = move.to_square
        row_to, col_to = divmod(to_square, 8)
        matrix[12, row_to, col_to] = 1

    return matrix


def create_input_for_nn(games):
    X = []
    y = []
    for game in games:
        board = game.board()
        for move in game.mainline_moves():
            X.append(board_to_matrix(board))
            y.append(move.uci())
            board.push(move)
    return np.array(X, dtype=np.float32), np.array(y)


def encode_moves(moves):
    move_to_int = {move: idx for idx, move in enumerate(set(moves))}
    return np.array([move_to_int[move] for move in moves], dtype=np.float32), move_to_int