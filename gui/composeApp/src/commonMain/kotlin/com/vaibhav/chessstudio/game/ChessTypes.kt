package com.vaibhav.chessstudio.game

enum class PieceColor {
    WHITE,
    BLACK;

    fun opposite(): PieceColor = if (this == WHITE) BLACK else WHITE
}

enum class PieceKind {
    PAWN,
    KNIGHT,
    BISHOP,
    ROOK,
    QUEEN,
    KING
}

data class ChessPiece(
    val color: PieceColor,
    val kind: PieceKind
)

enum class GameStatus {
    ONGOING,
    CHECKMATE,
    STALEMATE,
    DRAW
}

data class MoveResult(
    val success: Boolean,
    val san: String = ""
)
