package com.vaibhav.chessstudio.game

interface ChessGame {
    fun reset()
    fun fen(): String
    fun sideToMove(): PieceColor
    fun pieceAt(square: Int): ChessPiece?
    fun legalTargets(from: Int): Set<Int>
    fun promotionOptions(from: Int, to: Int): List<PieceKind>
    fun move(from: Int, to: Int, promotion: PieceKind = PieceKind.QUEEN): MoveResult
    fun undo(): Boolean
    fun canUndo(): Boolean
    fun status(): GameStatus
    fun inCheck(): Boolean
    fun kingSquare(): Int?
    fun lastMove(): Pair<Int, Int>?
    fun moveHistory(): List<String>
}
