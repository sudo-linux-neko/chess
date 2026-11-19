package com.vaibhav.chessstudio.game

import com.github.bhlangonijr.chesslib.Board
import com.github.bhlangonijr.chesslib.Piece
import com.github.bhlangonijr.chesslib.PieceType
import com.github.bhlangonijr.chesslib.Side
import com.github.bhlangonijr.chesslib.Square
import com.github.bhlangonijr.chesslib.move.Move
import kotlin.math.abs

class JvmChessGame : ChessGame {
    private val board = Board()
    private val history = mutableListOf<String>()
    private val previousMoves = mutableListOf<Pair<Int, Int>?>()

    init {
        reset()
    }

    override fun reset() {
        board.loadFromFen(START_FEN)
        history.clear()
        previousMoves.clear()
    }

    override fun fen(): String = board.fen

    override fun sideToMove(): PieceColor =
        if (board.sideToMove == Side.WHITE) PieceColor.WHITE else PieceColor.BLACK

    override fun pieceAt(square: Int): ChessPiece? {
        val piece = board.getPiece(Square.squareAt(square))
        if (piece == null || piece == Piece.NONE) return null
        return ChessPiece(
            color = if (piece.pieceSide == Side.WHITE) PieceColor.WHITE else PieceColor.BLACK,
            kind = piece.pieceType.toKind()
        )
    }

    override fun legalTargets(from: Int): Set<Int> {
        val origin = Square.squareAt(from)
        return board.legalMoves()
            .asSequence()
            .filter { it.from == origin }
            .map { it.to.ordinal }
            .toSet()
    }

    override fun promotionOptions(from: Int, to: Int): List<PieceKind> {
        val origin = Square.squareAt(from)
        val target = Square.squareAt(to)
        return board.legalMoves()
            .asSequence()
            .filter { it.from == origin && it.to == target && it.promotion != Piece.NONE }
            .map { it.promotion.pieceType.toKind() }
            .distinct()
            .toList()
    }

    override fun move(from: Int, to: Int, promotion: PieceKind): MoveResult {
        val origin = Square.squareAt(from)
        val target = Square.squareAt(to)
        val candidates = board.legalMoves().filter { it.from == origin && it.to == target }
        if (candidates.isEmpty()) return MoveResult(false)

        val promotingOptions = candidates.filter { it.promotion != Piece.NONE }
        val chosen = if (promotingOptions.isNotEmpty()) {
            val wanted = Piece.make(board.sideToMove, promotion.toPieceType())
            promotingOptions.firstOrNull { it.promotion == wanted }
                ?: promotingOptions.firstOrNull { it.promotion.pieceType == PieceType.QUEEN }
                ?: promotingOptions.first()
        } else {
            candidates.first()
        }

        val moving = board.getPiece(origin)
        val captured = board.getPiece(target) != Piece.NONE ||
            (moving.pieceType == PieceType.PAWN && from % 8 != to % 8 && board.getPiece(target) == Piece.NONE)
        val isCastle = moving.pieceType == PieceType.KING && abs(from % 8 - to % 8) == 2
        val san = buildSan(moving, origin, target, captured, isCastle, chosen)

        if (!board.doMove(chosen, true)) return MoveResult(false)

        val suffix = when {
            board.isMated() -> "#"
            board.isKingAttacked() -> "+"
            else -> ""
        }
        val fullSan = san + suffix
        history += fullSan
        previousMoves += from to to
        return MoveResult(true, fullSan)
    }

    override fun undo(): Boolean {
        if (history.isEmpty()) return false
        board.undoMove()
        history.removeLastOrNull()
        previousMoves.removeLastOrNull()
        return true
    }

    override fun canUndo(): Boolean = history.isNotEmpty()

    override fun status(): GameStatus = when {
        board.isMated() -> GameStatus.CHECKMATE
        board.isStaleMate() -> GameStatus.STALEMATE
        board.isDraw() -> GameStatus.DRAW
        else -> GameStatus.ONGOING
    }

    override fun inCheck(): Boolean = board.isKingAttacked()

    override fun kingSquare(): Int? =
        board.getKingSquare(board.sideToMove)?.ordinal

    override fun lastMove(): Pair<Int, Int>? = previousMoves.lastOrNull()

    override fun moveHistory(): List<String> = history.toList()

    private fun buildSan(
        moving: Piece,
        from: Square,
        to: Square,
        captured: Boolean,
        isCastle: Boolean,
        move: Move
    ): String {
        if (isCastle) {
            return if (to.ordinal % 8 == 6) "O-O" else "O-O-O"
        }

        val destination = to.toString().lowercase()
        val pieceLetter = when (moving.pieceType) {
            PieceType.PAWN -> ""
            PieceType.KNIGHT -> "N"
            PieceType.BISHOP -> "B"
            PieceType.ROOK -> "R"
            PieceType.QUEEN -> "Q"
            PieceType.KING -> "K"
            else -> ""
        }

        val disambiguation = if (moving.pieceType == PieceType.PAWN || moving.pieceType == PieceType.KING) {
            ""
        } else {
            val alternatives = board.legalMoves()
                .asSequence()
                .filter {
                    it.to == to &&
                        it.from != from &&
                        board.getPiece(it.from).pieceType == moving.pieceType
                }
                .map { it.from }
                .toList()
            if (alternatives.isEmpty()) {
                ""
            } else {
                val sameFile = alternatives.any { it.ordinal % 8 == from.ordinal % 8 }
                val sameRank = alternatives.any { it.ordinal / 8 == from.ordinal / 8 }
                when {
                    !sameFile -> ('a' + from.ordinal % 8).toString()
                    !sameRank -> (from.ordinal / 8 + 1).toString()
                    else -> "${('a' + from.ordinal % 8)}${from.ordinal / 8 + 1}"
                }
            }
        }

        val body = if (moving.pieceType == PieceType.PAWN) {
            if (captured) "${('a' + from.ordinal % 8)}x$destination" else destination
        } else {
            "$pieceLetter$disambiguation${if (captured) "x" else ""}$destination"
        }

        val promotion = move.promotion
        return if (promotion != Piece.NONE) {
            body + "=" + promotion.pieceType.letter()
        } else {
            body
        }
    }

    private fun PieceType.toKind(): PieceKind = when (this) {
        PieceType.PAWN -> PieceKind.PAWN
        PieceType.KNIGHT -> PieceKind.KNIGHT
        PieceType.BISHOP -> PieceKind.BISHOP
        PieceType.ROOK -> PieceKind.ROOK
        PieceType.QUEEN -> PieceKind.QUEEN
        PieceType.KING -> PieceKind.KING
        else -> PieceKind.PAWN
    }

    private fun PieceKind.toPieceType(): PieceType = when (this) {
        PieceKind.PAWN -> PieceType.PAWN
        PieceKind.KNIGHT -> PieceType.KNIGHT
        PieceKind.BISHOP -> PieceType.BISHOP
        PieceKind.ROOK -> PieceType.ROOK
        PieceKind.QUEEN -> PieceType.QUEEN
        PieceKind.KING -> PieceType.KING
    }

    private fun PieceType.letter(): String = when (this) {
        PieceType.KNIGHT -> "N"
        PieceType.BISHOP -> "B"
        PieceType.ROOK -> "R"
        PieceType.QUEEN -> "Q"
        else -> ""
    }

    private companion object {
        const val START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    }
}
