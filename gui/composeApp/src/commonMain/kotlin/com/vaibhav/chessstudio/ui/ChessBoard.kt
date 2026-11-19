package com.vaibhav.chessstudio.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Shadow
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.vaibhav.chessstudio.game.ChessGame
import com.vaibhav.chessstudio.game.PieceColor
import com.vaibhav.chessstudio.game.PieceKind

@Composable
fun ChessBoard(
    game: ChessGame,
    flipped: Boolean,
    selected: Int?,
    legalTargets: Set<Int>,
    lastMove: Pair<Int, Int>?,
    inCheck: Boolean,
    kingSquare: Int?,
    onSquareClick: (Int) -> Unit,
    modifier: Modifier = Modifier
) {
    BoxWithConstraints(
        modifier = modifier
            .clip(RoundedCornerShape(20.dp))
            .border(1.dp, StudioBorder, RoundedCornerShape(20.dp))
    ) {
        val cell = maxWidth / 8
        val density = LocalDensity.current
        val pieceSize = with(density) { (cell * 0.72f).toSp() }
        val coordinateSize = with(density) { (cell * 0.15f).toSp() }
        val showCoordinates = cell >= 42.dp

        Column(Modifier.fillMaxSize()) {
            for (row in 0..7) {
                Row(Modifier.weight(1f).fillMaxWidth()) {
                    for (col in 0..7) {
                        val square = if (!flipped) {
                            (7 - row) * 8 + col
                        } else {
                            row * 8 + (7 - col)
                        }
                        val rank = square / 8
                        val file = square % 8
                        val light = (rank + file) % 2 == 1
                        val piece = game.pieceAt(square)
                        val isSelected = selected == square
                        val isTarget = square in legalTargets
                        val isLast = lastMove?.first == square || lastMove?.second == square
                        val isCheckedKing = inCheck && kingSquare == square

                        val background = when {
                            isCheckedKing -> Color(0xFFC85A5A)
                            isSelected -> Color(0xFFD8A640)
                            isLast -> if (light) Color(0xFFD9BE83) else Color(0xFFB58A5C)
                            light -> BoardLight
                            else -> BoardDark
                        }

                        Box(
                            modifier = Modifier
                                .weight(1f)
                                .fillMaxHeight()
                                .background(background)
                                .clickable { onSquareClick(square) }
                                .then(
                                    if (isSelected) {
                                        Modifier.border(3.dp, StudioGold)
                                    } else {
                                        Modifier
                                    }
                                ),
                            contentAlignment = Alignment.Center
                        ) {
                            if (piece != null) {
                                Text(
                                    text = pieceGlyph(piece.kind),
                                    fontSize = pieceSize,
                                    lineHeight = pieceSize,
                                    fontFamily = FontFamily.Serif,
                                    style = TextStyle(
                                        fontWeight = FontWeight.Normal,
                                        shadow = Shadow(
                                            color = if (piece.color == PieceColor.WHITE) {
                                                Color(0xAA20170C)
                                            } else {
                                                Color(0x88F6EEDC)
                                            },
                                            offset = Offset(0.6f, 1.2f),
                                            blurRadius = 2.2f
                                        )
                                    ),
                                    color = if (piece.color == PieceColor.WHITE) {
                                        Color(0xFFF9F3E4)
                                    } else {
                                        Color(0xFF171A20)
                                    }
                                )
                            }

                            if (isTarget) {
                                if (piece == null) {
                                    Box(
                                        Modifier
                                            .size(cell * 0.18f)
                                            .clip(CircleShape)
                                            .background(StudioTeal.copy(alpha = 0.78f))
                                    )
                                } else {
                                    Box(
                                        Modifier
                                            .fillMaxSize(0.78f)
                                            .border(3.dp, StudioTeal.copy(alpha = 0.85f), CircleShape)
                                    )
                                }
                            }

                            if (showCoordinates) {
                                if (row == 7) {
                                    val fileName = if (flipped) 'h' - col else 'a' + col
                                    Text(
                                        text = fileName.toString(),
                                        fontSize = coordinateSize,
                                        color = if (light) BoardDark.copy(alpha = 0.72f) else BoardLight.copy(alpha = 0.72f),
                                        modifier = Modifier
                                            .align(Alignment.BottomEnd)
                                            .padding(3.dp)
                                    )
                                }
                                if (col == 0) {
                                    val rankName = if (flipped) row + 1 else 8 - row
                                    Text(
                                        text = rankName.toString(),
                                        fontSize = coordinateSize,
                                        color = if (light) BoardDark.copy(alpha = 0.72f) else BoardLight.copy(alpha = 0.72f),
                                        modifier = Modifier
                                            .align(Alignment.TopStart)
                                            .padding(3.dp)
                                    )
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

private fun pieceGlyph(kind: PieceKind): String = when (kind) {
    PieceKind.PAWN -> "\u265F"
    PieceKind.KNIGHT -> "\u265E"
    PieceKind.BISHOP -> "\u265D"
    PieceKind.ROOK -> "\u265C"
    PieceKind.QUEEN -> "\u265B"
    PieceKind.KING -> "\u265A"
}
