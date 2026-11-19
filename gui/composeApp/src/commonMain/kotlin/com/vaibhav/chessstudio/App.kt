package com.vaibhav.chessstudio

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawing
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.windowInsetsPadding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Slider
import androidx.compose.material3.Surface
import androidx.compose.material3.Switch
import androidx.compose.material3.SwitchDefaults
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalClipboardManager
import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.vaibhav.chessstudio.engine.EngineState
import com.vaibhav.chessstudio.game.ChessGame
import com.vaibhav.chessstudio.game.GameStatus
import com.vaibhav.chessstudio.game.MoveResult
import com.vaibhav.chessstudio.game.PieceColor
import com.vaibhav.chessstudio.game.PieceKind
import com.vaibhav.chessstudio.ui.ChessBoard
import com.vaibhav.chessstudio.ui.ChessStudioTheme
import com.vaibhav.chessstudio.ui.KeyValueRow
import com.vaibhav.chessstudio.ui.Metric
import com.vaibhav.chessstudio.ui.StatusPill
import com.vaibhav.chessstudio.ui.StudioBorder
import com.vaibhav.chessstudio.ui.StudioCard
import com.vaibhav.chessstudio.ui.StudioDeep
import com.vaibhav.chessstudio.ui.StudioGold
import com.vaibhav.chessstudio.ui.StudioMuted
import com.vaibhav.chessstudio.ui.StudioNavy
import com.vaibhav.chessstudio.ui.StudioOutlineButton
import com.vaibhav.chessstudio.ui.StudioPrimaryButton
import com.vaibhav.chessstudio.ui.StudioTeal
import com.vaibhav.chessstudio.ui.StudioText
import kotlin.math.roundToInt

@Composable
fun App() {
    ChessStudioTheme {
        val game = remember { createChessGame() }
        val engine = remember { createEngine() }
        val engineState by engine.state.collectAsState()
        val clipboard = LocalClipboardManager.current

        var version by remember { mutableIntStateOf(0) }
        var humanColor by remember { mutableStateOf(PieceColor.WHITE) }
        var flipped by remember { mutableStateOf(false) }
        var selected by remember { mutableStateOf<Int?>(null) }
        var pendingPromotion by remember { mutableStateOf<Pair<Int, Int>?>(null) }
        var depth by remember { mutableIntStateOf(10) }
        var engineEnabled by remember { mutableStateOf(true) }
        var lastSan by remember { mutableStateOf("") }
        var lastRequestedFen by remember { mutableStateOf<String?>(null) }

        val fen = remember(version) { game.fen() }
        val side = remember(version) { game.sideToMove() }
        val status = remember(version) { game.status() }
        val inCheck = remember(version) { game.inCheck() }
        val kingSquare = remember(version) { game.kingSquare() }
        val lastMove = remember(version) { game.lastMove() }
        val history = remember(version) { game.moveHistory() }

        DisposableEffect(Unit) {
            engine.start()
            onDispose { engine.close() }
        }

        fun startNew(color: PieceColor) {
            engine.stop()
            game.reset()
            selected = null
            pendingPromotion = null
            lastSan = ""
            lastRequestedFen = null
            humanColor = color
            flipped = color == PieceColor.BLACK
            version++
            engine.newGame()
        }

        fun playHuman(from: Int, to: Int, promotion: PieceKind) {
            val result = game.move(from, to, promotion)
            if (result.success) {
                lastSan = result.san
                selected = null
                pendingPromotion = null
                lastRequestedFen = null
                version++
            }
        }

        fun applyEngineMove(uci: String): Boolean {
            val result = uciMove(game, uci)
            if (result.success) {
                lastSan = result.san
                selected = null
                lastRequestedFen = null
                version++
                return true
            }
            return false
        }

        fun undoMove() {
            engine.stop()
            if (game.undo()) {
                while (game.sideToMove() != humanColor && game.canUndo()) {
                    game.undo()
                }
                selected = null
                pendingPromotion = null
                lastRequestedFen = null
                lastSan = game.moveHistory().lastOrNull() ?: ""
                version++
            }
        }

        fun onSquareClick(square: Int) {
            if (status != GameStatus.ONGOING || side != humanColor) return

            val current = selected
            if (current == null) {
                if (game.pieceAt(square)?.color == humanColor) selected = square
                return
            }

            if (square == current) {
                selected = null
                return
            }

            if (square in game.legalTargets(current)) {
                val promotions = game.promotionOptions(current, square)
                if (promotions.isNotEmpty()) {
                    pendingPromotion = current to square
                } else {
                    playHuman(current, square, PieceKind.QUEEN)
                }
            } else {
                selected = if (game.pieceAt(square)?.color == humanColor) square else null
            }
        }

        LaunchedEffect(fen, engineState.ready, humanColor, engineState.thinking, status, engineEnabled, engineState.error) {
            if (
                engineEnabled &&
                engineState.ready &&
                engineState.error == null &&
                status == GameStatus.ONGOING &&
                side != humanColor &&
                !engineState.thinking &&
                lastRequestedFen != fen
            ) {
                lastRequestedFen = fen
                engine.go(fen, depth)
            }
        }

        LaunchedEffect(engineState.bestMove, engineState.bestMoveFen, version) {
            val best = engineState.bestMove
            val bestFen = engineState.bestMoveFen
            if (
                engineEnabled &&
                best != null &&
                bestFen == game.fen() &&
                game.status() == GameStatus.ONGOING &&
                game.sideToMove() != humanColor
            ) {
                applyEngineMove(best)
            }
        }

        BoxWithConstraints(
            Modifier
                .fillMaxSize()
                .background(
                    Brush.linearGradient(
                        listOf(StudioNavy, StudioDeep, Color(0xFF101A2D))
                    )
                )
                .windowInsetsPadding(WindowInsets.safeDrawing)
        ) {
            val isWide = maxWidth >= 900.dp
            val pagePadding = if (isWide) 22.dp else 13.dp

            Column(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(horizontal = pagePadding, vertical = if (isWide) 18.dp else 10.dp)
            ) {
                Header(engineState, engineEnabled)

                Spacer(Modifier.height(if (isWide) 16.dp else 10.dp))

                if (isWide) {
                    Row(
                        modifier = Modifier.fillMaxSize(),
                        horizontalArrangement = Arrangement.spacedBy(18.dp)
                    ) {
                        BoardArea(
                            game = game,
                            flipped = flipped,
                            selected = selected,
                            side = side,
                            status = status,
                            inCheck = inCheck,
                            kingSquare = kingSquare,
                            lastMove = lastMove,
                            onSquareClick = { square -> onSquareClick(square) },
                            modifier = Modifier
                                .weight(1f)
                                .fillMaxHeight()
                        )

                        Sidebar(
                            game = game,
                            engineState = engineState,
                            humanColor = humanColor,
                            flipped = flipped,
                            depth = depth,
                            engineEnabled = engineEnabled,
                            lastSan = lastSan,
                            history = history,
                            status = status,
                            side = side,
                            onNewGame = { startNew(humanColor) },
                            onPlayAs = { color -> startNew(color) },
                            onFlip = { flipped = !flipped },
                            onUndo = { undoMove() },
                            onCopyFen = { clipboard.setText(AnnotatedString(game.fen())) },
                            onDepth = { depth = it },
                            onEngineEnabled = {
                                engineEnabled = it
                                if (!it) engine.stop()
                                lastRequestedFen = null
                            },
                            modifier = Modifier
                                .width(370.dp)
                                .fillMaxHeight()
                                .verticalScroll(rememberScrollState())
                        )
                    }
                } else {
                    Column(
                        modifier = Modifier
                            .fillMaxSize()
                            .verticalScroll(rememberScrollState())
                    ) {
                        BoardArea(
                            game = game,
                            flipped = flipped,
                            selected = selected,
                            side = side,
                            status = status,
                            inCheck = inCheck,
                            kingSquare = kingSquare,
                            lastMove = lastMove,
                            onSquareClick = { square -> onSquareClick(square) },
                            modifier = Modifier.fillMaxWidth()
                        )

                        Spacer(Modifier.height(12.dp))

                        Sidebar(
                            game = game,
                            engineState = engineState,
                            humanColor = humanColor,
                            flipped = flipped,
                            depth = depth,
                            engineEnabled = engineEnabled,
                            lastSan = lastSan,
                            history = history,
                            status = status,
                            side = side,
                            onNewGame = { startNew(humanColor) },
                            onPlayAs = { color -> startNew(color) },
                            onFlip = { flipped = !flipped },
                            onUndo = { undoMove() },
                            onCopyFen = { clipboard.setText(AnnotatedString(game.fen())) },
                            onDepth = { depth = it },
                            onEngineEnabled = {
                                engineEnabled = it
                                if (!it) engine.stop()
                                lastRequestedFen = null
                            },
                            modifier = Modifier.fillMaxWidth()
                        )
                    }
                }
            }
        }

        pendingPromotion?.let { (from, to) ->
            PromotionDialog(
                color = game.pieceAt(from)?.color ?: humanColor,
                onDismiss = { pendingPromotion = null },
                onChoose = { kind -> playHuman(from, to, kind) }
            )
        }
    }
}

@Composable
private fun Header(engineState: EngineState, engineEnabled: Boolean) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        verticalAlignment = Alignment.CenterVertically
    ) {
        Box(
            modifier = Modifier
                .size(44.dp)
                .clip(RoundedCornerShape(14.dp))
                .background(
                    Brush.linearGradient(listOf(StudioGold, Color(0xFFF1D18D)))
                ),
            contentAlignment = Alignment.Center
        ) {
            Text(
                text = "\u265E",
                fontSize = 27.sp,
                color = Color(0xFF1B160A),
                fontFamily = FontFamily.Serif
            )
        }

        Spacer(Modifier.width(12.dp))

        Column(Modifier.weight(1f)) {
            Text(
                "Chess Studio",
                style = MaterialTheme.typography.headlineSmall,
                color = StudioText,
                maxLines = 1
            )
            Text(
                "Stockfish 16.1  ·  final.nnue",
                fontSize = 11.sp,
                color = StudioMuted,
                maxLines = 1,
                overflow = TextOverflow.Ellipsis
            )
        }

        val label = when {
            !engineEnabled -> "Engine off"
            engineState.error != null -> "Engine error"
            engineState.thinking -> "Thinking"
            engineState.ready -> "Ready"
            else -> "Starting"
        }
        val tint = when {
            !engineEnabled -> StudioMuted
            engineState.error != null -> Color(0xFFFF7777)
            engineState.thinking -> StudioGold
            engineState.ready -> StudioTeal
            else -> StudioMuted
        }
        StatusPill(label, tint)
    }
}

@Composable
private fun BoardArea(
    game: ChessGame,
    flipped: Boolean,
    selected: Int?,
    side: PieceColor,
    status: GameStatus,
    inCheck: Boolean,
    kingSquare: Int?,
    lastMove: Pair<Int, Int>?,
    onSquareClick: (Int) -> Unit,
    modifier: Modifier = Modifier
) {
    val targets = if (selected == null) emptySet() else game.legalTargets(selected)

    BoxWithConstraints(modifier, contentAlignment = Alignment.Center) {
        val available = if (maxHeight == Dp.Infinity) maxWidth else minOf(maxWidth, maxHeight)
        val boardSize = available.coerceAtMost(760.dp)
        Column(horizontalAlignment = Alignment.CenterHorizontally) {
            ChessBoard(
                game = game,
                flipped = flipped,
                selected = selected,
                legalTargets = targets,
                lastMove = lastMove,
                inCheck = inCheck,
                kingSquare = kingSquare,
                onSquareClick = onSquareClick,
                modifier = Modifier.size(boardSize)
            )
            if (status == GameStatus.CHECKMATE) {
                Spacer(Modifier.height(8.dp))
                Text(
                    text = "Checkmate",
                    color = StudioGold,
                    fontWeight = FontWeight.SemiBold,
                    fontSize = 13.sp
                )
            }
        }
    }
}

@Composable
private fun Sidebar(
    game: ChessGame,
    engineState: EngineState,
    humanColor: PieceColor,
    flipped: Boolean,
    depth: Int,
    engineEnabled: Boolean,
    lastSan: String,
    history: List<String>,
    status: GameStatus,
    side: PieceColor,
    onNewGame: () -> Unit,
    onPlayAs: (PieceColor) -> Unit,
    onFlip: () -> Unit,
    onUndo: () -> Unit,
    onCopyFen: () -> Unit,
    onDepth: (Int) -> Unit,
    onEngineEnabled: (Boolean) -> Unit,
    modifier: Modifier = Modifier
) {
    Column(modifier, verticalArrangement = Arrangement.spacedBy(12.dp)) {
        StudioCard(title = "Game") {
            Row(
                Modifier.fillMaxWidth(),
                verticalAlignment = Alignment.CenterVertically
            ) {
                Column(Modifier.weight(1f)) {
                    Text(
                        when (status) {
                            GameStatus.ONGOING -> "${side.name.lowercase().replaceFirstChar { it.uppercase() }} to move"
                            GameStatus.CHECKMATE -> "Checkmate"
                            GameStatus.STALEMATE -> "Stalemate"
                            GameStatus.DRAW -> "Draw"
                        },
                        fontSize = 17.sp,
                        fontWeight = FontWeight.SemiBold,
                        color = StudioText
                    )
                    Text(
                        if (game.inCheck()) "King is in check" else "You are ${humanColor.name.lowercase()}",
                        fontSize = 12.sp,
                        color = if (game.inCheck()) StudioGold else StudioMuted
                    )
                }
                StatusPill(
                    if (side == PieceColor.WHITE) "White" else "Black",
                    if (side == PieceColor.WHITE) Color(0xFFF4EBD6) else StudioTeal
                )
            }

            Row(
                Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(18.dp)
            ) {
                Metric("Move", "${(history.size / 2) + 1}", Modifier.weight(1f))
                Metric("Eval", engineState.scoreText, Modifier.weight(1f))
                Metric("Depth", if (engineState.depth == 0) "—" else engineState.depth.toString(), Modifier.weight(1f))
            }

            KeyValueRow("Last move", lastSan.ifBlank { "—" })
            KeyValueRow("FEN", game.fen().take(28) + if (game.fen().length > 28) "…" else "")
        }

        StudioCard(title = "Engine") {
            Row(
                Modifier.fillMaxWidth(),
                verticalAlignment = Alignment.CenterVertically
            ) {
                StatusPill(
                    when {
                        engineState.error != null -> "Error"
                        engineState.thinking -> "Thinking"
                        engineState.ready -> "Ready"
                        else -> "Starting"
                    },
                    when {
                        engineState.error != null -> Color(0xFFFF7777)
                        engineState.thinking -> StudioGold
                        engineState.ready -> StudioTeal
                        else -> StudioMuted
                    }
                )
                Spacer(Modifier.weight(1f))
                Text(
                    "SF 16.1 · small NNUE",
                    fontSize = 10.sp,
                    color = StudioMuted,
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis
                )
            }

            Row(
                Modifier.fillMaxWidth(),
                verticalAlignment = Alignment.Bottom
            ) {
                Column(Modifier.weight(1f)) {
                    Text("EVALUATION", fontSize = 10.sp, color = StudioMuted, letterSpacing = 1.sp)
                    Text(
                        engineState.scoreText,
                        fontSize = 30.sp,
                        fontWeight = FontWeight.Bold,
                        color = StudioGold,
                        fontFamily = FontFamily.Monospace
                    )
                }
                Column(horizontalAlignment = Alignment.End) {
                    Text("DEPTH", fontSize = 10.sp, color = StudioMuted, letterSpacing = 1.sp)
                    Text(
                        if (engineState.depth == 0) "—" else engineState.depth.toString(),
                        fontSize = 22.sp,
                        fontWeight = FontWeight.SemiBold,
                        fontFamily = FontFamily.Monospace
                    )
                }
            }

            Text(
                engineState.pv.take(7).joinToString(" ").ifBlank { "Waiting for the engine..." },
                fontSize = 11.sp,
                color = StudioMuted,
                fontFamily = FontFamily.Monospace,
                maxLines = 3,
                overflow = TextOverflow.Ellipsis
            )

            engineState.error?.let {
                Text(
                    it,
                    color = Color(0xFFFF9A9A),
                    fontSize = 11.sp,
                    maxLines = 3,
                    overflow = TextOverflow.Ellipsis
                )
            }
        }

        StudioCard(title = "Play") {
            Row(
                Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                ChoiceChip("White", humanColor == PieceColor.WHITE, Modifier.weight(1f)) {
                    onPlayAs(PieceColor.WHITE)
                }
                ChoiceChip("Black", humanColor == PieceColor.BLACK, Modifier.weight(1f)) {
                    onPlayAs(PieceColor.BLACK)
                }
            }

            Row(
                Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                StudioPrimaryButton("New game", Modifier.weight(1f), onClick = onNewGame)
                StudioOutlineButton("Undo", Modifier.weight(1f), enabled = game.canUndo(), onClick = onUndo)
            }

            Row(
                Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                StudioOutlineButton(if (flipped) "Board: black" else "Board: white", Modifier.weight(1f), onClick = onFlip)
                StudioOutlineButton("Copy FEN", Modifier.weight(1f), onClick = onCopyFen)
            }

            Row(
                Modifier.fillMaxWidth(),
                verticalAlignment = Alignment.CenterVertically
            ) {
                Column(Modifier.weight(1f)) {
                    Text("Engine opponent", fontSize = 13.sp, color = StudioText)
                    Text("Stockfish picks the reply", fontSize = 10.sp, color = StudioMuted)
                }
                Switch(
                    checked = engineEnabled,
                    onCheckedChange = onEngineEnabled,
                    colors = SwitchDefaults.colors(
                        checkedThumbColor = Color(0xFF1B160A),
                        checkedTrackColor = StudioGold,
                        uncheckedTrackColor = StudioDeep
                    )
                )
            }

            Row(
                Modifier.fillMaxWidth(),
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text("Strength", fontSize = 13.sp, color = StudioText, modifier = Modifier.weight(1f))
                Text("depth $depth", fontSize = 12.sp, color = StudioGold, fontFamily = FontFamily.Monospace)
            }
            Slider(
                value = depth.toFloat(),
                onValueChange = { onDepth(it.roundToInt()) },
                valueRange = 6f..20f,
                steps = 13
            )
        }

        StudioCard(title = "Moves") {
            if (history.isEmpty()) {
                Text("No moves yet. White begins.", color = StudioMuted, fontSize = 12.sp)
            } else {
                Column(
                    Modifier
                        .fillMaxWidth()
                        .heightIn(max = 170.dp),
                    verticalArrangement = Arrangement.spacedBy(5.dp)
                ) {
                    val start = maxOf(0, history.size - 16).let { it - it % 2 }
                    history.drop(start).chunked(2).forEachIndexed { index, pair ->
                        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                            val number = start / 2 + index + 1
                            Text(
                                "$number.",
                                color = StudioMuted,
                                fontSize = 11.sp,
                                fontFamily = FontFamily.Monospace,
                                modifier = Modifier.width(28.dp)
                            )
                            Text(
                                pair.getOrNull(0) ?: "",
                                color = StudioText,
                                fontSize = 12.sp,
                                fontFamily = FontFamily.Monospace,
                                modifier = Modifier.weight(1f)
                            )
                            Text(
                                pair.getOrNull(1) ?: "",
                                color = StudioMuted,
                                fontSize = 12.sp,
                                fontFamily = FontFamily.Monospace,
                                modifier = Modifier.weight(1f)
                            )
                        }
                    }
                }
            }
            Text(
                "The trained small net is active in simplified positions. The official big net is bundled for normal positions.",
                fontSize = 9.sp,
                color = StudioMuted,
                lineHeight = 12.sp
            )
        }
    }
}

@Composable
private fun ChoiceChip(
    text: String,
    selected: Boolean,
    modifier: Modifier = Modifier,
    onClick: () -> Unit
) {
    Box(
        modifier = modifier
            .height(38.dp)
            .clip(RoundedCornerShape(11.dp))
            .background(if (selected) StudioGold.copy(alpha = 0.16f) else Color.Transparent)
            .border(
                1.dp,
                if (selected) StudioGold.copy(alpha = 0.7f) else StudioBorder,
                RoundedCornerShape(11.dp)
            )
            .clickable(onClick = onClick),
        contentAlignment = Alignment.Center
    ) {
        Text(
            text,
            fontSize = 12.sp,
            color = if (selected) StudioGold else StudioMuted,
            fontWeight = if (selected) FontWeight.SemiBold else FontWeight.Normal
        )
    }
}

@Composable
private fun PromotionDialog(
    color: PieceColor,
    onDismiss: () -> Unit,
    onChoose: (PieceKind) -> Unit
) {
    val options = listOf(
        PieceKind.QUEEN to "\u265B",
        PieceKind.ROOK to "\u265C",
        PieceKind.BISHOP to "\u265D",
        PieceKind.KNIGHT to "\u265E"
    )

    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text("Promote ${color.name.lowercase()} pawn") },
        text = {
            Row(
                Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                options.forEach { (kind, glyph) ->
                    Surface(
                        modifier = Modifier
                            .weight(1f)
                            .height(64.dp)
                            .clickable { onChoose(kind) },
                        shape = RoundedCornerShape(12.dp),
                        color = Color(0xFF1B2940),
                        border = androidx.compose.foundation.BorderStroke(1.dp, StudioBorder)
                    ) {
                        Box(contentAlignment = Alignment.Center) {
                            Text(
                                glyph,
                                fontSize = 32.sp,
                                fontFamily = FontFamily.Serif,
                                color = if (color == PieceColor.WHITE) Color(0xFFF9F3E4) else Color(0xFF171A20)
                            )
                        }
                    }
                }
            }
        },
        confirmButton = {
            TextButton(onClick = onDismiss) { Text("Cancel") }
        }
    )
}

private fun uciMove(game: ChessGame, uci: String): MoveResult {
    if (uci.length < 4) return MoveResult(false)
    val fromFile = uci[0] - 'a'
    val fromRank = uci[1] - '1'
    val toFile = uci[2] - 'a'
    val toRank = uci[3] - '1'
    if (fromFile !in 0..7 || fromRank !in 0..7 || toFile !in 0..7 || toRank !in 0..7) {
        return MoveResult(false)
    }
    val promotion = when (uci.getOrNull(4)) {
        'q' -> PieceKind.QUEEN
        'r' -> PieceKind.ROOK
        'b' -> PieceKind.BISHOP
        'n' -> PieceKind.KNIGHT
        else -> PieceKind.QUEEN
    }
    return game.move(fromRank * 8 + fromFile, toRank * 8 + toFile, promotion)
}
