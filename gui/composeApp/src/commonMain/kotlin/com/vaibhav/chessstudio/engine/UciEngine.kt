package com.vaibhav.chessstudio.engine

import kotlinx.coroutines.flow.StateFlow

data class EngineState(
    val ready: Boolean = false,
    val thinking: Boolean = false,
    val depth: Int = 0,
    val scoreCp: Int? = null,
    val mateIn: Int? = null,
    val pv: List<String> = emptyList(),
    val bestMove: String? = null,
    val bestMoveFen: String? = null,
    val info: String = "Starting Stockfish...",
    val error: String? = null
) {
    val scoreText: String
        get() {
            val mate = mateIn
            if (mate != null) {
                return if (mate > 0) "M$mate" else "M${-mate}"
            }
            val score = scoreCp ?: return "0.00"
            val whole = kotlin.math.abs(score) / 100
            val fraction = kotlin.math.abs(score) % 100
            val sign = if (score >= 0) "+" else "-"
            return "$sign$whole.${fraction.toString().padStart(2, '0')}"
        }
}

interface UciEngine {
    val state: StateFlow<EngineState>

    fun start()
    fun newGame()
    fun go(fen: String, depth: Int)
    fun stop()
    fun close()
}
