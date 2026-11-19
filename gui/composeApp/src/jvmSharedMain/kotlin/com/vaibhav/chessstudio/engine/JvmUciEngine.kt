package com.vaibhav.chessstudio.engine

import java.io.BufferedReader
import java.io.BufferedWriter
import java.io.File
import java.util.concurrent.atomic.AtomicBoolean
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update

class JvmUciEngine(
    private val engineFile: File?,
    private val smallNet: File?,
    private val bigNet: File?,
    private val workDir: File
) : UciEngine {
    private val _state = MutableStateFlow(EngineState())
    override val state: StateFlow<EngineState> = _state.asStateFlow()

    private val lock = Any()
    private val started = AtomicBoolean(false)
    private val closing = AtomicBoolean(false)
    private var process: Process? = null
    private var writer: BufferedWriter? = null
    private var currentFen: String? = null
    private var pendingGo: Pair<String, Int>? = null

    override fun start() {
        if (!started.compareAndSet(false, true)) return

        val file = engineFile
        if (file == null || !file.isFile) {
            _state.update {
                it.copy(
                    info = "Stockfish is not bundled",
                    error = "Engine binary missing. Build with the GitHub Actions workflow."
                )
            }
            return
        }

        Thread({ runEngine(file) }, "stockfish-uci").apply {
            isDaemon = true
            start()
        }
    }

    private fun runEngine(file: File) {
        try {
            workDir.mkdirs()
            val builder = ProcessBuilder(file.absolutePath)
                .directory(workDir)
                .redirectErrorStream(true)
            val running = builder.start()
            process = running
            val output = running.outputStream.bufferedWriter()
            writer = output
            send("uci")

            readLoop(running)
        } catch (t: Throwable) {
            if (!closing.get()) {
                _state.update {
                    it.copy(
                        ready = false,
                        thinking = false,
                        info = "Engine stopped",
                        error = t.message ?: t::class.simpleName ?: "Unknown engine error"
                    )
                }
            }
        }
    }

    private fun readLoop(running: Process) {
        val reader: BufferedReader = running.inputStream.bufferedReader()
        try {
            reader.useLines { lines ->
                lines.forEach { line ->
                    if (line.isNotBlank()) handleLine(line.trim())
                }
            }
        } finally {
            if (!closing.get()) {
                _state.update {
                    it.copy(
                        ready = false,
                        thinking = false,
                        info = "Engine exited",
                        error = "Stockfish exited unexpectedly."
                    )
                }
            }
        }
    }

    private fun handleLine(line: String) {
        when {
            line == "uciok" -> {
                val threads = Runtime.getRuntime().availableProcessors().coerceIn(1, 4)
                send("setoption name Threads value $threads")
                send("setoption name Hash value 128")

                bigNet?.takeIf { it.isFile }?.let {
                    send("setoption name EvalFile value ${it.absolutePath}")
                }
                smallNet?.takeIf { it.isFile }?.let {
                    send("setoption name EvalFileSmall value ${it.absolutePath}")
                }
                send("isready")
            }

            line == "readyok" -> {
                _state.update {
                    it.copy(
                        ready = true,
                        info = "Stockfish ready",
                        error = null
                    )
                }
                val request = synchronized(lock) {
                    pendingGo.also { pendingGo = null }
                }
                request?.let { goNow(it.first, it.second) }
            }

            line.startsWith("info string") -> Unit
            line.startsWith("info ") -> parseInfo(line)

            line.startsWith("bestmove") -> {
                val move = line.split(' ').getOrNull(1)?.takeIf { it != "(none)" }
                _state.update {
                    it.copy(
                        thinking = false,
                        bestMove = move,
                        bestMoveFen = currentFen,
                        info = if (move == null) "No move" else "Best move $move"
                    )
                }
            }

            line.contains("error", ignoreCase = true) ||
                line.contains("NNUE evaluation parameters", ignoreCase = true) -> {
                _state.update { it.copy(error = line) }
            }
        }
    }

    private fun parseInfo(line: String) {
        val tokens = line.split(Regex("\\s+"))
        var depth = _state.value.depth
        var scoreCp = _state.value.scoreCp
        var mateIn = _state.value.mateIn
        var pv: List<String> = _state.value.pv
        var index = 1

        while (index < tokens.size) {
            when (tokens[index]) {
                "depth" -> {
                    depth = tokens.getOrNull(index + 1)?.toIntOrNull() ?: depth
                    index += 2
                }

                "multipv" -> {
                    val multipv = tokens.getOrNull(index + 1)?.toIntOrNull() ?: 1
                    index += 2
                    if (multipv > 1) return
                }

                "score" -> {
                    val kind = tokens.getOrNull(index + 1)
                    val value = tokens.getOrNull(index + 2)?.toIntOrNull()
                    if (value != null) {
                        when (kind) {
                            "cp" -> {
                                scoreCp = value
                                mateIn = null
                            }

                            "mate" -> {
                                mateIn = value
                                scoreCp = null
                            }
                        }
                    }
                    index += 3
                }

                "pv" -> {
                    pv = tokens.drop(index + 1)
                    index = tokens.size
                }

                else -> index++
            }
        }

        _state.update {
            it.copy(
                thinking = true,
                depth = depth,
                scoreCp = scoreCp,
                mateIn = mateIn,
                pv = pv,
                info = "Thinking at depth $depth"
            )
        }
    }

    override fun newGame() {
        send("ucinewgame")
        send("isready")
        synchronized(lock) { pendingGo = null }
        currentFen = null
        _state.update {
            it.copy(
                thinking = false,
                depth = 0,
                scoreCp = null,
                mateIn = null,
                pv = emptyList(),
                bestMove = null,
                bestMoveFen = null,
                info = "New game"
            )
        }
    }

    override fun go(fen: String, depth: Int) {
        start()
        val request = fen to depth.coerceIn(1, 30)
        synchronized(lock) {
            if (!_state.value.ready) {
                pendingGo = request
                return
            }
        }
        goNow(request.first, request.second)
    }

    private fun goNow(fen: String, depth: Int) {
        currentFen = fen
        _state.update {
            it.copy(
                thinking = true,
                depth = 0,
                scoreCp = null,
                mateIn = null,
                pv = emptyList(),
                bestMove = null,
                bestMoveFen = null,
                info = "Thinking..."
            )
        }
        send("position fen $fen")
        send("go depth $depth")
    }

    override fun stop() {
        send("stop")
        _state.update { it.copy(info = "Stopped") }
    }

    override fun close() {
        closing.set(true)
        send("quit")
        val running = process
        process = null
        if (running != null) {
            try {
                running.waitFor(400, java.util.concurrent.TimeUnit.MILLISECONDS)
            } catch (_: InterruptedException) {
                Thread.currentThread().interrupt()
            }
            if (running.isAlive) running.destroy()
        }
        _state.update { it.copy(ready = false, thinking = false) }
    }

    private fun send(command: String) {
        synchronized(lock) {
            try {
                writer?.apply {
                    write(command)
                    newLine()
                    flush()
                }
            } catch (_: Throwable) {
                // The reader loop reports the process failure.
            }
        }
    }
}
