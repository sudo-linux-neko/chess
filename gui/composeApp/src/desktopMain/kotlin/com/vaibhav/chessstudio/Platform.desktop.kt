package com.vaibhav.chessstudio

import com.vaibhav.chessstudio.engine.JvmUciEngine
import com.vaibhav.chessstudio.engine.UciEngine
import com.vaibhav.chessstudio.game.ChessGame
import com.vaibhav.chessstudio.game.JvmChessGame
import java.io.File

actual fun createChessGame(): ChessGame = JvmChessGame()

actual fun createEngine(): UciEngine {
    val home = File(System.getProperty("user.home"), ".chessstudio").apply { mkdirs() }
    val isWindows = System.getProperty("os.name").startsWith("Windows", ignoreCase = true)
    val engineName = if (isWindows) "stockfish.exe" else "stockfish"

    val bundledEngine = copyResource("/engine/stockfish", File(home, engineName))
    val systemEngine = listOf(
        "/usr/games/stockfish",
        "/usr/bin/stockfish",
        "/usr/local/bin/stockfish",
        "/opt/homebrew/bin/stockfish"
    ).map(::File).firstOrNull { it.isFile }

    val smallNet = copyResource("/nnue/final.nnue", File(home, "final.nnue"))

    return JvmUciEngine(
        engineFile = bundledEngine ?: systemEngine,
        smallNet = smallNet,
        bigNet = null,
        workDir = home
    )
}

private fun copyResource(resource: String, target: File): File? {
    if (target.isFile && target.length() > 0L) return target

    val stream = JvmUciEngine::class.java.getResourceAsStream(resource) ?: return null
    target.parentFile?.mkdirs()
    return try {
        stream.use { input ->
            target.outputStream().use { output -> input.copyTo(output) }
        }
        target.setExecutable(true, false)
        target
    } catch (_: Throwable) {
        null
    }
}
