package com.vaibhav.chessstudio

import android.content.Context
import com.vaibhav.chessstudio.engine.JvmUciEngine
import com.vaibhav.chessstudio.engine.UciEngine
import com.vaibhav.chessstudio.game.ChessGame
import com.vaibhav.chessstudio.game.JvmChessGame
import java.io.File

object ChessStudioApp {
    lateinit var context: Context
}

actual fun createChessGame(): ChessGame = JvmChessGame()

actual fun createEngine(): UciEngine {
    val context = ChessStudioApp.context
    val workDir = File(context.filesDir, "engine").apply { mkdirs() }

    val engine = File(context.applicationInfo.nativeLibraryDir, "libstockfish.so")
    engine.setExecutable(true, false)

    val smallNet = copyAsset(context, "nnue/final.nnue", File(context.filesDir, "final.nnue"))

    return JvmUciEngine(
        engineFile = engine.takeIf { it.isFile },
        smallNet = smallNet,
        bigNet = null,
        workDir = workDir
    )
}

private fun copyAsset(context: Context, asset: String, target: File): File? {
    if (target.isFile && target.length() > 0L) return target

    return try {
        context.assets.open(asset).use { input ->
            target.outputStream().use { output -> input.copyTo(output) }
        }
        target
    } catch (_: Throwable) {
        null
    }
}
