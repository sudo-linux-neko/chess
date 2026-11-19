package com.vaibhav.chessstudio

import com.vaibhav.chessstudio.engine.UciEngine
import com.vaibhav.chessstudio.game.ChessGame

expect fun createChessGame(): ChessGame
expect fun createEngine(): UciEngine
