package com.vaibhav.chessstudio

import androidx.compose.ui.unit.dp
import androidx.compose.ui.window.Window
import androidx.compose.ui.window.application
import androidx.compose.ui.window.rememberWindowState

fun main() = application {
    Window(
        onCloseRequest = ::exitApplication,
        title = "Chess Studio",
        state = rememberWindowState(width = 1180.dp, height = 840.dp)
    ) {
        App()
    }
}
