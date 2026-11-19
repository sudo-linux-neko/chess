package com.vaibhav.chessstudio

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        ChessStudioApp.context = applicationContext
        enableEdgeToEdge()
        setContent { App() }
    }
}
