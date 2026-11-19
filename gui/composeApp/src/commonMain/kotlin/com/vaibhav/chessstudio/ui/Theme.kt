package com.vaibhav.chessstudio.ui

import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Typography
import androidx.compose.material3.darkColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.sp

val StudioNavy = Color(0xFF07111F)
val StudioDeep = Color(0xFF0B1627)
val StudioPanel = Color(0xF2111C2E)
val StudioPanelSoft = Color(0xB31A2940)
val StudioBorder = Color(0x24FFFFFF)
val StudioGold = Color(0xFFE9B44C)
val StudioTeal = Color(0xFF5FC9B6)
val StudioText = Color(0xFFF4F1E8)
val StudioMuted = Color(0xFF9BAAC2)
val BoardLight = Color(0xFFE9D3B0)
val BoardDark = Color(0xFF9C6B4B)

private val StudioColors = darkColorScheme(
    primary = StudioGold,
    onPrimary = Color(0xFF1D1608),
    secondary = StudioTeal,
    onSecondary = Color(0xFF06201C),
    background = StudioNavy,
    onBackground = StudioText,
    surface = StudioPanel,
    onSurface = StudioText,
    surfaceVariant = StudioPanelSoft,
    onSurfaceVariant = StudioMuted,
    outline = StudioBorder,
    error = Color(0xFFFF7777)
)

private val StudioTypography = Typography(
    displaySmall = TextStyle(
        fontFamily = FontFamily.SansSerif,
        fontWeight = FontWeight.Bold,
        fontSize = 34.sp,
        letterSpacing = (-0.6).sp
    ),
    headlineSmall = TextStyle(
        fontFamily = FontFamily.SansSerif,
        fontWeight = FontWeight.SemiBold,
        fontSize = 22.sp
    ),
    titleMedium = TextStyle(
        fontFamily = FontFamily.SansSerif,
        fontWeight = FontWeight.SemiBold,
        fontSize = 16.sp
    ),
    bodyMedium = TextStyle(
        fontFamily = FontFamily.SansSerif,
        fontSize = 14.sp
    ),
    labelMedium = TextStyle(
        fontFamily = FontFamily.SansSerif,
        fontWeight = FontWeight.Medium,
        fontSize = 12.sp,
        letterSpacing = 0.4.sp
    )
)

@Composable
fun ChessStudioTheme(content: @Composable () -> Unit) {
    MaterialTheme(
        colorScheme = StudioColors,
        typography = StudioTypography,
        content = content
    )
}
