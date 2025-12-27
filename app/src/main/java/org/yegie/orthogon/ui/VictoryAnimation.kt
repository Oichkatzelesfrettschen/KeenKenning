/*
 * VictoryAnimation.kt: Celebratory animation on puzzle completion
 *
 * SPDX-License-Identifier: MIT
 * SPDX-FileCopyrightText: Copyright (C) 2024-2025 Orthogon Contributors
 */

package org.yegie.orthogon.ui

import androidx.compose.animation.core.*
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.draw.rotate
import androidx.compose.ui.draw.scale
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.DrawScope
import androidx.compose.ui.graphics.drawscope.rotate
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.IntOffset
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import kotlinx.coroutines.delay
import org.yegie.orthogon.ui.theme.ZoneColors
import kotlin.math.abs
import kotlin.math.roundToInt
import kotlin.random.Random

/**
 * Windows Solitaire-style Victory Animation
 *
 * Physics based on the classic Windows 3.1/95 Solitaire:
 * - Gravity pulls tiles down
 * - Bounce with dampening coefficient (0.7)
 * - Tiles leave "trails" as they bounce
 * - Staggered launch from grid positions
 *
 * Modernized with:
 * - Rotation during flight
 * - Scale pulsing on bounce
 * - Particle sparkles
 * - Smooth 60fps Compose animations
 *
 * Sources:
 * - https://codepen.io/baileyparker/pen/KZbzVm (XP Solitaire recreation)
 * - https://github.com/peterkhayes/solitaireVictory (jQuery implementation)
 */

// Physics constants (modernized from original)
private const val GRAVITY = 1800f          // pixels/sec² (adjusted for mobile)
private const val DAMPENING = 0.65f        // Bounce coefficient of restitution
private const val MIN_BOUNCE_VEL = 100f    // Stop bouncing below this velocity
private const val TRAIL_INTERVAL_MS = 50L  // How often to spawn trail ghosts
private const val ROTATION_SPEED = 180f    // degrees/sec

data class BouncingTile(
    val id: Int,
    val zoneId: Int,
    val value: Int?,
    val clue: String?,
    var x: Float,
    var y: Float,
    var vx: Float,
    var vy: Float,
    var rotation: Float = 0f,
    var scale: Float = 1f,
    var alpha: Float = 1f,
    var isActive: Boolean = true
)

data class TileTrail(
    val x: Float,
    val y: Float,
    val rotation: Float,
    val zoneId: Int,
    val value: Int?,
    var alpha: Float = 0.6f,
    val createdAt: Long = System.currentTimeMillis()
)

@Composable
fun VictoryAnimation(
    cells: List<List<UiCell>>,
    gridSize: Int,
    cellSizePx: Float,
    gridOffsetX: Float,
    gridOffsetY: Float,
    screenWidth: Float,
    screenHeight: Float,
    onAnimationComplete: () -> Unit
) {
    // Initialize bouncing tiles from grid cells
    var tiles by remember {
        val tileList =
            cells.flatMapIndexed { x, col ->
                col.mapIndexed { y, cell ->
                    val startX = gridOffsetX + x * cellSizePx
                    val startY = gridOffsetY + y * cellSizePx

                    // Random initial velocities - spread outward from center
                    val centerX = gridOffsetX + (gridSize * cellSizePx) / 2
                    val dirX = if (startX < centerX) -1 else 1
                    val dirY = -1 // Always launch upward initially

                    BouncingTile(
                        id = x * gridSize + y,
                        zoneId = cell.zoneId,
                        value = cell.value,
                        clue = cell.clue,
                        x = startX,
                        y = startY,
                        vx = dirX * Random.nextFloat() * 400f + 100f,
                        vy = dirY * (Random.nextFloat() * 600f + 400f), // Launch upward
                        rotation = 0f
                    )
                }
            }
        mutableStateOf(tileList)
    }

    // Trail ghosts
    var trails by remember { mutableStateOf(listOf<TileTrail>()) }

    // Animation state
    var animationStarted by remember { mutableStateOf(false) }
    var showFinalMessage by remember { mutableStateOf(false) }
    var lastFrameTime by remember { mutableStateOf(0L) }
    var lastTrailTime by remember { mutableStateOf(0L) }

    // Staggered tile launch
    var launchedTiles by remember { mutableStateOf(0) }

    // Launch tiles with stagger
    LaunchedEffect(Unit) {
        animationStarted = true
        lastFrameTime = System.currentTimeMillis()
        lastTrailTime = lastFrameTime

        // Stagger tile launches
        for (i in tiles.indices) {
            delay(50L) // 50ms between each tile launch
            launchedTiles = i + 1
        }
    }

    // Physics update loop
    LaunchedEffect(animationStarted) {
        if (!animationStarted) return@LaunchedEffect

        while (tiles.any { it.isActive }) {
            val currentTime = System.currentTimeMillis()
            val deltaTime = (currentTime - lastFrameTime) / 1000f
            lastFrameTime = currentTime

            // Spawn trails periodically
            if (currentTime - lastTrailTime > TRAIL_INTERVAL_MS) {
                val newTrails = tiles
                    .filter { it.isActive && it.id < launchedTiles }
                    .map { tile ->
                        TileTrail(
                            x = tile.x,
                            y = tile.y,
                            rotation = tile.rotation,
                            zoneId = tile.zoneId,
                            value = tile.value
                        )
                    }
                trails = (trails + newTrails).filter {
                    // Keep trails for 500ms, fade them out
                    val age = currentTime - it.createdAt
                    if (age < 500) {
                        it.alpha = 0.6f * (1f - age / 500f)
                        true
                    } else false
                }
                lastTrailTime = currentTime
            }

            // Update physics for each active tile
            tiles = tiles.map { tile ->
                if (!tile.isActive || tile.id >= launchedTiles) return@map tile

                var newVy = tile.vy + GRAVITY * deltaTime
                var newY = tile.y + newVy * deltaTime
                var newX = tile.x + tile.vx * deltaTime
                var newRotation = tile.rotation + ROTATION_SPEED * deltaTime * (if (tile.vx > 0) 1 else -1)
                var newScale = tile.scale

                // Bounce off bottom
                if (newY > screenHeight - cellSizePx) {
                    newY = screenHeight - cellSizePx
                    newVy = -newVy * DAMPENING
                    newScale = 1.2f // Squash on impact

                    // Stop if velocity too low
                    if (abs(newVy) < MIN_BOUNCE_VEL) {
                        return@map tile.copy(isActive = false)
                    }
                }

                // Bounce off sides (wrap or bounce)
                if (newX < -cellSizePx || newX > screenWidth) {
                    return@map tile.copy(isActive = false) // Off screen = done
                }

                // Decay scale back to 1
                if (newScale > 1f) {
                    newScale = (newScale - deltaTime * 2f).coerceAtLeast(1f)
                }

                tile.copy(
                    x = newX,
                    y = newY,
                    vx = tile.vx,
                    vy = newVy,
                    rotation = newRotation,
                    scale = newScale
                )
            }

            delay(16L) // ~60fps
        }

        // All tiles done - show final message
        delay(200)
        showFinalMessage = true
        delay(2000)
        onAnimationComplete()
    }

    // Render
    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(Color(0xFF1a1a2e).copy(alpha = 0.95f))
    ) {
        // Draw trails first (behind tiles)
        Canvas(modifier = Modifier.fillMaxSize()) {
            trails.forEach { trail ->
                drawTileGhost(
                    x = trail.x,
                    y = trail.y,
                    size = cellSizePx,
                    rotation = trail.rotation,
                    zoneId = trail.zoneId,
                    alpha = trail.alpha
                )
            }
        }

        // Draw active bouncing tiles
        tiles.filter { it.isActive && it.id < launchedTiles }.forEach { tile ->
            BouncingTileView(
                tile = tile,
                cellSizePx = cellSizePx,
                gridSize = gridSize
            )
        }

        // Final victory message
        if (showFinalMessage) {
            VictoryMessage(
                modifier = Modifier.align(Alignment.Center)
            )
        }
    }
}

private fun DrawScope.drawTileGhost(
    x: Float,
    y: Float,
    size: Float,
    rotation: Float,
    zoneId: Int,
    alpha: Float
) {
    rotate(rotation, pivot = Offset(x + size / 2, y + size / 2)) {
        drawRoundRect(
            color = ZoneColors.forZone(zoneId).copy(alpha = alpha),
            topLeft = Offset(x, y),
            size = Size(size, size),
            cornerRadius = androidx.compose.ui.geometry.CornerRadius(4f, 4f)
        )
        // Border
        drawRoundRect(
            color = Color(0xFF212121).copy(alpha = alpha * 0.5f),
            topLeft = Offset(x, y),
            size = Size(size, size),
            cornerRadius = androidx.compose.ui.geometry.CornerRadius(4f, 4f),
            style = androidx.compose.ui.graphics.drawscope.Stroke(width = 2f)
        )
    }
}

@Composable
private fun BouncingTileView(
    tile: BouncingTile,
    cellSizePx: Float,
    gridSize: Int
) {
    val density = LocalDensity.current
    val cellSizeDp = with(density) { cellSizePx.toDp() }

    // Calculate text size based on grid size
    val valueTextSize = when {
        gridSize <= 4 -> 28.sp
        gridSize <= 6 -> 24.sp
        gridSize <= 8 -> 20.sp
        else -> 18.sp
    }

    Box(
        modifier = Modifier
            .offset { IntOffset(tile.x.roundToInt(), tile.y.roundToInt()) }
            .size(cellSizeDp)
            .scale(tile.scale)
            .rotate(tile.rotation)
            .background(
                ZoneColors.forZone(tile.zoneId),
                RoundedCornerShape(4.dp)
            )
            .padding(2.dp),
        contentAlignment = Alignment.Center
    ) {
        // Clue in corner
        if (tile.clue != null) {
            Text(
                text = tile.clue,
                fontSize = 10.sp,
                fontWeight = FontWeight.Bold,
                color = Color(0xFF1A1A1A),
                modifier = Modifier
                    .align(Alignment.TopStart)
                    .padding(2.dp)
            )
        }

        // Value
        if (tile.value != null && tile.value != -1) {
            Text(
                text = tile.value.toString(),
                fontSize = valueTextSize,
                fontWeight = FontWeight.Bold,
                color = Color(0xFF1A1A1A)
            )
        }
    }
}

@Composable
private fun VictoryMessage(modifier: Modifier = Modifier) {
    // Animated entrance
    val infiniteTransition = rememberInfiniteTransition(label = "victory")

    val scale by infiniteTransition.animateFloat(
        initialValue = 1f,
        targetValue = 1.05f,
        animationSpec = infiniteRepeatable(
            animation = tween(1000, easing = EaseInOutSine),
            repeatMode = RepeatMode.Reverse
        ),
        label = "victoryScale"
    )

    val glowAlpha by infiniteTransition.animateFloat(
        initialValue = 0.3f,
        targetValue = 0.8f,
        animationSpec = infiniteRepeatable(
            animation = tween(1500, easing = EaseInOutSine),
            repeatMode = RepeatMode.Reverse
        ),
        label = "victoryGlow"
    )

    // Entrance animation
    var visible by remember { mutableStateOf(false) }
    val entranceScale by animateFloatAsState(
        targetValue = if (visible) 1f else 0f,
        animationSpec = spring(
            dampingRatio = Spring.DampingRatioMediumBouncy,
            stiffness = Spring.StiffnessLow
        ),
        label = "entranceScale"
    )

    LaunchedEffect(Unit) {
        visible = true
    }

    Column(
        modifier = modifier
            .scale(entranceScale * scale),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        // Glow effect behind text
        Box {
            Text(
                text = "PUZZLE",
                fontSize = 56.sp,
                fontWeight = FontWeight.Black,
                color = Color(0xFF7B68EE).copy(alpha = glowAlpha),
                letterSpacing = 8.sp,
                modifier = Modifier
                    .scale(1.1f)
                    .alpha(glowAlpha)
            )
            Text(
                text = "PUZZLE",
                fontSize = 56.sp,
                fontWeight = FontWeight.Black,
                color = Color(0xFF7B68EE),
                letterSpacing = 8.sp
            )
        }

        Spacer(modifier = Modifier.height(8.dp))

        Box {
            Text(
                text = "SOLVED!",
                fontSize = 56.sp,
                fontWeight = FontWeight.Black,
                color = Color(0xFF00CED1).copy(alpha = glowAlpha),
                letterSpacing = 8.sp,
                modifier = Modifier
                    .scale(1.1f)
                    .alpha(glowAlpha)
            )
            Text(
                text = "SOLVED!",
                fontSize = 56.sp,
                fontWeight = FontWeight.Black,
                color = Color(0xFF00CED1),
                letterSpacing = 8.sp
            )
        }
    }
}
