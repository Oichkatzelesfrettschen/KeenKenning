/*
 * StoryBookUI.kt: Page-flip animation and book display for Story Mode
 *
 * Visual metaphor: An ancient book with pages that flip like a real tome.
 * Supports:
 * - Book cover display with gradient overlay
 * - Page-flip animation (3D perspective transform)
 * - Swipe gestures for navigation
 * - Accessibility: respects reduceMotion setting
 *
 * SPDX-License-Identifier: MIT
 * SPDX-FileCopyrightText: Copyright (C) 2024-2025 Orthogon Contributors
 */

package org.yegie.orthogon.ui

import androidx.compose.animation.core.Animatable
import androidx.compose.animation.core.FastOutSlowInEasing
import androidx.compose.animation.core.tween
import androidx.compose.foundation.background
import androidx.compose.foundation.gestures.detectHorizontalDragGestures
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.automirrored.filled.ArrowForward
import androidx.compose.material.icons.filled.PlayArrow
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.shadow
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import kotlinx.coroutines.launch
import org.yegie.orthogon.data.StoryBook
import org.yegie.orthogon.data.StoryChapter
import org.yegie.orthogon.ui.theme.LocalAccessibilitySettings
import kotlin.math.abs

/**
 * Page content types for the story book.
 */
sealed class PageContent {
    data class Cover(val book: StoryBook) : PageContent()
    data class ChapterIntro(val chapter: StoryChapter, val chapterNumber: Int) : PageContent()
    data class ChapterOutro(val chapter: StoryChapter, val chapterNumber: Int) : PageContent()
    data class BookEnd(val book: StoryBook) : PageContent()
}

/**
 * Main Story Book composable with page-flip animation.
 *
 * @param book The story book to display
 * @param currentChapterIndex Current chapter (0-based), -1 for cover
 * @param showOutro Whether to show the chapter outro (after puzzle completion)
 * @param dynamicNarratives Map of chapterId to Pair(intro, outro) for AI-generated narratives
 * @param isLoadingNarrative Whether narrative is currently being generated
 * @param onStartChapter Called when user starts a chapter puzzle
 * @param onNextChapter Called to advance to next chapter
 * @param onPreviousChapter Called to go back
 * @param onClose Called when user exits the book
 */
@Composable
fun StoryBookView(
    book: StoryBook,
    modifier: Modifier = Modifier,  // First optional parameter per Compose conventions
    currentChapterIndex: Int = -1,
    showOutro: Boolean = false,
    dynamicNarratives: Map<String, Pair<String?, String?>> = emptyMap(),
    isLoadingNarrative: Boolean = false,
    onStartChapter: (StoryChapter) -> Unit = {},
    onNextChapter: () -> Unit = {},
    onPreviousChapter: () -> Unit = {},
    onClose: () -> Unit = {}
) {
    val accessibilitySettings = LocalAccessibilitySettings.current
    val scope = rememberCoroutineScope()

    // Build page list: Cover -> (Intro, Outro)* -> End
    val pages = remember(book, currentChapterIndex, showOutro) {
        buildList {
            add(PageContent.Cover(book))
            book.chapters.forEachIndexed { index, chapter ->
                add(PageContent.ChapterIntro(chapter, index + 1))
                add(PageContent.ChapterOutro(chapter, index + 1))
            }
            add(PageContent.BookEnd(book))
        }
    }

    // Calculate current page index
    val currentPageIndex = remember(currentChapterIndex, showOutro) {
        when {
            currentChapterIndex < 0 -> 0  // Cover
            else -> {
                val baseIndex = 1 + (currentChapterIndex * 2)
                if (showOutro) baseIndex + 1 else baseIndex
            }
        }
    }

    var pageIndex by remember { mutableIntStateOf(currentPageIndex) }
    val flipProgress = remember { Animatable(0f) }
    var isFlipping by remember { mutableStateOf(false) }
    var flipDirection by remember { mutableIntStateOf(0) }  // -1 = back, 1 = forward

    // Swipe detection threshold
    val swipeThreshold = 100f

    // Animate page flip
    suspend fun flipToPage(targetIndex: Int) {
        if (targetIndex == pageIndex || isFlipping) return
        if (targetIndex < 0 || targetIndex >= pages.size) return

        isFlipping = true
        flipDirection = if (targetIndex > pageIndex) 1 else -1

        // Animate the flip
        val duration = if (accessibilitySettings.reduceMotion) 100 else 400
        flipProgress.animateTo(
            targetValue = 1f,
            animationSpec = tween(durationMillis = duration, easing = FastOutSlowInEasing)
        )

        pageIndex = targetIndex
        flipProgress.snapTo(0f)
        isFlipping = false
        flipDirection = 0
    }

    // Sync with external state changes
    LaunchedEffect(currentPageIndex) {
        if (currentPageIndex != pageIndex) {
            scope.launch { flipToPage(currentPageIndex) }
        }
    }

    Box(
        modifier = modifier
            .fillMaxSize()
            .background(
                Brush.verticalGradient(
                    colors = listOf(
                        Color(book.theme.primaryColor.toInt()).copy(alpha = 0.3f),
                        Color(0xFF1A1A2E)
                    )
                )
            )
            .pointerInput(Unit) {
                detectHorizontalDragGestures(
                    onDragEnd = {
                        // Gesture completed - handled in onHorizontalDrag
                    },
                    onHorizontalDrag = { change, dragAmount ->
                        change.consume()
                        if (!isFlipping && abs(dragAmount) > swipeThreshold / 10) {
                            scope.launch {
                                if (dragAmount < -swipeThreshold / 5 && pageIndex < pages.size - 1) {
                                    flipToPage(pageIndex + 1)
                                    onNextChapter()
                                } else if (dragAmount > swipeThreshold / 5 && pageIndex > 0) {
                                    flipToPage(pageIndex - 1)
                                    onPreviousChapter()
                                }
                            }
                        }
                    }
                )
            }
    ) {
        // Book frame with shadow
        Card(
            modifier = Modifier
                .fillMaxSize()
                .padding(16.dp)
                .shadow(8.dp, RoundedCornerShape(8.dp)),
            shape = RoundedCornerShape(8.dp),
            colors = CardDefaults.cardColors(containerColor = Color(0xFF2D2D3A))
        ) {
            Box(modifier = Modifier.fillMaxSize()) {
                // Current page (behind during flip)
                PageView(
                    content = pages[pageIndex],
                    onStartChapter = onStartChapter,
                    dynamicNarratives = dynamicNarratives,
                    isLoadingNarrative = isLoadingNarrative,
                    modifier = Modifier.fillMaxSize()
                )

                // Flipping page overlay (3D transform)
                if (isFlipping && flipDirection != 0) {
                    val nextPageIndex = pageIndex + flipDirection
                    if (nextPageIndex in pages.indices) {
                        val rotation = if (flipDirection > 0) {
                            // Forward flip: page rotates from 0 to -180 (like turning right page)
                            -180f * flipProgress.value
                        } else {
                            // Backward flip: page rotates from -180 to 0
                            -180f * (1f - flipProgress.value)
                        }

                        Box(
                            modifier = Modifier
                                .fillMaxSize()
                                .graphicsLayer {
                                    rotationY = rotation
                                    cameraDistance = 12f * density
                                    // Hide backface
                                    alpha = if (abs(rotation) > 90f) 0f else 1f
                                }
                        ) {
                            PageView(
                                content = if (flipDirection > 0) pages[pageIndex] else pages[nextPageIndex],
                                onStartChapter = onStartChapter,
                                dynamicNarratives = dynamicNarratives,
                                isLoadingNarrative = isLoadingNarrative,
                                modifier = Modifier.fillMaxSize()
                            )
                        }
                    }
                }

                // Navigation buttons
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .align(Alignment.BottomCenter)
                        .padding(16.dp),
                    horizontalArrangement = Arrangement.SpaceBetween
                ) {
                    // Back button
                    IconButton(
                        onClick = {
                            scope.launch {
                                if (pageIndex > 0) {
                                    flipToPage(pageIndex - 1)
                                    onPreviousChapter()
                                } else {
                                    onClose()
                                }
                            }
                        },
                        enabled = !isFlipping
                    ) {
                        Icon(
                            imageVector = Icons.AutoMirrored.Filled.ArrowBack,
                            contentDescription = "Previous page",
                            tint = Color.White.copy(alpha = 0.8f)
                        )
                    }

                    // Page indicator
                    Text(
                        text = "${pageIndex + 1} / ${pages.size}",
                        color = Color.White.copy(alpha = 0.6f),
                        fontSize = 14.sp
                    )

                    // Forward button
                    IconButton(
                        onClick = {
                            scope.launch {
                                if (pageIndex < pages.size - 1) {
                                    flipToPage(pageIndex + 1)
                                    onNextChapter()
                                }
                            }
                        },
                        enabled = !isFlipping && pageIndex < pages.size - 1
                    ) {
                        Icon(
                            imageVector = Icons.AutoMirrored.Filled.ArrowForward,
                            contentDescription = "Next page",
                            tint = Color.White.copy(alpha = if (pageIndex < pages.size - 1) 0.8f else 0.3f)
                        )
                    }
                }
            }
        }
    }
}

/**
 * Individual page renderer based on content type.
 */
@Composable
private fun PageView(
    content: PageContent,
    onStartChapter: (StoryChapter) -> Unit,
    dynamicNarratives: Map<String, Pair<String?, String?>>,
    isLoadingNarrative: Boolean,
    modifier: Modifier = Modifier
) {
    when (content) {
        is PageContent.Cover -> CoverPage(content.book, modifier)
        is PageContent.ChapterIntro -> {
            val narratives = dynamicNarratives[content.chapter.id]
            ChapterIntroPage(
                chapter = content.chapter,
                chapterNumber = content.chapterNumber,
                dynamicNarrative = narratives?.first,
                isLoadingNarrative = isLoadingNarrative,
                onStart = { onStartChapter(content.chapter) },
                modifier = modifier
            )
        }
        is PageContent.ChapterOutro -> {
            val narratives = dynamicNarratives[content.chapter.id]
            ChapterOutroPage(
                chapter = content.chapter,
                chapterNumber = content.chapterNumber,
                dynamicNarrative = narratives?.second,
                isLoadingNarrative = isLoadingNarrative,
                modifier = modifier
            )
        }
        is PageContent.BookEnd -> BookEndPage(content.book, modifier)
    }
}

/**
 * Book cover page with title and decorative elements.
 */
@Composable
private fun CoverPage(book: StoryBook, modifier: Modifier = Modifier) {
    val primaryColor = Color(book.theme.primaryColor.toInt())
    val accentColor = Color(book.theme.accentColor.toInt())

    Box(
        modifier = modifier
            .background(
                Brush.verticalGradient(
                    colors = listOf(primaryColor.copy(alpha = 0.9f), primaryColor.copy(alpha = 0.6f))
                )
            )
            .padding(32.dp),
        contentAlignment = Alignment.Center
    ) {
        Column(
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.Center
        ) {
            // Decorative border top
            Box(
                modifier = Modifier
                    .width(200.dp)
                    .height(4.dp)
                    .clip(RoundedCornerShape(2.dp))
                    .background(accentColor)
            )

            Spacer(modifier = Modifier.height(48.dp))

            // Book title
            Text(
                text = book.title,
                fontSize = 42.sp,
                fontWeight = FontWeight.Bold,
                fontFamily = FontFamily.Serif,
                color = Color.White,
                textAlign = TextAlign.Center
            )

            Spacer(modifier = Modifier.height(16.dp))

            // Book subtitle
            Text(
                text = book.subtitle,
                fontSize = 28.sp,
                fontStyle = FontStyle.Italic,
                fontFamily = FontFamily.Serif,
                color = accentColor,
                textAlign = TextAlign.Center
            )

            Spacer(modifier = Modifier.height(48.dp))

            // Decorative element
            Box(
                modifier = Modifier
                    .size(80.dp)
                    .clip(RoundedCornerShape(40.dp))
                    .background(accentColor.copy(alpha = 0.3f)),
                contentAlignment = Alignment.Center
            ) {
                Text(
                    text = when (book.id) {
                        "book1" -> "I"
                        "book2" -> "II"
                        "book3" -> "III"
                        "book4" -> "IV"
                        else -> "?"
                    },
                    fontSize = 32.sp,
                    fontWeight = FontWeight.Bold,
                    fontFamily = FontFamily.Serif,
                    color = Color.White
                )
            }

            Spacer(modifier = Modifier.height(48.dp))

            // Decorative border bottom
            Box(
                modifier = Modifier
                    .width(200.dp)
                    .height(4.dp)
                    .clip(RoundedCornerShape(2.dp))
                    .background(accentColor)
            )

            Spacer(modifier = Modifier.height(32.dp))

            // Chapter count
            Text(
                text = "${book.chapters.size} Chapters",
                fontSize = 16.sp,
                color = Color.White.copy(alpha = 0.7f)
            )
        }
    }
}

/**
 * Chapter introduction page with narrative and start button.
 */
@Composable
private fun ChapterIntroPage(
    chapter: StoryChapter,
    chapterNumber: Int,
    dynamicNarrative: String?,
    isLoadingNarrative: Boolean,
    onStart: () -> Unit,
    modifier: Modifier = Modifier
) {
    val backgroundColor = Color(chapter.theme.backgroundColor.toInt())
    val primaryColor = Color(chapter.theme.primaryColor.toInt())
    val textColor = Color(chapter.theme.textColor.toInt())

    // Use dynamic narrative if available, otherwise fall back to static
    val narrativeText = dynamicNarrative ?: chapter.introNarrative

    Box(
        modifier = modifier
            .background(backgroundColor)
            .padding(24.dp)
    ) {
        Column(
            modifier = Modifier.fillMaxSize(),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            // Chapter header
            Text(
                text = "Chapter $chapterNumber",
                fontSize = 16.sp,
                color = primaryColor,
                fontWeight = FontWeight.Medium
            )

            Spacer(modifier = Modifier.height(8.dp))

            Text(
                text = chapter.title,
                fontSize = 28.sp,
                fontWeight = FontWeight.Bold,
                fontFamily = FontFamily.Serif,
                color = textColor,
                textAlign = TextAlign.Center
            )

            Spacer(modifier = Modifier.height(4.dp))

            Text(
                text = chapter.description,
                fontSize = 14.sp,
                fontStyle = FontStyle.Italic,
                color = textColor.copy(alpha = 0.7f)
            )

            Spacer(modifier = Modifier.height(32.dp))

            // Narrative text with loading state
            Card(
                modifier = Modifier
                    .fillMaxWidth()
                    .weight(1f),
                colors = CardDefaults.cardColors(
                    containerColor = Color(chapter.theme.cellColor.toInt())
                ),
                shape = RoundedCornerShape(12.dp)
            ) {
                Column(
                    modifier = Modifier
                        .fillMaxSize()
                        .padding(20.dp),
                    verticalArrangement = Arrangement.Center,
                    horizontalAlignment = Alignment.CenterHorizontally
                ) {
                    if (isLoadingNarrative && dynamicNarrative == null) {
                        // Loading indicator
                        Text(
                            text = "Weaving the tale...",
                            fontSize = 14.sp,
                            fontStyle = FontStyle.Italic,
                            color = textColor.copy(alpha = 0.5f)
                        )
                    } else {
                        Text(
                            text = narrativeText,
                            fontSize = 16.sp,
                            fontFamily = FontFamily.Serif,
                            color = textColor,
                            lineHeight = 26.sp,
                            textAlign = TextAlign.Center
                        )
                    }
                }
            }

            Spacer(modifier = Modifier.height(24.dp))

            // Puzzle info
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceEvenly
            ) {
                InfoChip(label = "Grid", value = "${chapter.gridSize}x${chapter.gridSize}", color = primaryColor)
                InfoChip(label = "Puzzles", value = "${chapter.puzzleCount}", color = primaryColor)
                InfoChip(label = "Mode", value = chapter.gameMode.displayName, color = primaryColor)
            }

            Spacer(modifier = Modifier.height(24.dp))

            // Start button
            Button(
                onClick = onStart,
                colors = ButtonDefaults.buttonColors(containerColor = primaryColor),
                modifier = Modifier
                    .fillMaxWidth()
                    .height(56.dp)
            ) {
                Icon(
                    imageVector = Icons.Default.PlayArrow,
                    contentDescription = null,
                    modifier = Modifier.size(24.dp)
                )
                Spacer(modifier = Modifier.width(8.dp))
                Text(
                    text = "Begin Chapter",
                    fontSize = 18.sp,
                    fontWeight = FontWeight.Bold
                )
            }
        }
    }
}

/**
 * Chapter outro page shown after completing puzzles.
 */
@Composable
private fun ChapterOutroPage(
    chapter: StoryChapter,
    chapterNumber: Int,
    dynamicNarrative: String?,
    isLoadingNarrative: Boolean,
    modifier: Modifier = Modifier
) {
    val backgroundColor = Color(chapter.theme.backgroundColor.toInt())
    val primaryColor = Color(chapter.theme.primaryColor.toInt())
    val textColor = Color(chapter.theme.textColor.toInt())

    // Use dynamic narrative if available, otherwise fall back to static
    val narrativeText = dynamicNarrative ?: chapter.outroNarrative

    Box(
        modifier = modifier
            .background(backgroundColor)
            .padding(24.dp),
        contentAlignment = Alignment.Center
    ) {
        Column(
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.Center
        ) {
            // Completion badge
            Box(
                modifier = Modifier
                    .size(100.dp)
                    .clip(RoundedCornerShape(50.dp))
                    .background(primaryColor.copy(alpha = 0.2f)),
                contentAlignment = Alignment.Center
            ) {
                Text(
                    text = "Complete",
                    fontSize = 14.sp,
                    fontWeight = FontWeight.Bold,
                    color = primaryColor
                )
            }

            Spacer(modifier = Modifier.height(32.dp))

            Text(
                text = "Chapter $chapterNumber Complete",
                fontSize = 24.sp,
                fontWeight = FontWeight.Bold,
                color = textColor
            )

            Spacer(modifier = Modifier.height(32.dp))

            // Outro narrative with loading state
            Card(
                modifier = Modifier.fillMaxWidth(),
                colors = CardDefaults.cardColors(
                    containerColor = Color(chapter.theme.cellColor.toInt())
                ),
                shape = RoundedCornerShape(12.dp)
            ) {
                Column(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(24.dp),
                    horizontalAlignment = Alignment.CenterHorizontally
                ) {
                    if (isLoadingNarrative && dynamicNarrative == null) {
                        Text(
                            text = "Chronicling your victory...",
                            fontSize = 14.sp,
                            fontStyle = FontStyle.Italic,
                            color = textColor.copy(alpha = 0.5f)
                        )
                    } else {
                        Text(
                            text = narrativeText,
                            fontSize = 16.sp,
                            fontFamily = FontFamily.Serif,
                            color = textColor,
                            lineHeight = 26.sp,
                            textAlign = TextAlign.Center
                        )
                    }
                }
            }

            Spacer(modifier = Modifier.height(24.dp))

            Text(
                text = "Swipe to continue...",
                fontSize = 14.sp,
                fontStyle = FontStyle.Italic,
                color = textColor.copy(alpha = 0.5f)
            )
        }
    }
}

/**
 * Book end page (final page after all chapters).
 */
@Composable
private fun BookEndPage(book: StoryBook, modifier: Modifier = Modifier) {
    val primaryColor = Color(book.theme.primaryColor.toInt())
    val accentColor = Color(book.theme.accentColor.toInt())

    Box(
        modifier = modifier
            .background(
                Brush.verticalGradient(
                    colors = listOf(primaryColor.copy(alpha = 0.8f), Color(0xFF1A1A2E))
                )
            )
            .padding(32.dp),
        contentAlignment = Alignment.Center
    ) {
        Column(
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.Center
        ) {
            Text(
                text = "The End",
                fontSize = 42.sp,
                fontWeight = FontWeight.Bold,
                fontFamily = FontFamily.Serif,
                fontStyle = FontStyle.Italic,
                color = accentColor
            )

            Spacer(modifier = Modifier.height(24.dp))

            Text(
                text = "${book.title}: ${book.subtitle}",
                fontSize = 18.sp,
                color = Color.White.copy(alpha = 0.8f)
            )

            Spacer(modifier = Modifier.height(48.dp))

            Text(
                text = "Congratulations!",
                fontSize = 24.sp,
                fontWeight = FontWeight.Medium,
                color = Color.White
            )

            Spacer(modifier = Modifier.height(16.dp))

            Text(
                text = "You have completed all ${book.chapters.size} chapters.",
                fontSize = 16.sp,
                color = Color.White.copy(alpha = 0.7f),
                textAlign = TextAlign.Center
            )
        }
    }
}

/**
 * Small info chip for displaying puzzle metadata.
 */
@Composable
private fun InfoChip(
    label: String,
    value: String,
    color: Color
) {
    Column(
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Text(
            text = label,
            fontSize = 12.sp,
            color = color.copy(alpha = 0.7f)
        )
        Text(
            text = value,
            fontSize = 16.sp,
            fontWeight = FontWeight.Bold,
            color = color
        )
    }
}
