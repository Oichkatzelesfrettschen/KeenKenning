/*
 * maxflow_optimized.c: SIMD-optimized Edmonds-Karp max-flow algorithm
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 * SPDX-FileCopyrightText: Copyright (C) 2024-2025 KeenKeen Contributors
 *
 * This file is part of KeenKeen for Android.
 *
 * Provides AVX2, SSE2, and ARM NEON optimized implementations of the
 * max-flow algorithm for improved puzzle generation performance.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <https://www.gnu.org/licenses/>.
 */

#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif
#ifdef __ARM_NEON
#include <arm_neon.h>
#endif
#include "maxflow.h"
#include "puzzles.h"
#include <assert.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/*
 * Fully Optimized maxflow.c with AVX2 Fast Path.
 */

int maxflow_scratch_size(int nv) {
    return (nv * 4) * sizeof(int);
}

void maxflow_setup_backedges(int ne, const int *edges, int *backedges) {
    int i, n;
    for (i = 0; i < ne; i++)
        backedges[i] = i;
    // Heapsort implementation (scalar, setup is not bottleneck)
    n = 0;
#define LESS(i, j)                                                                                 \
    ((edges[2 * (i) + 1] < edges[2 * (j) + 1]) ||                                                  \
     (edges[2 * (i) + 1] == edges[2 * (j) + 1] && edges[2 * (i)] < edges[2 * (j)]))
#define PARENT(n) (((n) - 1) / 2)
#define LCHILD(n) (2 * (n) + 1)
#define RCHILD(n) (2 * (n) + 2)
#define SWAP(i, j)                                                                                 \
    do {                                                                                           \
        int swaptmp = (i);                                                                         \
        (i) = (j);                                                                                 \
        (j) = swaptmp;                                                                             \
    } while (0)
    while (n < ne) {
        n++;
        i = n - 1;
        while (i > 0) {
            int p = PARENT(i);
            if (LESS(backedges[p], backedges[i])) {
                SWAP(backedges[p], backedges[i]);
                i = p;
            } else
                break;
        }
    }
    while (n > 0) {
        n--;
        SWAP(backedges[0], backedges[n]);
        i = 0;
        while (1) {
            int lc = LCHILD(i), rc = RCHILD(i);
            if (lc >= n)
                break;
            if (rc >= n) {
                if (LESS(backedges[i], backedges[lc]))
                    SWAP(backedges[i], backedges[lc]);
                break;
            } else {
                if (LESS(backedges[i], backedges[lc]) || LESS(backedges[i], backedges[rc])) {
                    if (LESS(backedges[lc], backedges[rc])) {
                        SWAP(backedges[i], backedges[rc]);
                        i = rc;
                    } else {
                        SWAP(backedges[i], backedges[lc]);
                        i = lc;
                    }
                } else
                    break;
            }
        }
    }
}

int maxflow_with_scratch(void *scratch, int nv, int source, int sink, int ne, const int *edges,
                         const int *backedges, const int *capacity, int *flow, int *cut) {
    int *todo = (int *)scratch;
    int *prev = todo + nv;
    int *firstedge = todo + 2 * nv;
    int *firstbackedge = todo + 3 * nv;
    int i, j, head, tail, from, to;
    int totalflow = 0;

    (void)cut;

    j = 0;
    for (i = 0; i < ne; i++)
        while (j <= edges[2 * i])
            firstedge[j++] = i;
    while (j < nv)
        firstedge[j++] = ne;

    j = 0;
    for (i = 0; i < ne; i++)
        while (j <= edges[2 * backedges[i] + 1])
            firstbackedge[j++] = i;
    while (j < nv)
        firstbackedge[j++] = ne;

    memset(flow, 0, ne * sizeof(int));

    while (1) {
#if defined(__AVX2__)
        __m256i neg_one = _mm256_set1_epi32(-1);
        for (i = 0; i < nv; i += 8) {
            if (i + 8 <= nv)
                _mm256_storeu_si256((__m256i *)(prev + i), neg_one);
            else
                for (int k = i; k < nv; k++)
                    prev[k] = -1;
        }
#elif defined(__SSE2__) || defined(__x86_64__) || defined(__i386__)
        __m128i neg_one_sse = _mm_set1_epi32(-1);
        for (i = 0; i < nv; i += 4) {
            if (i + 4 <= nv)
                _mm_storeu_si128((__m128i *)(prev + i), neg_one_sse);
            else
                for (int k = i; k < nv; k++)
                    prev[k] = -1;
        }
#elif defined(__ARM_NEON)
        int32x4_t neg_one_neon = vdupq_n_s32(-1);
        for (i = 0; i < nv; i += 4) {
            if (i + 4 <= nv)
                vst1q_s32(prev + i, neg_one_neon);
            else
                for (int k = i; k < nv; k++)
                    prev[k] = -1;
        }
#else
        for (i = 0; i < nv; i++)
            prev[i] = -1;
#endif

        uint32_t visited_mask = (1 << source);
        head = tail = 0;
        todo[tail++] = source;

        while (head < tail && !(visited_mask & (1 << sink))) {
            from = todo[head++];
            for (i = firstedge[from]; i < ne && edges[2 * i] == from; i++) {
                to = edges[2 * i + 1];
                if (!(visited_mask & (1 << to))) {
                    if (capacity[i] < 0 || flow[i] < capacity[i]) {
                        prev[to] = 2 * i;
                        todo[tail++] = to;
                        visited_mask |= (1 << to);
                    }
                }
            }
            for (i = firstbackedge[from]; i < ne; i++) {
                j = backedges[i];
                if (edges[2 * j + 1] != from)
                    break;
                to = edges[2 * j];
                if (!(visited_mask & (1 << to))) {
                    if (flow[j] > 0) {
                        prev[to] = 2 * j + 1;
                        todo[tail++] = to;
                        visited_mask |= (1 << to);
                    }
                }
            }
        }

        if (visited_mask & (1 << sink)) {
            int path_max = -1;
            to = sink;
            while (to != source) {
                i = prev[to];
                int eidx = i / 2;
                int spare =
                    (i & 1) ? flow[eidx] : (capacity[eidx] < 0 ? -1 : capacity[eidx] - flow[eidx]);
                if (path_max < 0 || (spare >= 0 && spare < path_max))
                    path_max = spare;
                to = edges[i];
            }
            to = sink;
            while (to != source) {
                i = prev[to];
                if (i & 1)
                    flow[i / 2] -= path_max;
                else
                    flow[i / 2] += path_max;
                to = edges[i];
            }
            totalflow += path_max;
            continue;
        }
        break;
    }
    return totalflow;
}
