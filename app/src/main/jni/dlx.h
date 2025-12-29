/*
 * dlx.h: Algorithm X with Dancing Links for Exact Cover Problems
 *
 * SPDX-License-Identifier: MIT
 * SPDX-FileCopyrightText: Copyright (C) 2024-2025 KeenKenning Contributors
 *
 * Clean-room implementation based on Donald Knuth's algorithm description
 * from "Dancing Links" (2000), available at https://arxiv.org/abs/cs/0011047
 *
 * The algorithm itself is not copyrightable; this is an independent
 * implementation for the Keen Kenning project.
 */

#ifndef DLX_H
#define DLX_H

#include <stdlib.h>

/*
 * DLX Node: Element in the sparse matrix.
 * Each node has links to its four neighbors in a toroidal structure.
 */
typedef struct dlx_node {
    struct dlx_node *L, *R, *U, *D;  /* Left, Right, Up, Down links */
    struct dlx_node *C;              /* Column header for this node */
    int row;                         /* Row index (-1 for headers) */
    int size;                        /* Column size (headers only) */
    int col_idx;                     /* Column index (headers only) */
} dlx_node;

/*
 * DLX Context: Holds the matrix and search state.
 */
typedef struct dlx_ctx {
    dlx_node *header;           /* Root header node */
    dlx_node *columns;          /* Array of column headers */
    dlx_node *nodes;            /* Pool of all nodes */
    int n_cols;                 /* Number of columns */
    int n_nodes;                /* Number of nodes allocated */
    int *solution;              /* Current solution stack */
    int sol_size;               /* Current solution size */
    int sol_cap;                /* Solution capacity */
    int found;                  /* Solution found flag */
} dlx_ctx;

/*
 * Create a DLX context from a constraint matrix.
 * matrix[row][col] = 1 if row covers column, 0 otherwise.
 * Returns NULL on failure.
 */
dlx_ctx *dlx_new(int n_rows, int n_cols, int **matrix);

/*
 * Free all resources associated with a DLX context.
 */
void dlx_destroy(dlx_ctx *ctx);

/*
 * Solve the exact cover problem.
 * Returns 1 if solution found, 0 otherwise.
 * Solution rows are stored in ctx->solution[0..ctx->sol_size-1].
 */
int dlx_solve(dlx_ctx *ctx);

/*
 * Get the solution (array of row indices).
 * Returns NULL if no solution was found.
 * The returned pointer is valid until dlx_destroy is called.
 */
int *dlx_get_solution(dlx_ctx *ctx, int *size);

#endif /* DLX_H */
