/*
 * dlx.c: Algorithm X with Dancing Links for Exact Cover Problems
 *
 * SPDX-License-Identifier: MIT
 * SPDX-FileCopyrightText: Copyright (C) 2024-2025 KeenKenning Contributors
 *
 * Clean-room implementation based on Donald Knuth's algorithm description.
 * The core insight: removing and restoring nodes in doubly-linked lists
 * can be done in O(1) by exploiting the fact that removed nodes still
 * remember their neighbors.
 */

#include "dlx.h"

#include <string.h>

/*
 * Cover column c: Remove c from header list and remove all rows
 * that have a 1 in column c from their respective columns.
 */
static void cover(dlx_node* c) {
    dlx_node *i, *j;

    /* Remove c from the horizontal header list */
    c->R->L = c->L;
    c->L->R = c->R;

    /* For each row i in column c */
    for (i = c->D; i != c; i = i->D) {
        /* For each node j in row i (except the node in column c) */
        for (j = i->R; j != i; j = j->R) {
            /* Remove j from its column */
            j->D->U = j->U;
            j->U->D = j->D;
            j->C->size--;
        }
    }
}

/*
 * Uncover column c: Restore c and all its rows.
 * Must be called in exact reverse order of cover().
 */
static void uncover(dlx_node* c) {
    dlx_node *i, *j;

    /* For each row i in column c (in reverse order) */
    for (i = c->U; i != c; i = i->U) {
        /* For each node j in row i (in reverse order) */
        for (j = i->L; j != i; j = j->L) {
            /* Restore j to its column */
            j->C->size++;
            j->D->U = j;
            j->U->D = j;
        }
    }

    /* Restore c to the horizontal header list */
    c->R->L = c;
    c->L->R = c;
}

/*
 * Choose the column with minimum size (S heuristic).
 * This reduces branching factor and speeds up the search.
 */
static dlx_node* choose_column(dlx_ctx* ctx) {
    dlx_node *c, *best;
    int min_size;

    best = ctx->header->R;
    if (best == ctx->header) return NULL; /* Empty matrix */

    min_size = best->size;
    for (c = best->R; c != ctx->header; c = c->R) {
        if (c->size < min_size) {
            min_size = c->size;
            best = c;
        }
    }

    return best;
}

/*
 * Recursive search function.
 * Returns 1 when a solution is found, 0 otherwise.
 */
static int search(dlx_ctx* ctx, int k) {
    dlx_node *c, *r, *j;

    /* If the matrix is empty, we found a solution */
    if (ctx->header->R == ctx->header) {
        ctx->sol_size = k;
        ctx->found = 1;
        return 1;
    }

    /* Choose a column with minimum size */
    c = choose_column(ctx);
    if (!c || c->size == 0) {
        return 0; /* Dead end: column with no rows */
    }

    /* Cover the chosen column */
    cover(c);

    /* Try each row in column c */
    for (r = c->D; r != c; r = r->D) {
        /* Add this row to the solution */
        if (k >= ctx->sol_cap) {
            int new_cap = ctx->sol_cap * 2;
            int* new_sol = (int*)realloc(ctx->solution, (size_t)new_cap * sizeof(int));
            if (!new_sol) return 0;
            ctx->solution = new_sol;
            ctx->sol_cap = new_cap;
        }
        ctx->solution[k] = r->row;

        /* Cover all other columns in this row */
        for (j = r->R; j != r; j = j->R) {
            cover(j->C);
        }

        /* Recurse */
        if (search(ctx, k + 1)) {
            return 1;
        }

        /* Backtrack: uncover columns in reverse order */
        for (j = r->L; j != r; j = j->L) {
            uncover(j->C);
        }
    }

    /* Uncover the chosen column */
    uncover(c);

    return 0;
}

/*
 * Create a new DLX context from a constraint matrix.
 */
dlx_ctx* dlx_new(int n_rows, int n_cols, int** matrix) {
    dlx_ctx* ctx;
    int i, j, node_count, node_idx;
    int* row_first; /* First node in each row */
    int* row_last;  /* Last node in each row */

    if (n_rows <= 0 || n_cols <= 0 || !matrix) return NULL;

    ctx = (dlx_ctx*)calloc(1, sizeof(dlx_ctx));
    if (!ctx) return NULL;

    ctx->n_cols = n_cols;

    /* Count total nodes needed */
    node_count = 0;
    for (i = 0; i < n_rows; i++) {
        for (j = 0; j < n_cols; j++) {
            if (matrix[i][j]) node_count++;
        }
    }

    /* Allocate header + column headers */
    ctx->header = (dlx_node*)calloc(1, sizeof(dlx_node));
    ctx->columns = (dlx_node*)calloc((size_t)n_cols, sizeof(dlx_node));
    ctx->nodes = (dlx_node*)calloc((size_t)node_count, sizeof(dlx_node));

    if (!ctx->header || !ctx->columns || !ctx->nodes) {
        dlx_destroy(ctx);
        return NULL;
    }

    ctx->n_nodes = node_count;

    /* Initialize header node */
    ctx->header->L = ctx->header->R = ctx->header;
    ctx->header->U = ctx->header->D = ctx->header;
    ctx->header->C = ctx->header;
    ctx->header->row = -1;

    /* Initialize column headers and link them horizontally */
    for (j = 0; j < n_cols; j++) {
        dlx_node* col = &ctx->columns[j];
        col->U = col->D = col;
        col->C = col;
        col->row = -1;
        col->size = 0;
        col->col_idx = j;

        /* Link into header list */
        col->L = ctx->header->L;
        col->R = ctx->header;
        ctx->header->L->R = col;
        ctx->header->L = col;
    }

    /* Track first/last node in each row for horizontal linking */
    row_first = (int*)calloc((size_t)n_rows, sizeof(int));
    row_last = (int*)calloc((size_t)n_rows, sizeof(int));
    if (!row_first || !row_last) {
        free(row_first);
        free(row_last);
        dlx_destroy(ctx);
        return NULL;
    }
    for (i = 0; i < n_rows; i++) {
        row_first[i] = -1;
        row_last[i] = -1;
    }

    /* Create nodes and link vertically */
    node_idx = 0;
    for (j = 0; j < n_cols; j++) {
        dlx_node* col = &ctx->columns[j];
        for (i = 0; i < n_rows; i++) {
            if (matrix[i][j]) {
                dlx_node* node = &ctx->nodes[node_idx];
                node->row = i;
                node->C = col;
                node->col_idx = j;

                /* Link vertically into column */
                node->U = col->U;
                node->D = col;
                col->U->D = node;
                col->U = node;
                col->size++;

                /* Track for horizontal linking */
                if (row_first[i] < 0) {
                    row_first[i] = node_idx;
                }
                row_last[i] = node_idx;

                node_idx++;
            }
        }
    }

    /* Link nodes horizontally within rows */
    for (i = 0; i < n_rows; i++) {
        if (row_first[i] >= 0) {
            dlx_node *first = NULL, *prev = NULL;
            for (j = 0; j < node_count; j++) {
                dlx_node* node = &ctx->nodes[j];
                if (node->row == i) {
                    if (!first) {
                        first = node;
                    }
                    if (prev) {
                        prev->R = node;
                        node->L = prev;
                    }
                    prev = node;
                }
            }
            if (first && prev) {
                prev->R = first;
                first->L = prev;
            }
        }
    }

    free(row_first);
    free(row_last);

    /* Allocate solution stack */
    ctx->sol_cap = n_rows > 16 ? n_rows : 16;
    ctx->solution = (int*)malloc((size_t)ctx->sol_cap * sizeof(int));
    if (!ctx->solution) {
        dlx_destroy(ctx);
        return NULL;
    }
    ctx->sol_size = 0;
    ctx->found = 0;

    return ctx;
}

/*
 * Free all resources.
 */
void dlx_destroy(dlx_ctx* ctx) {
    if (!ctx) return;
    free(ctx->header);
    free(ctx->columns);
    free(ctx->nodes);
    free(ctx->solution);
    free(ctx);
}

/*
 * Solve the exact cover problem.
 */
int dlx_solve(dlx_ctx* ctx) {
    if (!ctx) return 0;
    ctx->found = 0;
    ctx->sol_size = 0;
    return search(ctx, 0);
}

/*
 * Get the solution.
 */
int* dlx_get_solution(dlx_ctx* ctx, int* size) {
    if (!ctx || !ctx->found) {
        if (size) *size = 0;
        return NULL;
    }
    if (size) *size = ctx->sol_size;
    return ctx->solution;
}
