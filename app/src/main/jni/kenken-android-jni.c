/*
 * kenken-android-jni.c: JNI bridge for KenKen puzzle generation
 *
 * Exposes native C puzzle generation to Java/Kotlin via JNI. Provides
 * structured error handling with typed response envelopes (OK:/ERR:).
 *
 * Entry points:
 *   - getLevelFromC: Random puzzle generation with configurable difficulty
 *   - getLevelFromAI: Generation from ML-provided Latin square solution
 *
 * SPDX-License-Identifier: MIT
 * SPDX-FileCopyrightText: Copyright (C) 2016 Sergey
 * SPDX-FileCopyrightText: Copyright (C) 2024-2025 Orthogon Contributors
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 */

#include <jni.h>
#include <stdio.h>
#include <string.h>

#include "kenken.h"
#include "kenken_validate.h"
#include "kenken_hints.h"
#include "jni_error_codes.h"

/**
 * Create a structured error response string.
 * Format: "ERR:code:message"
 * Caller must free the returned string with sfree().
 */
static char *jni_make_error(int code, const char *message) {
    size_t len = strlen(JNI_PREFIX_ERR) + 16 + strlen(message) + 1;
    char *result = snewn(len, char);
    snprintf(result, len, JNI_ERR_FMT, code, message);
    return result;
}

/**
 * Create a structured success response string.
 * Format: "OK:payload"
 * Caller must free the returned string with sfree().
 */
static char *jni_make_success(const char *payload) {
    size_t len = strlen(JNI_PREFIX_OK) + strlen(payload) + 1;
    char *result = snewn(len, char);
    strcpy(result, JNI_PREFIX_OK);
    strcat(result, payload);
    return result;
}

JNIEXPORT jstring JNICALL Java_org_yegie_orthogon_KenKenModelBuilder_getLevelFromC(
    JNIEnv *env, jobject instance, jint size, jint diff, jint multOnly, jlong seed, jint modeFlags) {

    /* Validate parameters */
    if (size < 3 || size > 16) {
        char *err = jni_make_error(JNI_ERR_INVALID_PARAMS, "Size must be 3-16");
        jstring retval = (*env)->NewStringUTF(env, err);
        sfree(err);
        return retval;
    }

    if (diff < 0 || diff > 4) {
        char *err = jni_make_error(JNI_ERR_INVALID_PARAMS, "Difficulty must be 0-4");
        jstring retval = (*env)->NewStringUTF(env, err);
        sfree(err);
        return retval;
    }

    struct game_params params;

    params.w = size;
    params.diff = diff;
    params.multiplication_only = multOnly;
    params.mode_flags = modeFlags;

    /* The seed is used as a set of bytes, so passing the content
     * of the memory occupied by the jlong we have. */
    long lseed = seed;
    struct random_state *rs = random_new((char *)&lseed, sizeof(long));

    char *aux = NULL;
    int interactive = 0;

    char *level = new_game_desc(&params, rs, &aux, interactive);

    if (level == NULL) {
        random_free(rs);
        char *err = jni_make_error(JNI_ERR_GENERATION_FAIL, "Native generation returned null");
        jstring retval = (*env)->NewStringUTF(env, err);
        sfree(err);
        return retval;
    }

    /* Combine level and aux into payload */
    char *combined = snewn((strlen(level) + strlen(aux) + 2), char);
    if (combined == NULL) {
        random_free(rs);
        sfree(level);
        sfree(aux);
        char *err = jni_make_error(JNI_ERR_MEMORY, "Failed to allocate combined buffer");
        jstring retval = (*env)->NewStringUTF(env, err);
        sfree(err);
        return retval;
    }

    strcpy(combined, level);
    strcat(combined, ";");
    strcat(combined, aux);

    /* Wrap in success envelope */
    char *result = jni_make_success(combined);
    jstring retval = (*env)->NewStringUTF(env, result);

    random_free(rs);
    sfree(level);
    sfree(combined);
    sfree(aux);
    sfree(result);

    return retval;
}

JNIEXPORT jstring JNICALL Java_org_yegie_orthogon_KenKenModelBuilder_getLevelFromAI(
    JNIEnv *env, jobject instance, jint size, jint diff, jint multOnly, jlong seed,
    jintArray gridFlat, jint modeFlags) {

    /* Validate parameters */
    if (size < 3 || size > 16) {
        char *err = jni_make_error(JNI_ERR_INVALID_PARAMS, "Size must be 3-16");
        jstring retval = (*env)->NewStringUTF(env, err);
        sfree(err);
        return retval;
    }

    struct game_params params;
    params.w = size;
    params.diff = diff;
    params.multiplication_only = multOnly;
    params.mode_flags = modeFlags;

    long lseed = seed;
    struct random_state *rs = random_new((char *)&lseed, sizeof(long));

    /* Convert Java int array to C digit array */
    jsize len = (*env)->GetArrayLength(env, gridFlat);
    if (len != size * size) {
        random_free(rs);
        char *err = jni_make_error(JNI_ERR_GRID_SIZE, "Grid array length does not match size*size");
        jstring retval = (*env)->NewStringUTF(env, err);
        sfree(err);
        return retval;
    }

    jint *body = (*env)->GetIntArrayElements(env, gridFlat, 0);
    if (body == NULL) {
        random_free(rs);
        char *err = jni_make_error(JNI_ERR_MEMORY, "Failed to access grid array");
        jstring retval = (*env)->NewStringUTF(env, err);
        sfree(err);
        return retval;
    }

    digit *input_grid = snewn(len, digit);
    if (input_grid == NULL) {
        (*env)->ReleaseIntArrayElements(env, gridFlat, body, 0);
        random_free(rs);
        char *err = jni_make_error(JNI_ERR_MEMORY, "Failed to allocate input grid");
        jstring retval = (*env)->NewStringUTF(env, err);
        sfree(err);
        return retval;
    }

    for (int i = 0; i < len; i++) {
        input_grid[i] = (digit)body[i];
    }
    (*env)->ReleaseIntArrayElements(env, gridFlat, body, 0);

    char *aux = NULL;
    int interactive = 0;

    /* Try to generate a game description from the provided grid */
    char *level = new_game_desc_from_grid(&params, rs, input_grid, &aux, interactive);

    if (level == NULL) {
        sfree(input_grid);
        random_free(rs);
        char *err = jni_make_error(JNI_ERR_INVALID_GRID, "AI grid rejected - not valid for requested difficulty");
        jstring retval = (*env)->NewStringUTF(env, err);
        sfree(err);
        return retval;
    }

    /* Combine level and aux into payload */
    char *combined = snewn((strlen(level) + strlen(aux) + 2), char);
    if (combined == NULL) {
        sfree(input_grid);
        random_free(rs);
        sfree(level);
        sfree(aux);
        char *err = jni_make_error(JNI_ERR_MEMORY, "Failed to allocate combined buffer");
        jstring retval = (*env)->NewStringUTF(env, err);
        sfree(err);
        return retval;
    }

    strcpy(combined, level);
    strcat(combined, ";");
    strcat(combined, aux);

    /* Wrap in success envelope */
    char *result = jni_make_success(combined);
    jstring retval = (*env)->NewStringUTF(env, result);

    random_free(rs);
    sfree(input_grid);
    sfree(level);
    sfree(combined);
    sfree(aux);
    sfree(result);

    return retval;
}

void fatal(char *fmt, ...) {
    va_list ap;

    fprintf(stderr, "fatal error: ");

    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);

    fprintf(stderr, "\n");
    exit(1);
}

static void memswap(void *av, void *bv, int size) {
    char tmpbuf[512];
    char *a = av, *b = bv;

    while (size > 0) {
        size_t thislen = min((size_t)size, sizeof(tmpbuf));
        memcpy(tmpbuf, a, thislen);
        memcpy(a, b, thislen);
        memcpy(b, tmpbuf, thislen);
        a += thislen;
        b += thislen;
        size -= thislen;
    }
}

void shuffle(void *array, int nelts, int eltsize, random_state *rs) {
    char *carray = (char *)array;
    unsigned long i;

    for (i = (unsigned long)nelts; i-- > 1;) {
        unsigned long j = random_upto(rs, i + 1);
        if (j != i)
            memswap(carray + eltsize * i, carray + eltsize * j, eltsize);
    }
}

/*
 * Validation JNI Entry Points
 * ---------------------------
 * Expose real-time validation for UI error highlighting.
 */

/**
 * Validate grid and return error flags for each cell.
 *
 * @param size Grid dimension (NxN)
 * @param gridFlat Current cell values as flat array (0 = empty)
 * @param dsfFlat DSF array for cage membership
 * @param cluesFlat Cage clues with operation in upper bits
 * @param modeFlags Mode flags (e.g., MODE_KILLER)
 * @return IntArray of error flags (VALID_ERR_ROW | VALID_ERR_COL | VALID_ERR_CAGE)
 */
JNIEXPORT jintArray JNICALL Java_org_yegie_orthogon_KenKenValidator_validateGrid(
    JNIEnv *env, jclass clazz, jint size, jintArray gridFlat,
    jintArray dsfFlat, jlongArray cluesFlat, jint modeFlags) {

    (void)clazz;  /* Unused static method receiver */

    int n = size * size;

    /* Get array elements */
    jint *grid_body = (*env)->GetIntArrayElements(env, gridFlat, 0);
    jint *dsf_body = (*env)->GetIntArrayElements(env, dsfFlat, 0);
    jlong *clues_body = (*env)->GetLongArrayElements(env, cluesFlat, 0);

    if (!grid_body || !dsf_body || !clues_body) {
        if (grid_body) (*env)->ReleaseIntArrayElements(env, gridFlat, grid_body, 0);
        if (dsf_body) (*env)->ReleaseIntArrayElements(env, dsfFlat, dsf_body, 0);
        if (clues_body) (*env)->ReleaseLongArrayElements(env, cluesFlat, clues_body, 0);
        return NULL;
    }

    /* Convert to native types */
    digit *grid = snewn(n, digit);
    int *dsf = snewn(n, int);
    long *clues = snewn(n, long);
    int *errors = snewn(n, int);

    for (int i = 0; i < n; i++) {
        grid[i] = (digit)grid_body[i];
        dsf[i] = dsf_body[i];
        clues[i] = (long)clues_body[i];
    }

    /* Release Java arrays */
    (*env)->ReleaseIntArrayElements(env, gridFlat, grid_body, 0);
    (*env)->ReleaseIntArrayElements(env, dsfFlat, dsf_body, 0);
    (*env)->ReleaseLongArrayElements(env, cluesFlat, clues_body, 0);

    /* Set up validation context */
    validate_ctx ctx;
    ctx.w = size;
    ctx.grid = grid;
    ctx.dsf = dsf;
    ctx.clues = clues;
    ctx.mode_flags = modeFlags;

    /* Validate */
    kenken_validate_grid(&ctx, errors);

    /* Create result array */
    jintArray result = (*env)->NewIntArray(env, n);
    if (result) {
        (*env)->SetIntArrayRegion(env, result, 0, n, errors);
    }

    /* Cleanup */
    sfree(grid);
    sfree(dsf);
    sfree(clues);
    sfree(errors);

    return result;
}

/**
 * Check if puzzle is complete and valid.
 *
 * @return 1 if complete and valid, 0 otherwise
 */
JNIEXPORT jint JNICALL Java_org_yegie_orthogon_KenKenValidator_isComplete(
    JNIEnv *env, jclass clazz, jint size, jintArray gridFlat,
    jintArray dsfFlat, jlongArray cluesFlat, jint modeFlags) {

    (void)clazz;

    int n = size * size;

    jint *grid_body = (*env)->GetIntArrayElements(env, gridFlat, 0);
    jint *dsf_body = (*env)->GetIntArrayElements(env, dsfFlat, 0);
    jlong *clues_body = (*env)->GetLongArrayElements(env, cluesFlat, 0);

    if (!grid_body || !dsf_body || !clues_body) {
        if (grid_body) (*env)->ReleaseIntArrayElements(env, gridFlat, grid_body, 0);
        if (dsf_body) (*env)->ReleaseIntArrayElements(env, dsfFlat, dsf_body, 0);
        if (clues_body) (*env)->ReleaseLongArrayElements(env, cluesFlat, clues_body, 0);
        return 0;
    }

    digit *grid = snewn(n, digit);
    int *dsf = snewn(n, int);
    long *clues = snewn(n, long);

    for (int i = 0; i < n; i++) {
        grid[i] = (digit)grid_body[i];
        dsf[i] = dsf_body[i];
        clues[i] = (long)clues_body[i];
    }

    (*env)->ReleaseIntArrayElements(env, gridFlat, grid_body, 0);
    (*env)->ReleaseIntArrayElements(env, dsfFlat, dsf_body, 0);
    (*env)->ReleaseLongArrayElements(env, cluesFlat, clues_body, 0);

    validate_ctx ctx;
    ctx.w = size;
    ctx.grid = grid;
    ctx.dsf = dsf;
    ctx.clues = clues;
    ctx.mode_flags = modeFlags;

    int result = kenken_is_complete(&ctx);

    sfree(grid);
    sfree(dsf);
    sfree(clues);

    return result;
}

/*
 * Hint System JNI Entry Points
 * ----------------------------
 * Provide step-by-step hints for solving puzzles.
 */

/**
 * Get the next hint for the puzzle.
 *
 * @param size Grid dimension
 * @param gridFlat Current cell values (0 = empty)
 * @param dsfFlat DSF array for cage membership
 * @param cluesFlat Cage clues with operation in upper bits
 * @param solutionFlat Known solution (can be empty for partial hints)
 * @param modeFlags Mode flags
 * @return IntArray: [hint_type, cell, row, col, value, cage_root, related_pos]
 *         Returns null if no hint available
 */
JNIEXPORT jintArray JNICALL Java_org_yegie_orthogon_KenKenHints_getNextHint(
    JNIEnv *env, jclass clazz, jint size, jintArray gridFlat,
    jintArray dsfFlat, jlongArray cluesFlat, jintArray solutionFlat, jint modeFlags) {

    (void)clazz;

    int n = size * size;

    jint *grid_body = (*env)->GetIntArrayElements(env, gridFlat, 0);
    jint *dsf_body = (*env)->GetIntArrayElements(env, dsfFlat, 0);
    jlong *clues_body = (*env)->GetLongArrayElements(env, cluesFlat, 0);
    jint *solution_body = solutionFlat ?
        (*env)->GetIntArrayElements(env, solutionFlat, 0) : NULL;

    if (!grid_body || !dsf_body || !clues_body) {
        if (grid_body) (*env)->ReleaseIntArrayElements(env, gridFlat, grid_body, 0);
        if (dsf_body) (*env)->ReleaseIntArrayElements(env, dsfFlat, dsf_body, 0);
        if (clues_body) (*env)->ReleaseLongArrayElements(env, cluesFlat, clues_body, 0);
        if (solution_body) (*env)->ReleaseIntArrayElements(env, solutionFlat, solution_body, 0);
        return NULL;
    }

    digit *grid = snewn(n, digit);
    int *dsf = snewn(n, int);
    long *clues = snewn(n, long);
    digit *solution = solutionFlat ? snewn(n, digit) : NULL;

    for (int i = 0; i < n; i++) {
        grid[i] = (digit)grid_body[i];
        dsf[i] = dsf_body[i];
        clues[i] = (long)clues_body[i];
        if (solution) solution[i] = (digit)solution_body[i];
    }

    (*env)->ReleaseIntArrayElements(env, gridFlat, grid_body, 0);
    (*env)->ReleaseIntArrayElements(env, dsfFlat, dsf_body, 0);
    (*env)->ReleaseLongArrayElements(env, cluesFlat, clues_body, 0);
    if (solution_body) (*env)->ReleaseIntArrayElements(env, solutionFlat, solution_body, 0);

    hint_ctx ctx;
    ctx.w = size;
    ctx.grid = grid;
    ctx.dsf = dsf;
    ctx.clues = clues;
    ctx.mode_flags = modeFlags;
    ctx.solution = solution;

    hint_result result;
    jintArray ret = NULL;

    if (kenken_get_hint(&ctx, &result)) {
        ret = (*env)->NewIntArray(env, 7);
        if (ret) {
            jint data[7] = {
                result.hint_type,
                result.cell,
                result.row,
                result.col,
                result.value,
                result.cage_root,
                result.related_pos
            };
            (*env)->SetIntArrayRegion(env, ret, 0, 7, data);
        }
    }

    sfree(grid);
    sfree(dsf);
    sfree(clues);
    if (solution) sfree(solution);

    return ret;
}

/**
 * Explain why a specific cell has a particular value.
 */
JNIEXPORT jintArray JNICALL Java_org_yegie_orthogon_KenKenHints_explainCell(
    JNIEnv *env, jclass clazz, jint size, jint cell, jintArray gridFlat,
    jintArray dsfFlat, jlongArray cluesFlat, jintArray solutionFlat, jint modeFlags) {

    (void)clazz;

    int n = size * size;

    jint *grid_body = (*env)->GetIntArrayElements(env, gridFlat, 0);
    jint *dsf_body = (*env)->GetIntArrayElements(env, dsfFlat, 0);
    jlong *clues_body = (*env)->GetLongArrayElements(env, cluesFlat, 0);
    jint *solution_body = solutionFlat ?
        (*env)->GetIntArrayElements(env, solutionFlat, 0) : NULL;

    if (!grid_body || !dsf_body || !clues_body) {
        if (grid_body) (*env)->ReleaseIntArrayElements(env, gridFlat, grid_body, 0);
        if (dsf_body) (*env)->ReleaseIntArrayElements(env, dsfFlat, dsf_body, 0);
        if (clues_body) (*env)->ReleaseLongArrayElements(env, cluesFlat, clues_body, 0);
        if (solution_body) (*env)->ReleaseIntArrayElements(env, solutionFlat, solution_body, 0);
        return NULL;
    }

    digit *grid = snewn(n, digit);
    int *dsf = snewn(n, int);
    long *clues = snewn(n, long);
    digit *solution = solutionFlat ? snewn(n, digit) : NULL;

    for (int i = 0; i < n; i++) {
        grid[i] = (digit)grid_body[i];
        dsf[i] = dsf_body[i];
        clues[i] = (long)clues_body[i];
        if (solution) solution[i] = (digit)solution_body[i];
    }

    (*env)->ReleaseIntArrayElements(env, gridFlat, grid_body, 0);
    (*env)->ReleaseIntArrayElements(env, dsfFlat, dsf_body, 0);
    (*env)->ReleaseLongArrayElements(env, cluesFlat, clues_body, 0);
    if (solution_body) (*env)->ReleaseIntArrayElements(env, solutionFlat, solution_body, 0);

    hint_ctx ctx;
    ctx.w = size;
    ctx.grid = grid;
    ctx.dsf = dsf;
    ctx.clues = clues;
    ctx.mode_flags = modeFlags;
    ctx.solution = solution;

    hint_result result;
    jintArray ret = NULL;

    if (kenken_explain_cell(&ctx, cell, &result)) {
        ret = (*env)->NewIntArray(env, 7);
        if (ret) {
            jint data[7] = {
                result.hint_type,
                result.cell,
                result.row,
                result.col,
                result.value,
                result.cage_root,
                result.related_pos
            };
            (*env)->SetIntArrayRegion(env, ret, 0, 7, data);
        }
    }

    sfree(grid);
    sfree(dsf);
    sfree(clues);
    if (solution) sfree(solution);

    return ret;
}
