/*
 * keen-android-jni.c: JNI bridge for KenKen puzzle generation
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 * SPDX-FileCopyrightText: Copyright (C) 2016 Sergey
 * SPDX-FileCopyrightText: Copyright (C) 2024-2025 KeenKeen Contributors
 *
 * This file is part of KeenKeen for Android.
 *
 * Provides JNI functions to call the native C puzzle generator from
 * Java/Kotlin code. Supports both traditional random generation and
 * AI-assisted generation from pre-computed Latin squares.
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

#include <jni.h>
#include <stdio.h>
#include <string.h>

#include "keen.h"
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

JNIEXPORT jstring JNICALL Java_org_yegie_keenkeenforandroid_KeenModelBuilder_getLevelFromC(
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

JNIEXPORT jstring JNICALL Java_org_yegie_keenkeenforandroid_KeenModelBuilder_getLevelFromAI(
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
