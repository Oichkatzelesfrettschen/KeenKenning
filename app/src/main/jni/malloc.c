/*
 * malloc.c: safe wrappers around malloc, realloc, free, strdup
 *
 * SPDX-License-Identifier: MIT
 * SPDX-FileCopyrightText: Copyright (C) 2004-2024 Simon Tatham
 *
 * From Simon Tatham's Portable Puzzle Collection
 * https://www.chiark.greenend.org.uk/~sgtatham/puzzles/
 *
 * Permission is hereby granted, free of charge, to any person
 * obtaining a copy of this software and associated documentation
 * files (the "Software"), to deal in the Software without
 * restriction, including without limitation the rights to use,
 * copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following
 * conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
 * OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR
 * ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
 * CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include <stdlib.h>
#include <string.h>

#include "puzzles.h"

/*
 * smalloc should guarantee to return a useful pointer - Halibut
 * can do nothing except die when it's out of memory anyway.
 */
void* smalloc(size_t size) {
    void* p;
    p = malloc(size);
    if (!p) fatal("out of memory");
    return p;
}

/*
 * sfree should guaranteeably deal gracefully with freeing NULL
 */
void sfree(void* p) {
    if (p) {
        free(p);
    }
}

/*
 * srealloc should guaranteeably be able to realloc NULL
 */
void* srealloc(void* p, size_t size) {
    void* q;
    if (p) {
        q = realloc(p, size);
    } else {
        q = malloc(size);
    }
    if (!q) fatal("out of memory");
    return q;
}

/*
 * dupstr is like strdup, but with the never-return-NULL property
 * of smalloc (and also reliably defined in all environments :-)
 */
char* dupstr(const char* s) {
    char* r = smalloc(1 + strlen(s));
    strcpy(r, s);
    return r;
}
