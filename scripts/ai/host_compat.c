#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include "puzzles.h"

/*
 * Compatability shim for building standalone tools from the puzzle C sources.
 * Provides implementations for functions usually provided by the platform layer.
 */

void fatal(char *fmt, ...)
{
    va_list ap;
    fprintf(stderr, "fatal error: ");
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fprintf(stderr, "\n");
    exit(1);
}

static void memswap(void *av, void *bv, int size)
{
    char tmpbuf[512];
    char *a = av, *b = bv;
    while (size > 0) {
        size_t thislen = ((size_t)size > sizeof(tmpbuf)) ? sizeof(tmpbuf) : (size_t)size;
        memcpy(tmpbuf, a, thislen);
        memcpy(a, b, thislen);
        memcpy(b, tmpbuf, thislen);
        a += thislen;
        b += thislen;
        size -= thislen;
    }
}

void shuffle(void *array, int nelts, int eltsize, random_state *rs)
{
    char *carray = (char *)array;
    int i;
    for (i = nelts; i-- > 1 ;) {
        int j = random_upto(rs, i+1);
        if (j != i)
            memswap(carray + eltsize * i, carray + eltsize * j, eltsize);
    }
}

