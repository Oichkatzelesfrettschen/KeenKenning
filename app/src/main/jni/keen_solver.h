#ifndef KEEN_SOLVER_H
#define KEEN_SOLVER_H

#include "keen_internal.h"

int keen_solver(int w, int* dsf, clue_t* clues, digit* soln, int maxdiff, int mode_flags);

#endif
