#ifndef INCUBATOR_FORTRAN_H
#define INCUBATOR_FORTRAN_H

void * cube_init(void * dims, void * data, void * var);
void cube_flats(void * cube);
void cube_get_flats(void * cube, void ** white, void ** red, void ** green, void ** blue, void ** inv_var);
void cube_deallocate(void * cube);
void cube_print(void * cube);

void muselet_step1(void * cube, intptr_t left, intptr_t cw, intptr_t lwsize, double * lw, void (void *, float *, intptr_t *), void * c_cube);

#endif //INCUBATOR_FORTRAN_H
