#include <string.h>
#include "fitsio.h"

#include "fortran.h"


static int get_datatype(int dtype) {
    if (dtype == FLOAT_IMG) return TFLOAT;
    if (dtype == DOUBLE_IMG) return TDOUBLE;
    if (dtype == BYTE_IMG) return TBYTE;
    if (dtype == SHORT_IMG) return TSHORT;
    if (dtype == LONG_IMG) return TLONG;
    if (dtype == LONGLONG_IMG) return TLONGLONG;
    return 0;
}

static void io_error(int status) {
    if (status)
        fits_report_error(stderr, status);
}


typedef struct {
    char * dataname;
    char * varname;
    char * nbdir;
    void * fcube;
    void * data;
    void * var;

    int dtype;
    intptr_t naxes[3];
    long size, planesize;
    size_t bytesize;

    char * wcs;
} cube;


static void wcs_read(fitsfile * file, char ** cards, int * status) {
    (*cards) = malloc(1200);
    for (int i=0; i<1200; ++i)
        (*cards)[i] = ' ';
    char * keys[] = {
            "BUNIT",
            "CRPIX1", "CRPIX2",
            "CD1_1", "CD1_2", "CD2_1", "CD2_2",
            "CUNIT1", "CUNIT2",
            "CTYPE1", "CTYPE2",
            "CSYER1", "CSYER2",
            "CRVAL1", "CRVAL2",
    };
    for (int i=0; i<15; ++i)
        fits_read_card(file, keys[i], &((*cards)[i*80]), status);

    for (int i=0; i<1200; ++i)
        if ((*cards)[i] == 0)
            (*cards)[i] = ' ';
}

static void wcs_write(fitsfile * file, char * cards, int * status) {
    for (int i=0; i<15; ++i)
        fits_write_record(file, &(cards[i*80]), status);
}


static void cube_write_img(cube * c, const char * fname, void * img) {
    fitsfile * file;
    int status = 0;

    long naxes[2];
    naxes[0] = (long) c->naxes[0];
    naxes[1] = (long) c->naxes[1];

    fits_create_file(&file, fname, &status);

    fits_create_img(file, c->dtype, 2, naxes, &status);

    wcs_write(file, c->wcs, &status);

    naxes[0] = 1;
    naxes[1] = 1;

    fits_write_img(file, get_datatype(c->dtype), 1, c->planesize, img, &status);

    fits_close_file(file, &status);
}


static void cube_load(cube * c) {
    fitsfile * file;
    int status = 0, anynul = 0;
    fits_open_file(&file, c->dataname, 0, &status);

    wcs_read(file, &(c->wcs), &status);

    fits_get_img_type(file, &(c->dtype), &status);

    long naxes[3];
    fits_get_img_size(file, 3, naxes, &status);
    c->naxes[0] = (intptr_t) naxes[0]; c->planesize = naxes[0];
    c->naxes[1] = (intptr_t) naxes[1]; c->planesize *= naxes[1];
    c->naxes[2] = (intptr_t) naxes[2]; c->size = c->planesize*naxes[2];

    c->bytesize = (size_t) (c->size * (abs(c->dtype) / 8));

    c->data = malloc(c->bytesize);
    fits_read_img(file, get_datatype(c->dtype), 1, c->size, NULL, c->data, &anynul, &status);
    fits_close_file(file, &status);

    fits_open_file(&file, c->varname, 0, &status);
    c->var = malloc(c->bytesize);
    fits_read_img(file, get_datatype(c->dtype), 1, c->size, NULL, c->var, &anynul, &status);
    fits_close_file(file, &status);

    c->fcube = cube_init(&(c->naxes), c->data, c->var);
}

static void cube_free(cube * c) {
    free(c->data);
    free(c->var);
    free(c->wcs);
}


