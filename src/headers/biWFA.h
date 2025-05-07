#ifndef BIWFA_H
#define BIWFA_H

#define num_wavefronts 50
#define wf_length 1024

#define OFFSET_NULL (int32_t)(INT32_MIN/2)
#define NUM_THREADS 32
#define max_alignment_steps 10000
#define penalty_mismatch 4
#define penalty_gap_open 6
#define penalty_gap_ext 2
#define NOW std::chrono::high_resolution_clock::now();

typedef struct {
    bool null;
    int lo;
    int hi;
    int32_t *offsets;
    int wf_elements_init_max;
    int wf_elements_init_min;
} wf_t;

typedef struct {
    char *pattern;
    char *text;
    int num_null_steps;
    int historic_max_hi;
    int historic_min_lo;
} wf_alignment_t;

typedef struct {
    wf_t *mwavefronts;
    wf_t *dwavefronts;
    wf_t *iwavefronts;
    wf_t wavefront_null;
    wf_alignment_t alignment;
} wf_components_t;

#endif 