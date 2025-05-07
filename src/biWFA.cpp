#include "headers/commons.h"
#include "headers/biWFA.h"
#include <cstdlib>
#include <chrono>

#define OFFSET_NULL (int32_t)(INT32_MIN/2)
#define max_alignment_steps 10000
#define num_wavefronts 300
#define wf_length 256
#define penalty_mismatch 4
#define penalty_gap_open 6
#define penalty_gap_ext 2
#define NOW std::chrono::high_resolution_clock::now();

int32_t ****matrix_wf; // Matrix used to allocate the batch of wavefronts used during the algorithm

// wavefront_extend_end2end_max()
void extend_max(bool *finish, int score, int32_t *max_ak, wf_components_t *wf, int max_score_scope, int alignment_k, int32_t alignment_offset) {
    if (wf->mwavefronts[score].offsets == NULL) {
        if (wf->alignment.num_null_steps > max_score_scope) {
            *finish = true;
        } else {
            *finish = false;
        }
    } else {
        // wavefront_extend_matches_packed_end2end_max()
        bool end_reached = false;
        int32_t max_antidiag_loc = 0;
        int k;
        for (k = wf->mwavefronts[score].lo; k <= wf->mwavefronts[score].hi; ++k) {
            int32_t offset = wf->mwavefronts[score].offsets[k];
            if (offset == OFFSET_NULL) {continue;}
            // wavefront_extend_matches_kernel_blockwise() or wavefront_extend_matches_kernel()
            uint64_t *pattern_blocks = (uint64_t*)(wf->alignment.pattern + offset - k);
            uint64_t *text_blocks = (uint64_t*)(wf->alignment.text + offset);

            uint64_t cmp = (*pattern_blocks) ^ (*text_blocks);
            while (__builtin_expect(cmp==0,0)) {
                offset += 8;
                ++pattern_blocks;
                ++text_blocks;
                cmp = *pattern_blocks ^ *text_blocks;
            }
            // Count equal characters
            int equal_right_bits = __builtin_ctzl(cmp);
            int equal_chars = equal_right_bits / 8;
            offset += equal_chars;
            // Return extended offset
            wf->mwavefronts[score].offsets[k] = offset;

            int32_t antidiag = (2 * wf->mwavefronts[score].offsets[k]) - k;
            if (max_antidiag_loc < antidiag) {
                max_antidiag_loc = antidiag;
            }
        }
        *max_ak = max_antidiag_loc;

        // wavefront_termination_end2end()
        // End component matrixM
        if (wf->mwavefronts[score].lo > alignment_k || alignment_k > wf->mwavefronts[score].hi) {
            end_reached = false;
        } else {
            int32_t moffset = wf->mwavefronts[score].offsets[alignment_k];
            if (moffset < alignment_offset) {
                end_reached = false;
            } else {
                end_reached = true;
            }
        }

        if (end_reached) {
            *finish = true;
        } else {
            *finish = false;
        }
    }
}

// wavefront_extend_end2end()
void extend(bool *finish, int score, wf_components_t *wf, int max_score_scope, int alignment_k, int32_t alignment_offset) {
    if (wf->mwavefronts[score].offsets == NULL) {
        if (wf->alignment.num_null_steps > max_score_scope) {
            *finish = true;
        } else {
            *finish = false;
        }
    } else {
        // wavefront_extend_matches_packed_end2end()
        bool end_reached = false;
        int k;
        for (k = wf->mwavefronts[score].lo; k <= wf->mwavefronts[score].hi; ++k) {
            int32_t offset = wf->mwavefronts[score].offsets[k];
            if (offset == OFFSET_NULL) {continue;}
            // wavefront_extend_matches_kernel_blockwise() or wavefront_extend_matches_kernel()
            uint64_t* pattern_blocks = (uint64_t*)(wf->alignment.pattern + offset - k);
            uint64_t* text_blocks = (uint64_t*)(wf->alignment.text + offset);

            uint64_t cmp = (*pattern_blocks) ^ (*text_blocks);
            while (__builtin_expect(cmp==0,0)) {
                offset += 8;
                ++pattern_blocks;
                ++text_blocks;
                cmp = *pattern_blocks ^ *text_blocks;
            }
            // Count equal characters
            int equal_right_bits = __builtin_ctzl(cmp);
            int equal_chars = equal_right_bits / 8;
            offset += equal_chars;
            // Return extended offset
            wf->mwavefronts[score].offsets[k] = offset;
        }

        // wavefront_termination_end2end()
        // End component matrixM
        if (wf->mwavefronts[score].lo > alignment_k || alignment_k > wf->mwavefronts[score].hi) {
            end_reached = false;
        } else {
            int32_t moffset = wf->mwavefronts[score].offsets[alignment_k];
            if (moffset < alignment_offset) {
                end_reached = false;
            } else {
                end_reached = true;
            }
        }
        if (end_reached) {
            *finish = true;
        } else {
            *finish = false;
        }
    }
}

void nextWF(int *score, wf_components_t *wf, bool forward, int max_score_scope, int text_len, int pattern_len) {
    // Compute next (s+1) wavefront
    ++(*score);

    // wavefront_compute_affine()
    int mismatch = *score - penalty_mismatch;
    int gap_open = *score - penalty_gap_open - penalty_gap_ext;
    int gap_extend = *score - penalty_gap_ext;

    // wavefront_compute_get_mwavefront()
    wf->mwavefronts[*score].offsets = matrix_wf[forward][0][*score] + wf_length/2;
    wf->mwavefronts[*score].null = false;
    wf->iwavefronts[*score].offsets = matrix_wf[forward][1][*score] + wf_length/2;
    wf->iwavefronts[*score].null = false;
    wf->dwavefronts[*score].offsets = matrix_wf[forward][2][*score] + wf_length/2;
    wf->dwavefronts[*score].null = false;

    wf_t in_mwavefront_misms = (mismatch < 0 || wf->mwavefronts[mismatch].offsets == NULL || wf->mwavefronts[mismatch].null) ? wf->wavefront_null : wf->mwavefronts[mismatch];
    wf_t in_mwavefront_open = (gap_open < 0 || wf->mwavefronts[gap_open].offsets == NULL || wf->mwavefronts[gap_open].null) ? wf->wavefront_null : wf->mwavefronts[gap_open];
    wf_t in_iwavefront_ext = (gap_extend < 0 || wf->iwavefronts[gap_extend].offsets == NULL || wf->iwavefronts[gap_extend].null) ? wf->wavefront_null : wf->iwavefronts[gap_extend];
    wf_t in_dwavefront_ext = (gap_extend < 0 || wf->dwavefronts[gap_extend].offsets == NULL || wf->dwavefronts[gap_extend].null) ? wf->wavefront_null : wf->dwavefronts[gap_extend];

    if (in_mwavefront_misms.null && in_mwavefront_open.null && in_iwavefront_ext.null && in_dwavefront_ext.null) {
        // wavefront_compute_allocate_output_null()
        wf->alignment.num_null_steps++; // Increment null-steps
        // Nullify Wavefronts
        wf->mwavefronts[*score].null = true;
        wf->iwavefronts[*score].null = true;
        wf->dwavefronts[*score].null = true;
    } else {
        wf->alignment.num_null_steps = 0;
        int hi, lo;

        // wavefront_compute_limits_input()
        int min_lo = in_mwavefront_misms.lo;
        int max_hi = in_mwavefront_misms.hi;

        if (!in_mwavefront_open.null && min_lo > (in_mwavefront_open.lo - 1)) min_lo = in_mwavefront_open.lo - 1;
        if (!in_mwavefront_open.null && max_hi < (in_mwavefront_open.hi + 1)) max_hi = in_mwavefront_open.hi + 1;
        if (!in_iwavefront_ext.null && min_lo > (in_iwavefront_ext.lo + 1)) min_lo = in_iwavefront_ext.lo + 1;
        if (!in_iwavefront_ext.null && max_hi < (in_iwavefront_ext.hi + 1)) max_hi = in_iwavefront_ext.hi + 1;
        if (!in_dwavefront_ext.null && min_lo > (in_dwavefront_ext.lo - 1)) min_lo = in_dwavefront_ext.lo - 1;
        if (!in_dwavefront_ext.null && max_hi < (in_dwavefront_ext.hi - 1)) max_hi = in_dwavefront_ext.hi - 1;
        lo = min_lo;
        hi = max_hi;

        // wavefront_compute_allocate_output()
        int effective_lo = lo;
        int effective_hi = hi;

        // wavefront_compute_limits_output()
        int eff_lo = effective_lo - (max_score_scope + 1);
        int eff_hi = effective_hi + (max_score_scope + 1);
        effective_lo = MIN(eff_lo, wf->alignment.historic_min_lo);
        effective_hi = MAX(eff_hi, wf->alignment.historic_max_hi);
        wf->alignment.historic_min_lo = effective_lo;
        wf->alignment.historic_max_hi = effective_hi;

        // Allocate M-Wavefront
        wf->mwavefronts[*score].lo = lo;
        wf->mwavefronts[*score].hi = hi;
        // Allocate I1-Wavefront
        if (!in_mwavefront_open.null || !in_iwavefront_ext.null) {
            wf->iwavefronts[*score].lo = lo;
            wf->iwavefronts[*score].hi = hi;
        } else {
            wf->iwavefronts[*score].null = true;
        }
        // Allocate D1-Wavefront
        if (!in_mwavefront_open.null || !in_dwavefront_ext.null) {
            wf->dwavefronts[*score].lo = lo;
            wf->dwavefronts[*score].hi = hi;
        } else {
            wf->dwavefronts[*score].null = true;
        }

        // wavefront_compute_init_ends()
        // Init wavefront ends
        bool m_misms_null = in_mwavefront_misms.null;
        bool m_gap_null = in_mwavefront_open.null;
        bool i_ext_null = in_iwavefront_ext.null;
        bool d_ext_null = in_dwavefront_ext.null;

        if (!m_misms_null) {
            // wavefront_compute_init_ends_wf_higher()
            if (in_mwavefront_misms.wf_elements_init_max >= hi) {
            } else {
                // Initialize lower elements
                int max_init = MAX(in_mwavefront_misms.wf_elements_init_max, in_mwavefront_misms.hi);
                int k;
                for (k = max_init + 1; k <= hi; ++k) {
                    in_mwavefront_misms.offsets[k] = OFFSET_NULL;
                }
                // Set new maximum
                in_mwavefront_misms.wf_elements_init_max = hi;
            }
            // wavefront_compute_init_ends_wf_lower()
            if (in_mwavefront_misms.wf_elements_init_min <= lo) {
            } else {
                // Initialize lower elements
                int min_init = MIN(in_mwavefront_misms.wf_elements_init_min, in_mwavefront_misms.lo);
                int k;
                for (k = lo; k < min_init; ++k) {
                    in_mwavefront_misms.offsets[k] = OFFSET_NULL;
                }
                // Set new minimum
                in_mwavefront_misms.wf_elements_init_min = lo;
            }
        }
        if (!m_gap_null) {
            // wavefront_compute_init_ends_wf_higher()
            if (in_mwavefront_open.wf_elements_init_max >= hi + 1) {
            } else {
                // Initialize lower elements
                int max_init = MAX(in_mwavefront_open.wf_elements_init_max, in_mwavefront_open.hi);
                int k;
                for (k = max_init + 1; k <= hi + 1; ++k) {
                    in_mwavefront_open.offsets[k] = OFFSET_NULL;
                }
                // Set new maximum
                in_mwavefront_open.wf_elements_init_max = hi + 1;
            }
            // wavefront_compute_init_ends_wf_lower()
            if (in_mwavefront_open.wf_elements_init_min <= lo - 1) {
            } else {
                // Initialize lower elements
                int min_init = MIN(in_mwavefront_open.wf_elements_init_min, in_mwavefront_open.lo);
                int k;
                for (k = lo - 1; k < min_init; ++k) {
                    in_mwavefront_open.offsets[k] = OFFSET_NULL;
                }
                // Set new minimum
                in_mwavefront_open.wf_elements_init_min = lo - 1;
            }
        }
        if (!i_ext_null) {
            // wavefront_compute_init_ends_wf_higher()
            if (in_iwavefront_ext.wf_elements_init_max >= hi) {
            } else {
                // Initialize lower elements
                int max_init = MAX(in_iwavefront_ext.wf_elements_init_max, in_iwavefront_ext.hi);
                int k;
                for (k = max_init + 1; k <= hi; ++k) {
                    in_iwavefront_ext.offsets[k] = OFFSET_NULL;
                }
                // Set new maximum
                in_iwavefront_ext.wf_elements_init_max = hi;
            }
            // wavefront_compute_init_ends_wf_lower()
            if (in_iwavefront_ext.wf_elements_init_min <= lo - 1) {
            } else {
                // Initialize lower elements
                int min_init = MIN(in_iwavefront_ext.wf_elements_init_min, in_iwavefront_ext.lo);
                int k;
                for (k = lo - 1; k < min_init; ++k) {
                    in_iwavefront_ext.offsets[k] = OFFSET_NULL;
                }
                // Set new minimum
                in_iwavefront_ext.wf_elements_init_min = lo - 1;
            }
        }
        if (!d_ext_null) {
            // wavefront_compute_init_ends_wf_higher()
            if (in_dwavefront_ext.wf_elements_init_max >= hi + 1) {
            } else {
                // Initialize lower elements
                int max_init = MAX(in_dwavefront_ext.wf_elements_init_max, in_dwavefront_ext.hi);
                int k;
                for (k = max_init + 1; k <= hi + 1; ++k) {
                    in_dwavefront_ext.offsets[k] = OFFSET_NULL;
                }
                // Set new maximum
                in_dwavefront_ext.wf_elements_init_max = hi + 1;
            }
            // wavefront_compute_init_ends_wf_lower()
            if (in_dwavefront_ext.wf_elements_init_min <= lo) {
            } else {
                // Initialize lower elements
                int min_init = MIN(in_dwavefront_ext.wf_elements_init_min, in_dwavefront_ext.lo);
                int k;
                for (k = lo; k < min_init; ++k) {
                    in_dwavefront_ext.offsets[k] = OFFSET_NULL;
                }
                // Set new minimum
                in_dwavefront_ext.wf_elements_init_min = lo;
            }
        }

        //wavefront_compute_affine_idm()
        int32_t* m_misms = in_mwavefront_misms.offsets;
        int32_t* m_open = in_mwavefront_open.offsets;
        int32_t* i_ext = in_iwavefront_ext.offsets;
        int32_t* d_ext = in_dwavefront_ext.offsets;

        // Compute-Next kernel loop
        int k;
        for (k = lo; k <= hi; ++k) {
            // Update I1
            int32_t ins_o = m_open[k - 1];
            int32_t ins_e = i_ext[k - 1];
            int32_t ins = MAX(ins_o, ins_e) + 1;
            wf->iwavefronts[*score].offsets[k] = ins;

            // Update D1
            int32_t del_o = m_open[k + 1];
            int32_t del_e = d_ext[k + 1];
            int32_t del = MAX(del_o, del_e);
            wf->dwavefronts[*score].offsets[k] = del;

            // Update M
            int32_t misms = m_misms[k] + 1;
            int32_t max = MAX(del, MAX(misms, ins));

            // Adjust offset out of boundaries
            uint32_t h = max;
            uint32_t v = max - k;
            if (h > text_len) max = OFFSET_NULL;
            if (v > pattern_len) max = OFFSET_NULL;
            wf->mwavefronts[*score].offsets[k] = max;
        }

        // wavefront_compute_process_ends()
        if (wf->mwavefronts[*score].offsets) {
            // wavefront_compute_trim_ends()
            int k;
            int lo = wf->mwavefronts[*score].lo;
            for (k = wf->mwavefronts[*score].hi; k >= lo; --k) {
                // Fetch offset
                int32_t offset = wf->mwavefronts[*score].offsets[k];
                // Check boundaries
                uint32_t h = offset; // Make unsigned to avoid checking negative
                uint32_t v = offset - k; // Make unsigned to avoid checking negative
                if (h <= text_len && v <= pattern_len) break;
            }
            wf->mwavefronts[*score].hi = k; // Set new hi
            wf->mwavefronts[*score].wf_elements_init_max = k;
            // Trim from lo
            int hi = wf->mwavefronts[*score].hi;
            for (k = wf->mwavefronts[*score].lo ; k <= hi; ++k) {
                // Fetch offset
                int32_t offset = wf->mwavefronts[*score].offsets[k];
                // Check boundaries
                uint32_t h = offset; // Make unsigned to avoid checking negative
                uint32_t v = offset - k; // Make unsigned to avoid checking negative
                if (h <= text_len && v <= pattern_len) break;
            }
            wf->mwavefronts[*score].lo = k; // Set new lo
            wf->mwavefronts[*score].wf_elements_init_min = k;
            wf->mwavefronts[*score].null = (wf->mwavefronts[*score].lo > wf->mwavefronts[*score].hi);
        }
        if (wf->iwavefronts[*score].offsets) {
            // wavefront_compute_trim_ends()
            int k;
            int lo = wf->iwavefronts[*score].lo;
            for (k = wf->iwavefronts[*score].hi; k >= lo; --k) {
                // Fetch offset
                int32_t offset = wf->iwavefronts[*score].offsets[k];
                // Check boundaries
                uint32_t h = offset; // Make unsigned to avoid checking negative
                uint32_t v = offset - k; // Make unsigned to avoid checking negative
                if (h <= text_len && v <= pattern_len) break;
            }
            wf->iwavefronts[*score].hi = k; // Set new hi
            wf->iwavefronts[*score].wf_elements_init_max = k;
            // Trim from lo
            int hi = wf->iwavefronts[*score].hi;
            for (k = wf->iwavefronts[*score].lo; k <= hi; ++k) {
                // Fetch offset
                int32_t offset = wf->iwavefronts[*score].offsets[k];
                // Check boundaries
                uint32_t h = offset; // Make unsigned to avoid checking negative
                uint32_t v = offset - k; // Make unsigned to avoid checking negative
                if (h <= text_len && v <= pattern_len) break;
            }
            wf->iwavefronts[*score].lo = k; // Set new lo
            wf->iwavefronts[*score].wf_elements_init_min = k;
            wf->iwavefronts[*score].null = (wf->iwavefronts[*score].lo > wf->iwavefronts[*score].hi);
        }

        if (wf->dwavefronts[*score].offsets) {
            // wavefront_compute_trim_ends()
            int k;
            int lo = wf->dwavefronts[*score].lo;
            for (k = wf->dwavefronts[*score].hi; k >= lo ; --k) {
                // Fetch offset
                int32_t offset = wf->dwavefronts[*score].offsets[k];
                // Check boundaries
                uint32_t h = offset; // Make unsigned to avoid checking negative
                uint32_t v = offset - k; // Make unsigned to avoid checking negative
                if (h <= text_len && v <= pattern_len) break;
            }
            wf->dwavefronts[*score].hi = k; // Set new hi
            wf->dwavefronts[*score].wf_elements_init_max = k;
            // Trim from lo
            int hi = wf->dwavefronts[*score].hi;
            for (k = wf->dwavefronts[*score].lo; k <= hi; ++k) {
                // Fetch offset
                int32_t offset = wf->dwavefronts[*score].offsets[k];
                // Check boundaries
                uint32_t h = offset; // Make unsigned to avoid checking negative
                uint32_t v = offset - k; // Make unsigned to avoid checking negative
                if (h <= text_len && v <= pattern_len) break;
            }
            wf->dwavefronts[*score].lo = k; // Set new lo
            wf->dwavefronts[*score].wf_elements_init_min = k;
            wf->dwavefronts[*score].null = (wf->dwavefronts[*score].lo > wf->dwavefronts[*score].hi);
        }
    }
}

// wavefront_bialign_breakpoint_indel2indel()
void breakpoint_indel2indel(int score_0, int score_1, wf_t *dwf_0, wf_t *dwf_1, int *breakpoint_score, int text_len, int pattern_len) {
    int lo_0 = dwf_0->lo;
    int hi_0 = dwf_0->hi;
    int lo_1 = text_len - pattern_len - dwf_1->hi;
    int hi_1 = text_len - pattern_len - dwf_1->lo;

    if (hi_1 < lo_0 || hi_0 < lo_1) return;
    // Compute overlapping interval
    int min_hi = MIN(hi_0, hi_1);
    int max_lo = MAX(lo_0, lo_1);
    int k_0;
    for (k_0 = max_lo; k_0 <= min_hi; k_0++) {
        int k_1 = text_len - pattern_len - k_0;
        // Fetch offsets
        int dh_0 = dwf_0->offsets[k_0];
        int dh_1 = dwf_1->offsets[k_1];
        // Check breakpoint d2d
        if (dh_0 + dh_1 >= text_len && score_0 + score_1 - penalty_gap_open < *breakpoint_score) {
            *breakpoint_score = score_0 + score_1 - penalty_gap_open;
            return;
        }
    }
}

// wavefront_bialign_breakpoint_m2m()
void breakpoint_m2m(int score_0, int score_1, wf_t *mwf_0, wf_t *mwf_1, int *breakpoint_score, int text_len, int pattern_len) {
    // Check wavefronts overlapping
    int lo_0 = mwf_0->lo;
    int hi_0 = mwf_0->hi;
    int lo_1 = text_len - pattern_len - mwf_1->hi;
    int hi_1 = text_len - pattern_len - mwf_1->lo;
    if (hi_1 < lo_0 || hi_0 < lo_1) return;
    // Compute overlapping interval
    int min_hi = MIN(hi_0, hi_1);
    int max_lo = MAX(lo_0, lo_1);
    int k_0;
    for (k_0 = max_lo; k_0 <= min_hi; k_0++) {
        const int k_1 = text_len - pattern_len - k_0;
        // Fetch offsets
        const int mh_0 = mwf_0->offsets[k_0];
        const int mh_1 = mwf_1->offsets[k_1];
        // Check breakpoint m2m
        if ((mh_0 + mh_1 >= text_len) && (score_0 + score_1 < *breakpoint_score)) {
            *breakpoint_score = score_0 + score_1;
            return;
        }
    }
}

// wavefront_bialign_overlap()
void overlap(int score_0, wf_components_t *wf_0, int score_1, wf_components_t *wf_1, int max_score_scope, int *breakpoint_score, int text_len, int pattern_len) {
    // Fetch wavefront-0
    int score_mod_0 = score_0;
    wf_t *mwf_0 = &wf_0->mwavefronts[score_mod_0];

    if (mwf_0 == NULL) return;
    wf_t *d1wf_0 = &wf_0->dwavefronts[score_mod_0];
    wf_t *i1wf_0 = &wf_0->iwavefronts[score_mod_0];

    // Traverse all scores-1
    int i;
    for (i = 0; i < max_score_scope; ++i) {
        // Compute score
        const int score_i = score_1 - i;
        if (score_i < 0) break;
        int score_mod_i = score_i;

        if (score_0 + score_i - penalty_gap_open >= *breakpoint_score) continue;
        // Check breakpoint d2d
        wf_t *d1wf_1 = &wf_1->dwavefronts[score_mod_i];
        if (d1wf_0 != NULL && d1wf_1 != NULL) {
            breakpoint_indel2indel(score_0, score_i, d1wf_0, d1wf_1, breakpoint_score, text_len, pattern_len);
        }
        // Check breakpoint i2i
        wf_t *i1wf_1 = &wf_1->iwavefronts[score_mod_i];
        if (i1wf_0 != NULL && i1wf_1 != NULL) {
            breakpoint_indel2indel(score_0, score_i, i1wf_0, i1wf_1, breakpoint_score, text_len, pattern_len);
        }
        // Check M-breakpoints (indel, edit, gap-linear)
        if (score_0 + score_i >= *breakpoint_score) continue;
        wf_t *mwf_1 = &wf_1->mwavefronts[score_mod_i];
        if (mwf_1 != NULL) {
            breakpoint_m2m(score_0, score_i, mwf_0, mwf_1, breakpoint_score, text_len, pattern_len);
        }
    }
}

// wavefront_bialign() | wavefront_bialign_compute_score()
int main(int argc, char const *argv[]) {
    if(argc != 2) {
        printf("Wrong file name!\n");
        return 0;
    }

    FILE *fp = fopen(argv[1], "r");
    if(fp == NULL) {
        printf("Cannot open file!\n");
        return 1;
    }

    int num_alignments, pattern_len, text_len;
    fscanf(fp, "%d", &num_alignments);
    fscanf(fp, "%d", &pattern_len);
    fscanf(fp, "%d", &text_len);

    char *pattern_f = (char *)malloc(sizeof(char) * pattern_len);
    char *text_f = (char *)malloc(sizeof(char) * text_len);
    int *breakpoint_score = (int *)malloc(sizeof(int) * num_alignments);

    std::chrono::duration<double> time_temp = (std::chrono::duration<double>)0;

    for (int n = 0; n < num_alignments; n++) {
        #pragma omp parallel for
        fscanf(fp, "%s", pattern_f);
        fscanf(fp, "%s", text_f);

        char *pattern_r = (char *) malloc(sizeof(char) * pattern_len);
        char *text_r = (char *) malloc(sizeof(char) * text_len);
        for (int i = 0; i < pattern_len; i++) {
            pattern_r[i] = pattern_f[pattern_len - 1 - i];
        }
        for (int i = 0; i < text_len; i++) {
            text_r[i] = text_f[text_len - 1 - i];
        }

        // Forward wavefront
        wf_components_t wf_f;
        wf_alignment_t alignment_f;
        alignment_f.pattern = pattern_f;
        alignment_f.text = text_f;

        // wavefront_components_dimensions()
        alignment_f.historic_max_hi = 0;
        alignment_f.historic_min_lo = 0;
        wf_f.alignment = alignment_f;

        // Reverse wavefront
        wf_components_t wf_r;
        wf_alignment_t alignment_r;
        alignment_r.pattern = pattern_r;
        alignment_r.text = text_r;

        // wavefront_components_dimensions()
        alignment_r.historic_max_hi = 0;
        alignment_r.historic_min_lo = 0;
        wf_r.alignment = alignment_r;

        // wavefront_components_dimensions_affine()
        int max_score_scope_indel = MAX(penalty_gap_open + penalty_gap_ext, penalty_mismatch) + 1;
        //int abs_seq_diff = ABS(pattern_len - text_len);
        //int max_score_misms = MIN(pattern_len, text_len) * penalty_mismatch;
        //int max_score_indel = penalty_gap_open + abs_seq_diff * penalty_gap_ext;
        //int num_wavefronts = max_score_misms + max_score_indel + 1;
        int max_score_scope = MAX(max_score_scope_indel, penalty_mismatch) + 1;

        // wavefront_compute_limits_output()
        int hi = 0;
        int lo = 0;
        int eff_lo = lo - (max_score_scope + 1);
        int eff_hi = hi + (max_score_scope + 1);
        lo = MIN(eff_lo, 0);
        hi = MAX(eff_hi, 0);
        //int wf_length = hi - lo + 1;

        // wavefront_aligner_init()
        wf_f.alignment.num_null_steps = 0;
        wf_f.alignment.historic_max_hi = hi;
        wf_f.alignment.historic_min_lo = lo;
        wf_r.alignment.num_null_steps = 0;
        wf_r.alignment.historic_max_hi = hi;
        wf_r.alignment.historic_min_lo = lo;

        // allocate a matrix of wavefronts
        matrix_wf = (int32_t ****) (malloc(sizeof(int32_t ****) * 2));
        for (int k = 0; k < 2; k++) { // 0: forward, 1: reverse
            matrix_wf[k] = (int32_t ***) (malloc(sizeof(int32_t ***) * 3));
            for (int i = 0; i < 3; i++) { // 0: matrix_m, 1: matrix_i, 2: matrix_d
                matrix_wf[k][i] = (int32_t **) (malloc(sizeof(int32_t **) * num_wavefronts));
                for (int j = 0; j < num_wavefronts; j++) { // scores
                    matrix_wf[k][i][j] = (int32_t *) (malloc(sizeof(int32_t *) * wf_length));
                    for (int z = 0; z < wf_length; z++) {
                        matrix_wf[k][i][j][z] = OFFSET_NULL;
                    }
                }
            }
        }

        std::chrono::high_resolution_clock::time_point start = NOW;

        // wavefront_components_allocate_wf()
        // Forward wavefront
        wf_f.mwavefronts = new wf_t[num_wavefronts];
        wf_f.iwavefronts = new wf_t[num_wavefronts];
        wf_f.dwavefronts = new wf_t[num_wavefronts];
        wf_f.mwavefronts[0].offsets = matrix_wf[0][0][0] + wf_length / 2;
        wf_f.iwavefronts[0].offsets = matrix_wf[0][1][0] + wf_length / 2;
        wf_f.dwavefronts[0].offsets = matrix_wf[0][2][0] + wf_length / 2;

        // wavefront_init()
        wf_f.mwavefronts[0].null = false;
        wf_f.mwavefronts[0].lo = -1;
        wf_f.mwavefronts[0].hi = 1;
        wf_f.mwavefronts[0].offsets[0] = 0;
        wf_f.mwavefronts[0].wf_elements_init_min = 0;
        wf_f.mwavefronts[0].wf_elements_init_max = 0;

        wf_f.iwavefronts[0].null = true;
        wf_f.iwavefronts[0].lo = -1;
        wf_f.iwavefronts[0].hi = 1;
        wf_f.iwavefronts[0].wf_elements_init_min = 0;
        wf_f.iwavefronts[0].wf_elements_init_max = 0;

        wf_f.dwavefronts[0].null = true;
        wf_f.dwavefronts[0].lo = -1;
        wf_f.dwavefronts[0].hi = 1;
        wf_f.dwavefronts[0].wf_elements_init_min = 0;
        wf_f.dwavefronts[0].wf_elements_init_max = 0;

        // wavefront_init_null()
        wf_f.wavefront_null.null = true;
        wf_f.wavefront_null.lo = 1;
        wf_f.wavefront_null.hi = -1;
        wf_f.wavefront_null.offsets = new int32_t[wf_length];
        wf_f.wavefront_null.wf_elements_init_min = 0;
        wf_f.wavefront_null.wf_elements_init_max = 0;

        // wavefront_components_allocate_wf()
        // Reverse wavefront
        wf_r.mwavefronts = new wf_t[num_wavefronts];
        wf_r.iwavefronts = new wf_t[num_wavefronts];
        wf_r.dwavefronts = new wf_t[num_wavefronts];
        wf_r.mwavefronts[0].offsets = matrix_wf[1][0][0] + wf_length / 2;
        wf_r.iwavefronts[0].offsets = matrix_wf[1][1][0] + wf_length / 2;
        wf_r.dwavefronts[0].offsets = matrix_wf[1][2][0] + wf_length / 2;

        wf_r.mwavefronts[0].null = false;
        wf_r.mwavefronts[0].lo = -1;
        wf_r.mwavefronts[0].hi = 1;
        wf_r.mwavefronts[0].offsets[0] = 0;
        wf_r.mwavefronts[0].wf_elements_init_min = 0;
        wf_r.mwavefronts[0].wf_elements_init_max = 0;

        wf_r.iwavefronts[0].null = true;
        wf_r.iwavefronts[0].lo = -1;
        wf_r.iwavefronts[0].hi = 1;
        wf_r.iwavefronts[0].wf_elements_init_min = 0;
        wf_r.iwavefronts[0].wf_elements_init_max = 0;

        wf_r.dwavefronts[0].null = true;
        wf_r.dwavefronts[0].lo = -1;
        wf_r.dwavefronts[0].hi = 1;
        wf_r.dwavefronts[0].wf_elements_init_min = 0;
        wf_r.dwavefronts[0].wf_elements_init_max = 0;

        // wavefront_init_null()
        wf_r.wavefront_null.null = true;
        wf_r.wavefront_null.lo = 1;
        wf_r.wavefront_null.hi = -1;
        wf_r.wavefront_null.offsets = new int32_t[wf_length];
        wf_r.wavefront_null.wf_elements_init_min = 0;
        wf_r.wavefront_null.wf_elements_init_max = 0;

        for (int i = -(wf_length / 2); i < (wf_length / 2); i++) {
            wf_f.wavefront_null.offsets[i] = OFFSET_NULL;
            wf_r.wavefront_null.offsets[i] = OFFSET_NULL;
        }

        // wavefront_bialign_find_breakpoint()
        int max_antidiag = text_len + pattern_len - 1;
        int score_f = 0;
        int score_r = 0;
        int forward_max_ak = 0;
        int reverse_max_ak = 0;

        // Prepare and perform first bialignment step
        int alignment_score = INT_MAX;

        bool finish = false;
        int alignment_k = text_len - pattern_len;

        extend_max(&finish, score_f, &forward_max_ak, &wf_f, max_score_scope, alignment_k, (int32_t) text_len);
        if (finish) return (0);
        extend_max(&finish, score_r, &reverse_max_ak, &wf_r, max_score_scope, alignment_k, (int32_t) text_len);
        if (finish) return (0);

        // Compute wavefronts of increasing score until both wavefronts overlap
        int max_ak = 0;
        bool last_wf_forward = false;
        while (true) {
            // Compute next wavefront (Forward)
            if (forward_max_ak + reverse_max_ak >= max_antidiag) break;
            nextWF(&score_f, &wf_f, true, max_score_scope, text_len, pattern_len);
            extend_max(&finish, score_f, &max_ak, &wf_f, max_score_scope, alignment_k, (int32_t) text_len);
            if (forward_max_ak < max_ak) forward_max_ak = max_ak;
            last_wf_forward = true;

            // Compute next wavefront (Reverse)
            if (forward_max_ak + reverse_max_ak >= max_antidiag) break;
            nextWF(&score_r, &wf_r, false, max_score_scope, text_len, pattern_len);
            extend_max(&finish, score_r, &max_ak, &wf_r, max_score_scope, alignment_k, (int32_t) text_len);
            if (reverse_max_ak < max_ak) reverse_max_ak = max_ak;
            last_wf_forward = false;
        }

        // Advance until overlap is found
        while (true) {
            if (last_wf_forward) {
                // Check overlapping wavefronts
                const int min_score_r = (score_r > max_score_scope - 1) ? score_r - (max_score_scope - 1) : 0;
                if (score_f + min_score_r - penalty_gap_open >= alignment_score) break;
                overlap(score_f, &wf_f, score_r, &wf_r, max_score_scope, &alignment_score, text_len, pattern_len);

                // Compute next wavefront (Reverse)
                nextWF(&score_r, &wf_r, true, max_score_scope, text_len, pattern_len);
                extend(&finish, score_r, &wf_r, max_score_scope, alignment_k, (int32_t) text_len);
            }
            // Check overlapping wavefronts
            const int min_score_f = (score_f > max_score_scope - 1) ? score_f - (max_score_scope - 1) : 0;
            if (min_score_f + score_r - penalty_gap_open >= alignment_score) break;
            overlap(score_r, &wf_r, score_f, &wf_f, max_score_scope, &alignment_score, text_len, pattern_len);
            // Compute next wavefront (Forward)
            nextWF(&score_f, &wf_f, false, max_score_scope, text_len, pattern_len);
            extend(&finish, score_f, &wf_f, max_score_scope, alignment_k, (int32_t) text_len);

            if (score_r + score_f >= max_alignment_steps) break;
            // Enable always
            last_wf_forward = true;
        }

        alignment_score = -alignment_score;
        breakpoint_score[n] = alignment_score;

        std::chrono::high_resolution_clock::time_point end = NOW;
        time_temp += (end - start);
    }

    long double gcups = pattern_len * text_len;
    gcups /= 1E9;
    gcups /= time_temp.count();
    gcups *= num_alignments;
    printf("CPU Time: %lf\n", time_temp.count());
    printf("Estimated GCUPS GPU: : %Lf\n", gcups);

    /*printf("Scores:\n");
    for(int n = 0; n < num_alignments; n++) {
        printf("%d\n", breakpoint_score[n]);
    }*/

    return 0;
}
