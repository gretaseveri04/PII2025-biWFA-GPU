extern "C" {
	#include "wavefront/wavefront_align.h"
}

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include "headers/commons.h"
#include "headers/biWFA.h"
#include <chrono>

#define CHECK(call)                                                                     \
{                                                                                     \
	const cudaError_t err = call;                                                     \
	if (err != cudaSuccess)                                                           \
	{                                                                                 \
		printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
		exit(EXIT_FAILURE);                                                           \
	}                                                                                 \
}

#define CHECK_KERNELCALL()                                                            \
{                                                                                     \
	const cudaError_t err = cudaGetLastError();                                       \
	if (err != cudaSuccess)                                                           \
	{                                                                                 \
		printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
		exit(EXIT_FAILURE);                                                           \
	}                                                                                 \
}

__device__ void extend_max(bool *finish, const int score, int32_t *max_ak, wf_components_t *wf, const int max_score_scope, const int alignment_k, const int32_t alignment_offset, const int pattern_len) {
    if (wf->mwavefronts[score%num_wavefronts].offsets == NULL) {
        if (wf->alignment.num_null_steps > max_score_scope) {
            *finish = true;
        } else {
            *finish = false;
        }
    } else {
        // wavefront_extend_matches_packed_end2end_max()
        bool end_reached = false;
        int32_t max_antidiag_loc = 0;
        
        // Iterate over all wavefront offsets
        int k_start = wf->mwavefronts[score%num_wavefronts].lo;
        int k_end = wf->mwavefronts[score%num_wavefronts].hi;
        
        int tidx = threadIdx.x;
        
        for (int k = k_start; k <= k_end; ++k) {
            int32_t offset = wf->mwavefronts[score%num_wavefronts].offsets[k];
            if (offset == OFFSET_NULL) continue;
            
            // wavefront_extend_matches_kernel_blockwise() or wavefront_extend_matches_kernel()
            int equal_chars = 0;
            for (int i = offset; i < pattern_len; i++) {
                if((i - k) >= 0 && (i - k) < pattern_len) {
                    if (wf->alignment.pattern[i - k] == wf->alignment.text[i]) {
                        equal_chars++;
                    } else break;
                }
            }
            offset += equal_chars;
            
            // Return extended offset
            wf->mwavefronts[score%num_wavefronts].offsets[k] = offset;
            
            // Calculate antidiagonal and update max if needed
            int32_t antidiag = (2 * offset) - k;
            if (max_antidiag_loc < antidiag) {
                max_antidiag_loc = antidiag;
            }
        }
        
        // Update the max antidiagonal location
        *max_ak = max_antidiag_loc;
        
        // wavefront_termination_end2end()
        if (wf->mwavefronts[score%num_wavefronts].lo > alignment_k || alignment_k > wf->mwavefronts[score%num_wavefronts].hi) {
            end_reached = false;
        } else {
            int32_t moffset = wf->mwavefronts[score%num_wavefronts].offsets[alignment_k];
            if (moffset < alignment_offset) {
                end_reached = false;
            } else {
                end_reached = true;
            }
        }
        
        *finish = end_reached;
    }
}

__device__ void extend(bool *finish, const int score, const wf_components_t *wf, const int max_score_scope, const int alignment_k, const int32_t alignment_offset, const int pattern_len) {
    wf_t *mwf = &wf->mwavefronts[score % num_wavefronts];
    
    if (mwf->offsets == NULL) {
        *finish = (wf->alignment.num_null_steps > max_score_scope);
        return;
    }

    int lo = mwf->lo;
    int hi = mwf->hi;
    int k = lo + threadIdx.x;

    int32_t offset = 0;
    if (k <= hi) {
        offset = mwf->offsets[k];

        for (int i = offset; i < pattern_len; ++i) {
            int pattern_pos = i - k;
            int text_pos = i;

            if (pattern_pos < 0 || pattern_pos >= pattern_len) break;
            if (wf->alignment.pattern[pattern_pos] != wf->alignment.text[text_pos]) break;

            ++offset;
        }

        mwf->offsets[k] = offset;
    }

    __syncthreads();

    if (threadIdx.x == 0) {
        bool end_reached = false;
        if (alignment_k >= lo && alignment_k <= hi) {
            int32_t moffset = mwf->offsets[alignment_k];
            end_reached = (moffset >= alignment_offset);
        }
        *finish = end_reached;
    }
}

__device__ void nextWF(int *score, wf_components_t *wf, const bool forward, const int max_score_scope, const int text_len, const int pattern_len, int32_t *matrix_wf_m_g, int32_t *matrix_wf_i_g, int32_t *matrix_wf_d_g) {
    // Compute next (s+1) wavefront
    ++(*score);

    int score_mod = *score%num_wavefronts;

    // wavefront_compute_affine()
    int mismatch = *score - penalty_mismatch;
    int gap_open = *score - penalty_gap_open - penalty_gap_ext;
    int gap_extend = *score - penalty_gap_ext;

    // wavefront_compute_get_mwavefront()
    if((*score / num_wavefronts) > 0) {
        // Resetting old wavefronts' values
        wf->mwavefronts[score_mod].lo = -1;
        wf->mwavefronts[score_mod].hi = 1;
        wf->iwavefronts[score_mod].lo = -1;
        wf->iwavefronts[score_mod].hi = 1;
        wf->dwavefronts[score_mod].lo = -1;
        wf->dwavefronts[score_mod].hi = 1;
    }
    wf->mwavefronts[score_mod].offsets = matrix_wf_m_g + (num_wavefronts * wf_length * blockIdx.x) + (score_mod * wf_length) + wf_length/2;
    wf->mwavefronts[score_mod].null = false;
    wf->iwavefronts[score_mod].offsets = matrix_wf_i_g + (num_wavefronts * wf_length * blockIdx.x) + (score_mod * wf_length) + wf_length/2;
    wf->iwavefronts[score_mod].null = false;
    wf->dwavefronts[score_mod].offsets = matrix_wf_d_g + (num_wavefronts * wf_length * blockIdx.x) + (score_mod * wf_length) + wf_length/2;
    wf->dwavefronts[score_mod].null = false;

    wf_t in_mwavefront_misms = (mismatch < 0 || wf->mwavefronts[mismatch%num_wavefronts].offsets == NULL || wf->mwavefronts[mismatch%num_wavefronts].null) ? wf->wavefront_null : wf->mwavefronts[mismatch%num_wavefronts];
    wf_t in_mwavefront_open = (gap_open < 0 || wf->mwavefronts[gap_open%num_wavefronts].offsets == NULL || wf->mwavefronts[gap_open%num_wavefronts].null) ? wf->wavefront_null : wf->mwavefronts[gap_open%num_wavefronts];
    wf_t in_iwavefront_ext = (gap_extend < 0 || wf->iwavefronts[gap_extend%num_wavefronts].offsets == NULL || wf->iwavefronts[gap_extend%num_wavefronts].null) ? wf->wavefront_null : wf->iwavefronts[gap_extend%num_wavefronts];
    wf_t in_dwavefront_ext = (gap_extend < 0 || wf->dwavefronts[gap_extend%num_wavefronts].offsets == NULL || wf->dwavefronts[gap_extend%num_wavefronts].null) ? wf->wavefront_null : wf->dwavefronts[gap_extend%num_wavefronts];

    if (in_mwavefront_misms.null && in_mwavefront_open.null && in_iwavefront_ext.null && in_dwavefront_ext.null) {
        // wavefront_compute_allocate_output_null()
        wf->alignment.num_null_steps++; // Increment null-steps
        // Nullify Wavefronts
        wf->mwavefronts[score_mod].null = true;
        wf->iwavefronts[score_mod].null = true;
        wf->dwavefronts[score_mod].null = true;
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
        wf->mwavefronts[score_mod].lo = lo;
        wf->mwavefronts[score_mod].hi = hi;
        // Allocate I1-Wavefront
        if (!in_mwavefront_open.null || !in_iwavefront_ext.null) {
            wf->iwavefronts[score_mod].lo = lo;
            wf->iwavefronts[score_mod].hi = hi;
        } else {
            wf->iwavefronts[score_mod].null = true;
        }
        // Allocate D1-Wavefront
        if (!in_mwavefront_open.null || !in_dwavefront_ext.null) {
            wf->dwavefronts[score_mod].lo = lo;
            wf->dwavefronts[score_mod].hi = hi;
        } else {
            wf->dwavefronts[score_mod].null = true;
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
        // Compute-Next kernel loop
        int tidx = threadIdx.x;
        for (int i = lo; i <= hi; i += blockDim.x) {
            int idx = tidx + i;
            if (idx <= hi) {
                // Update I1
                int32_t ins_o = in_mwavefront_open.offsets[idx - 1];
                int32_t ins_e = in_iwavefront_ext.offsets[idx - 1];
                int32_t ins = MAX(ins_o, ins_e) + 1;
                wf->iwavefronts[score_mod].offsets[idx] = ins;

                // Update D1
                int32_t del_o = in_mwavefront_open.offsets[idx + 1];
                int32_t del_e = in_dwavefront_ext.offsets[idx + 1];
                int32_t del = MAX(del_o, del_e);
                wf->dwavefronts[score_mod].offsets[idx] = del;

                // Update M
                int32_t misms = in_mwavefront_misms.offsets[idx] + 1;
                int32_t max = MAX(del, MAX(misms, ins));

                // Adjust offset out of boundaries
                uint32_t h = max;
                uint32_t v = max - idx;
                if (h > text_len) max = OFFSET_NULL;
                if (v > pattern_len) max = OFFSET_NULL;
                wf->mwavefronts[score_mod].offsets[idx] = max;
            }
        }

        // wavefront_compute_process_ends()
        if (wf->mwavefronts[score_mod].offsets) {
            // wavefront_compute_trim_ends()
            int k;
            int lo = wf->mwavefronts[score_mod].lo;
            for (k = wf->mwavefronts[score_mod].hi; k >= lo; --k) {
                // Fetch offset
                int32_t offset = wf->mwavefronts[score_mod].offsets[k];
                // Check boundaries
                uint32_t h = offset; // Make unsigned to avoid checking negative
                uint32_t v = offset - k; // Make unsigned to avoid checking negative
                if (h <= text_len && v <= pattern_len) break;
            }
            wf->mwavefronts[score_mod].hi = k; // Set new hi
            wf->mwavefronts[score_mod].wf_elements_init_max = k;
            // Trim from lo
            int hi = wf->mwavefronts[score_mod].hi;
            for (k = wf->mwavefronts[score_mod].lo ; k <= hi; ++k) {
                // Fetch offset
                int32_t offset = wf->mwavefronts[score_mod].offsets[k];
                // Check boundaries
                uint32_t h = offset; // Make unsigned to avoid checking negative
                uint32_t v = offset - k; // Make unsigned to avoid checking negative
                if (h <= text_len && v <= pattern_len) break;
            }
            wf->mwavefronts[score_mod].lo = k; // Set new lo
            wf->mwavefronts[score_mod].wf_elements_init_min = k;
            wf->mwavefronts[score_mod].null = (wf->mwavefronts[score_mod].lo > wf->mwavefronts[score_mod].hi);
        }
        if (wf->iwavefronts[score_mod].offsets) {
            // wavefront_compute_trim_ends()
            int k;
            int lo = wf->iwavefronts[score_mod].lo;
            for (k = wf->iwavefronts[score_mod].hi; k >= lo; --k) {
                // Fetch offset
                int32_t offset = wf->iwavefronts[score_mod].offsets[k];
                // Check boundaries
                uint32_t h = offset; // Make unsigned to avoid checking negative
                uint32_t v = offset - k; // Make unsigned to avoid checking negative
                if (h <= text_len && v <= pattern_len) break;
            }
            wf->iwavefronts[score_mod].hi = k; // Set new hi
            wf->iwavefronts[score_mod].wf_elements_init_max = k;
            // Trim from lo
            int hi = wf->iwavefronts[score_mod].hi;
            for (k = wf->iwavefronts[score_mod].lo; k <= hi; ++k) {
                // Fetch offset
                int32_t offset = wf->iwavefronts[score_mod].offsets[k];
                // Check boundaries
                uint32_t h = offset; // Make unsigned to avoid checking negative
                uint32_t v = offset - k; // Make unsigned to avoid checking negative
                if (h <= text_len && v <= pattern_len) break;
            }
            wf->iwavefronts[score_mod].lo = k; // Set new lo
            wf->iwavefronts[score_mod].wf_elements_init_min = k;
            wf->iwavefronts[score_mod].null = (wf->iwavefronts[score_mod].lo > wf->iwavefronts[score_mod].hi);
        }
        if (wf->dwavefronts[score_mod].offsets) {
            // wavefront_compute_trim_ends()
            int k;
            int lo = wf->dwavefronts[score_mod].lo;
            for (k = wf->dwavefronts[score_mod].hi; k >= lo ; --k) {
                // Fetch offset
                int32_t offset = wf->dwavefronts[score_mod].offsets[k];
                // Check boundaries
                uint32_t h = offset; // Make unsigned to avoid checking negative
                uint32_t v = offset - k; // Make unsigned to avoid checking negative
                if (h <= text_len && v <= pattern_len) break;
            }
            wf->dwavefronts[score_mod].hi = k; // Set new hi
            wf->dwavefronts[score_mod].wf_elements_init_max = k;
            // Trim from lo
            int hi = wf->dwavefronts[score_mod].hi;
            for (k = wf->dwavefronts[score_mod].lo; k <= hi; ++k) {
                // Fetch offset
                int32_t offset = wf->dwavefronts[score_mod].offsets[k];
                // Check boundaries
                uint32_t h = offset; // Make unsigned to avoid checking negative
                uint32_t v = offset - k; // Make unsigned to avoid checking negative
                if (h <= text_len && v <= pattern_len) break;
            }
            wf->dwavefronts[score_mod].lo = k; // Set new lo
            wf->dwavefronts[score_mod].wf_elements_init_min = k;
            wf->dwavefronts[score_mod].null = (wf->dwavefronts[score_mod].lo > wf->dwavefronts[score_mod].hi);
        }
    }
}

__device__ void breakpoint_indel2indel(const int score_0, const int score_1, const wf_t *dwf_0, const wf_t *dwf_1, int *breakpoint_score, const int text_len, const int pattern_len) {
    int lo_0 = dwf_0->lo;
    int hi_0 = dwf_0->hi;
    int lo_1 = text_len - pattern_len - dwf_1->hi;
    int hi_1 = text_len - pattern_len - dwf_1->lo;

    if (hi_1 < lo_0 || hi_0 < lo_1) return;

    int min_hi = min(hi_0, hi_1);
    int max_lo = max(lo_0, lo_1);

    __shared__ int local_min[NUM_THREADS];
    int tid = threadIdx.x;
    local_min[tid] = INT_MAX;

    for (int k_0 = max_lo + tid; k_0 <= min_hi; k_0 += NUM_THREADS) {
        int k_1 = text_len - pattern_len - k_0;
        int dh_0 = dwf_0->offsets[k_0];
        int dh_1 = dwf_1->offsets[k_1];

        if ((dh_0 + dh_1) >= text_len) {
            int candidate = score_0 + score_1 - penalty_gap_open;
            if (candidate < local_min[tid]) {
                local_min[tid] = candidate;
            }
        }
    }

    __syncthreads();

    if (tid == 0) {
        int min_val = INT_MAX;
        for (int i = 0; i < NUM_THREADS; i++) {
            if (local_min[i] < min_val) {
                min_val = local_min[i];
            }
        }

        if (min_val < *breakpoint_score) {
            *breakpoint_score = min_val;
        }
    }
}

__device__ void breakpoint_m2m(const int score_0, const int score_1, const wf_t *mwf_0, const wf_t *mwf_1, int *breakpoint_score, const int text_len, const int pattern_len) {
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
        if (mh_0 + mh_1 >= text_len && score_0 + score_1 < *breakpoint_score) {
            *breakpoint_score = score_0 + score_1; 
            return;
        }
    }
}

__device__ void overlap(const int score_0, const wf_components_t *wf_0, const int score_1, const wf_components_t *wf_1, const int max_score_scope, int *breakpoint_score, const int text_len, const int pattern_len) {
    // Fetch wavefront-0
    int score_mod_0 = score_0%num_wavefronts;
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
        int score_mod_i = score_i%num_wavefronts;

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

__global__ void biWFA_kernel(char *pattern_f_g, char *text_f_g, char *pattern_r_g, char *text_r_g, int *breakpoint_score_g, 
                      wf_t *mwavefronts_f, wf_t *iwavefronts_f, wf_t *dwavefronts_f, 
                      wf_t *mwavefronts_r, wf_t *iwavefronts_r, wf_t *dwavefronts_r,
                      const int lo_g, const int hi_g, int32_t *offsets_g, 
                      const int *pattern_len_array, const int *text_len_array, 
                      const int *pattern_offsets, const int *text_offsets,
                      const int max_score_scope, int32_t *matrix_wf_m_f,
                      int32_t *matrix_wf_i_f, int32_t *matrix_wf_d_f, 
                      int32_t *matrix_wf_m_r, int32_t *matrix_wf_i_r, int32_t *matrix_wf_d_r) {

    int lo = lo_g;
    int hi = hi_g;
    
    int pattern_len = pattern_len_array[blockIdx.x];
    int text_len = text_len_array[blockIdx.x];
    
    int pattern_offset = pattern_offsets[blockIdx.x];
    int text_offset = text_offsets[blockIdx.x];

    if (pattern_len + text_len > wf_length / 2 - 10) {
        if (threadIdx.x == 0) {
            *(breakpoint_score_g + blockIdx.x) = INT_MAX; 
        }
        return; 
    }

    int wf_matrix_size = num_wavefronts * wf_length;
    for (int i = 0; i < wf_matrix_size; i += blockDim.x) {
        int idx = i + threadIdx.x;
        if (idx < wf_matrix_size) {
            int block_offset = wf_matrix_size * blockIdx.x;
            *(matrix_wf_m_f + block_offset + idx) = OFFSET_NULL;
            *(matrix_wf_i_f + block_offset + idx) = OFFSET_NULL;
            *(matrix_wf_d_f + block_offset + idx) = OFFSET_NULL;
            *(matrix_wf_m_r + block_offset + idx) = OFFSET_NULL;
            *(matrix_wf_i_r + block_offset + idx) = OFFSET_NULL;
            *(matrix_wf_d_r + block_offset + idx) = OFFSET_NULL;
        }
    }
    
    __syncthreads();

    for (int i = 0; i < num_wavefronts; i += blockDim.x) {
        int idx = i + threadIdx.x;
        if (idx < num_wavefronts) {
            int block_offset = num_wavefronts * blockIdx.x;
            
            (mwavefronts_f + block_offset + idx)->null = true;
            (mwavefronts_f + block_offset + idx)->lo = 0;
            (mwavefronts_f + block_offset + idx)->hi = 0;
            (mwavefronts_f + block_offset + idx)->offsets = NULL;
            (mwavefronts_f + block_offset + idx)->wf_elements_init_max = 0;
            (mwavefronts_f + block_offset + idx)->wf_elements_init_min = 0;
            
            (mwavefronts_r + block_offset + idx)->null = true;
            (mwavefronts_r + block_offset + idx)->lo = 0;
            (mwavefronts_r + block_offset + idx)->hi = 0;
            (mwavefronts_r + block_offset + idx)->offsets = NULL;
            (mwavefronts_r + block_offset + idx)->wf_elements_init_max = 0;
            (mwavefronts_r + block_offset + idx)->wf_elements_init_min = 0;

            (iwavefronts_f + block_offset + idx)->null = true;
            (iwavefronts_f + block_offset + idx)->lo = 0;
            (iwavefronts_f + block_offset + idx)->hi = 0;
            (iwavefronts_f + block_offset + idx)->offsets = NULL;
            (iwavefronts_f + block_offset + idx)->wf_elements_init_max = 0;
            (iwavefronts_f + block_offset + idx)->wf_elements_init_min = 0;
            
            (iwavefronts_r + block_offset + idx)->null = true;
            (iwavefronts_r + block_offset + idx)->lo = 0;
            (iwavefronts_r + block_offset + idx)->hi = 0;
            (iwavefronts_r + block_offset + idx)->offsets = NULL;
            (iwavefronts_r + block_offset + idx)->wf_elements_init_max = 0;
            (iwavefronts_r + block_offset + idx)->wf_elements_init_min = 0;

            (dwavefronts_f + block_offset + idx)->null = true;
            (dwavefronts_f + block_offset + idx)->lo = 0;
            (dwavefronts_f + block_offset + idx)->hi = 0;
            (dwavefronts_f + block_offset + idx)->offsets = NULL;
            (dwavefronts_f + block_offset + idx)->wf_elements_init_max = 0;
            (dwavefronts_f + block_offset + idx)->wf_elements_init_min = 0;
            
            (dwavefronts_r + block_offset + idx)->null = true;
            (dwavefronts_r + block_offset + idx)->lo = 0;
            (dwavefronts_r + block_offset + idx)->hi = 0;
            (dwavefronts_r + block_offset + idx)->offsets = NULL;
            (dwavefronts_r + block_offset + idx)->wf_elements_init_max = 0;
            (dwavefronts_r + block_offset + idx)->wf_elements_init_min = 0;
        }
    }

    __syncthreads();

    if (threadIdx.x == 0) {
        wf_components_t wf_f, wf_r;
        wf_alignment_t alignment_f, alignment_r;
        int max_antidiag, score_f, score_r, forward_max_ak, reverse_max_ak, breakpoint_score, alignment_k;
        bool finish;

        alignment_f.pattern = pattern_f_g + pattern_offset;
        alignment_f.text = text_f_g + text_offset;
        alignment_f.historic_max_hi = 0;
        alignment_f.historic_min_lo = 0;
        wf_f.alignment = alignment_f;

        alignment_r.pattern = pattern_r_g + pattern_offset;
        alignment_r.text = text_r_g + text_offset;
        alignment_r.historic_max_hi = 0;
        alignment_r.historic_min_lo = 0;
        wf_r.alignment = alignment_r;

        wf_f.alignment.num_null_steps = 0;
        wf_f.alignment.historic_max_hi = hi;
        wf_f.alignment.historic_min_lo = lo;
        wf_r.alignment.num_null_steps = 0;
        wf_r.alignment.historic_max_hi = hi;
        wf_r.alignment.historic_min_lo = lo;

        int block_offset = num_wavefronts * blockIdx.x;
        wf_f.mwavefronts = mwavefronts_f + block_offset;
        wf_f.iwavefronts = iwavefronts_f + block_offset;
        wf_f.dwavefronts = dwavefronts_f + block_offset;

        int matrix_block_offset = num_wavefronts * wf_length * blockIdx.x;
        wf_f.mwavefronts[0].offsets = matrix_wf_m_f + matrix_block_offset + wf_length/2;
        wf_f.iwavefronts[0].offsets = matrix_wf_i_f + matrix_block_offset + wf_length/2;
        wf_f.dwavefronts[0].offsets = matrix_wf_d_f + matrix_block_offset + wf_length/2;

        wf_f.mwavefronts[0].null = false;
        wf_f.mwavefronts[0].lo = -1;
        wf_f.mwavefronts[0].hi = 1;
        wf_f.mwavefronts[0].offsets[-1] = OFFSET_NULL;  
        wf_f.mwavefronts[0].offsets[0] = 0;             
        wf_f.mwavefronts[0].offsets[1] = OFFSET_NULL;   
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

        wf_f.wavefront_null.null = true;
        wf_f.wavefront_null.lo = 1;
        wf_f.wavefront_null.hi = -1;
        wf_f.wavefront_null.offsets = offsets_g + wf_length/2;
        wf_f.wavefront_null.wf_elements_init_min = 0;
        wf_f.wavefront_null.wf_elements_init_max = 0;

        wf_r.mwavefronts = mwavefronts_r + block_offset;
        wf_r.iwavefronts = iwavefronts_r + block_offset;
        wf_r.dwavefronts = dwavefronts_r + block_offset;
        
        wf_r.mwavefronts[0].offsets = matrix_wf_m_r + matrix_block_offset + wf_length/2;
        wf_r.iwavefronts[0].offsets = matrix_wf_i_r + matrix_block_offset + wf_length/2;
        wf_r.dwavefronts[0].offsets = matrix_wf_d_r + matrix_block_offset + wf_length/2;

        wf_r.mwavefronts[0].null = false;
        wf_r.mwavefronts[0].lo = -1;
        wf_r.mwavefronts[0].hi = 1;
        wf_r.mwavefronts[0].offsets[-1] = OFFSET_NULL;  
        wf_r.mwavefronts[0].offsets[0] = 0;             
        wf_r.mwavefronts[0].offsets[1] = OFFSET_NULL;   
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

        wf_r.wavefront_null.null = true;
        wf_r.wavefront_null.lo = 1;
        wf_r.wavefront_null.hi = -1;
        wf_r.wavefront_null.offsets = offsets_g + wf_length/2;
        wf_r.wavefront_null.wf_elements_init_min = 0;
        wf_r.wavefront_null.wf_elements_init_max = 0;

        max_antidiag = text_len + pattern_len - 1;
        score_f = 0;
        score_r = 0;
        forward_max_ak = 0;
        reverse_max_ak = 0;

        breakpoint_score = INT_MAX;

        finish = false;
        alignment_k = text_len - pattern_len;

        int iteration_count = 0;
        const int max_iterations = max_alignment_steps; 

        extend_max(&finish, score_f, &forward_max_ak, &wf_f, max_score_scope, alignment_k, (int32_t)text_len, pattern_len);
        if(finish) {
            *(breakpoint_score_g + blockIdx.x) = breakpoint_score;
            return;
        }
        
        extend_max(&finish, score_r, &reverse_max_ak, &wf_r, max_score_scope, alignment_k, (int32_t)text_len, pattern_len);
        if(finish) {
            *(breakpoint_score_g + blockIdx.x) = breakpoint_score;
            return;
        }

        int max_ak;
        bool last_wf_forward;
        max_ak = 0;
        last_wf_forward = false;
        
        while (true) {
            iteration_count++;
            if (iteration_count > max_iterations) break;
            
            if (forward_max_ak + reverse_max_ak >= max_antidiag) break;
            
            nextWF(&score_f, &wf_f, true, max_score_scope, text_len, pattern_len, matrix_wf_m_f + matrix_block_offset, 
                  matrix_wf_i_f + matrix_block_offset, matrix_wf_d_f + matrix_block_offset);
            
            extend_max(&finish, score_f, &max_ak, &wf_f, max_score_scope, alignment_k, (int32_t) text_len, pattern_len);
            if (forward_max_ak < max_ak) forward_max_ak = max_ak;
            last_wf_forward = true;
            
            if (forward_max_ak + reverse_max_ak >= max_antidiag) break;
            
            nextWF(&score_r, &wf_r, false, max_score_scope, text_len, pattern_len, matrix_wf_m_r + matrix_block_offset, 
                  matrix_wf_i_r + matrix_block_offset, matrix_wf_d_r + matrix_block_offset);
            
            extend_max(&finish, score_r, &max_ak, &wf_r, max_score_scope, alignment_k, (int32_t) text_len, pattern_len);
            if (reverse_max_ak < max_ak) reverse_max_ak = max_ak;
            last_wf_forward = false;
        }

        int min_score_f, min_score_r;
        while (true) {
            iteration_count++;
            if (iteration_count > max_iterations) break;
            
            if (last_wf_forward) {
                min_score_r = (score_r > max_score_scope - 1) ? score_r - (max_score_scope - 1) : 0;
                if (score_f + min_score_r - penalty_gap_open >= breakpoint_score) break;
                
                overlap(score_f, &wf_f, score_r, &wf_r, max_score_scope, &breakpoint_score, text_len, pattern_len);
                
                nextWF(&score_r, &wf_r, true, max_score_scope, text_len, pattern_len, matrix_wf_m_r + matrix_block_offset, 
                      matrix_wf_i_r + matrix_block_offset, matrix_wf_d_r + matrix_block_offset);
                
                extend(&finish, score_r, &wf_r, max_score_scope, alignment_k, (int32_t) text_len, pattern_len);
            }

            min_score_f = (score_f > max_score_scope - 1) ? score_f - (max_score_scope - 1) : 0;
            if (min_score_f + score_r - penalty_gap_open >= breakpoint_score) break;
            
            overlap(score_r, &wf_r, score_f, &wf_f, max_score_scope, &breakpoint_score, text_len, pattern_len);
            
            nextWF(&score_f, &wf_f, false, max_score_scope, text_len, pattern_len, matrix_wf_m_f + matrix_block_offset, 
                  matrix_wf_i_f + matrix_block_offset, matrix_wf_d_f + matrix_block_offset);
            
            extend(&finish, score_f, &wf_f, max_score_scope, alignment_k, (int32_t) text_len, pattern_len);

            if (score_r + score_f >= max_alignment_steps) break;
            last_wf_forward = true;
        }

        breakpoint_score = -breakpoint_score;
        *(breakpoint_score_g + blockIdx.x) = breakpoint_score;
    }
}

int main(int argc, char *argv[]) {
    if(argc != 2) {
        printf("Error\n");
        return 1;
    }

    FILE *fp = fopen(argv[1], "r");
    if(fp == NULL) {
        printf("File open error\n");
        return 1;
    }

    int num_alignments;
    fscanf(fp, "%d", &num_alignments);
    
    int *pattern_len_array = (int *)malloc(sizeof(int) * num_alignments);
    int *text_len_array = (int *)malloc(sizeof(int) * num_alignments);
    
    int *pattern_offsets = (int *)malloc(sizeof(int) * num_alignments);
    int *text_offsets = (int *)malloc(sizeof(int) * num_alignments);
    
    int *pattern_len_array_g, *text_len_array_g;
    int *pattern_offsets_g, *text_offsets_g;
    
    int total_pattern_len = 0;
    int total_text_len = 0;
    
    for (int i = 0; i < num_alignments; i++) {
        fscanf(fp, "%d", &pattern_len_array[i]);
        fscanf(fp, "%d", &text_len_array[i]);
        
        pattern_offsets[i] = total_pattern_len;
        text_offsets[i] = total_text_len;
        
        total_pattern_len += pattern_len_array[i];
        total_text_len += text_len_array[i];
    }
    
    int *breakpoint_score, *breakpoint_score_g;
    char *pattern_f, *text_f, *pattern_r, *text_r;
    char *pattern_f_g, *text_f_g, *pattern_r_g, *text_r_g;
    wf_t *mwavefronts_f, *iwavefronts_f, *dwavefronts_f;
    wf_t *mwavefronts_r, *iwavefronts_r, *dwavefronts_r;
    int32_t *matrix_wf_m_f, *matrix_wf_i_f, *matrix_wf_d_f, *matrix_wf_m_r, *matrix_wf_i_r, *matrix_wf_d_r;

    pattern_f = (char *)malloc(sizeof(char) * total_pattern_len);
    text_f = (char *)malloc(sizeof(char) * total_text_len);
    
    for (int i = 0; i < num_alignments; i++) {
        fscanf(fp, "%s", pattern_f + pattern_offsets[i]);
        fscanf(fp, "%s", text_f + text_offsets[i]);
    }

    pattern_r = (char *)malloc(sizeof(char) * total_pattern_len);
    text_r = (char *)malloc(sizeof(char) * total_text_len);
    
    for (int j = 0; j < num_alignments; j++) {
        int pattern_len = pattern_len_array[j];
        int text_len = text_len_array[j];
        int pattern_offset = pattern_offsets[j];
        int text_offset = text_offsets[j];
        
        for (int i = 0; i < pattern_len; i++) {
            pattern_r[pattern_offset + i] = pattern_f[pattern_offset + pattern_len - 1 - i];
        }
        
        for (int i = 0; i < text_len; i++) {
            text_r[text_offset + i] = text_f[text_offset + text_len - 1 - i];
        }
    }

    breakpoint_score = (int *)malloc(sizeof(int) * num_alignments);
    for (int i = 0; i < num_alignments; i++) {
        breakpoint_score[i] = INT_MAX;
    }

    int max_score_scope_indel = MAX(penalty_gap_open + penalty_gap_ext, penalty_mismatch) + 1;
    int max_score_scope = MAX(max_score_scope_indel, penalty_mismatch) + 1;

    int hi = 0;
    int lo = 0;
    int eff_lo = lo - (max_score_scope + 1);
    int eff_hi = hi + (max_score_scope + 1);
    lo = MIN(eff_lo, 0);
    hi = MAX(eff_hi, 0);

    int32_t *offsets, *offsets_g;
    offsets = (int32_t *)malloc(sizeof(int32_t) * wf_length);
    for(int i = 0; i < wf_length; i++) {
        offsets[i] = OFFSET_NULL;
    }

    CHECK(cudaSetDevice(0));
    
    CHECK(cudaMalloc(&pattern_f_g, sizeof(char) * total_pattern_len));
    CHECK(cudaMalloc(&pattern_r_g, sizeof(char) * total_pattern_len));
    CHECK(cudaMalloc(&text_f_g, sizeof(char) * total_text_len));
    CHECK(cudaMalloc(&text_r_g, sizeof(char) * total_text_len));
    
    CHECK(cudaMalloc(&pattern_len_array_g, sizeof(int) * num_alignments));
    CHECK(cudaMalloc(&text_len_array_g, sizeof(int) * num_alignments));
    CHECK(cudaMalloc(&pattern_offsets_g, sizeof(int) * num_alignments));
    CHECK(cudaMalloc(&text_offsets_g, sizeof(int) * num_alignments));
    
    CHECK(cudaMalloc(&breakpoint_score_g, sizeof(int) * num_alignments));
    CHECK(cudaMalloc(&matrix_wf_m_f, sizeof(int32_t) * num_wavefronts * wf_length * num_alignments));
    CHECK(cudaMalloc(&matrix_wf_i_f, sizeof(int32_t) * num_wavefronts * wf_length * num_alignments));
    CHECK(cudaMalloc(&matrix_wf_d_f, sizeof(int32_t) * num_wavefronts * wf_length * num_alignments));
    CHECK(cudaMalloc(&matrix_wf_m_r, sizeof(int32_t) * num_wavefronts * wf_length * num_alignments));
    CHECK(cudaMalloc(&matrix_wf_i_r, sizeof(int32_t) * num_wavefronts * wf_length * num_alignments));
    CHECK(cudaMalloc(&matrix_wf_d_r, sizeof(int32_t) * num_wavefronts * wf_length * num_alignments));
    CHECK(cudaMalloc(&mwavefronts_f, sizeof(wf_t) * num_wavefronts * num_alignments));
    CHECK(cudaMalloc(&iwavefronts_f, sizeof(wf_t) * num_wavefronts * num_alignments));
    CHECK(cudaMalloc(&dwavefronts_f, sizeof(wf_t) * num_wavefronts * num_alignments));
    CHECK(cudaMalloc(&mwavefronts_r, sizeof(wf_t) * num_wavefronts * num_alignments));
    CHECK(cudaMalloc(&iwavefronts_r, sizeof(wf_t) * num_wavefronts * num_alignments));
    CHECK(cudaMalloc(&dwavefronts_r, sizeof(wf_t) * num_wavefronts * num_alignments));
    CHECK(cudaMalloc(&offsets_g, sizeof(int32_t) * wf_length));

    CHECK(cudaMemcpy(breakpoint_score_g, breakpoint_score, sizeof(int) * num_alignments, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(pattern_f_g, pattern_f, sizeof(char) * total_pattern_len, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(pattern_r_g, pattern_r, sizeof(char) * total_pattern_len, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(text_f_g, text_f, sizeof(char) * total_text_len, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(text_r_g, text_r, sizeof(char) * total_text_len, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(pattern_len_array_g, pattern_len_array, sizeof(int) * num_alignments, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(text_len_array_g, text_len_array, sizeof(int) * num_alignments, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(pattern_offsets_g, pattern_offsets, sizeof(int) * num_alignments, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(text_offsets_g, text_offsets, sizeof(int) * num_alignments, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(offsets_g, offsets, sizeof(int32_t) * wf_length, cudaMemcpyHostToDevice));

    dim3 blocksPerGrid(num_alignments, 1, 1);
    dim3 threadsPerBlock(NUM_THREADS, 1, 1);

    std::chrono::high_resolution_clock::time_point start = NOW;
    
    biWFA_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        pattern_f_g, text_f_g, pattern_r_g, text_r_g, 
        breakpoint_score_g, 
        mwavefronts_f, iwavefronts_f, dwavefronts_f, 
        mwavefronts_r, iwavefronts_r, dwavefronts_r, 
        lo, hi, offsets_g, 
        pattern_len_array_g, text_len_array_g,  
        pattern_offsets_g, text_offsets_g,      
        max_score_scope, 
        matrix_wf_m_f, matrix_wf_i_f, matrix_wf_d_f, 
        matrix_wf_m_r, matrix_wf_i_r, matrix_wf_d_r
    );

    CHECK_KERNELCALL();
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    std::chrono::high_resolution_clock::time_point end = NOW;
    std::chrono::duration<double> time_temp = (end - start);

    CHECK(cudaMemcpy(breakpoint_score, breakpoint_score_g, sizeof(int) * num_alignments, cudaMemcpyDeviceToHost));
    
    long double total_cells = 0;
    for (int i = 0; i < num_alignments; i++) {
        total_cells += pattern_len_array[i] * text_len_array[i];
    }
    
    long double gcups = total_cells;
    gcups /= 1E9;
    gcups /= time_temp.count();
    
    printf("GPU Time: %lf\n", time_temp.count());
    printf("Estimated GCUPS GPU: : %Lf\n", gcups);
    printf("\n");

    CHECK(cudaFree(pattern_f_g));
    CHECK(cudaFree(pattern_r_g));
    CHECK(cudaFree(text_f_g));
    CHECK(cudaFree(text_r_g));
    CHECK(cudaFree(pattern_len_array_g));
    CHECK(cudaFree(text_len_array_g));
    CHECK(cudaFree(pattern_offsets_g));
    CHECK(cudaFree(text_offsets_g));
    CHECK(cudaFree(breakpoint_score_g));
    CHECK(cudaFree(matrix_wf_m_f));
    CHECK(cudaFree(matrix_wf_i_f));
    CHECK(cudaFree(matrix_wf_d_f));
    CHECK(cudaFree(matrix_wf_m_r));
    CHECK(cudaFree(matrix_wf_i_r));
    CHECK(cudaFree(matrix_wf_d_r));
    CHECK(cudaFree(mwavefronts_f));
    CHECK(cudaFree(iwavefronts_f));
    CHECK(cudaFree(dwavefronts_f));
    CHECK(cudaFree(mwavefronts_r));
    CHECK(cudaFree(iwavefronts_r));
    CHECK(cudaFree(dwavefronts_r));
    CHECK(cudaFree(offsets_g));

    printf("Alignment scores:\n\n");
    for (int i = 0; i < num_alignments; i++) {
        int pattern_len = pattern_len_array[i];
        int text_len = text_len_array[i];
        int pattern_offset = pattern_offsets[i];
        int text_offset = text_offsets[i];
        
        printf("%.*s\n%.*s : %d\n", 
               pattern_len, &pattern_f[pattern_offset], 
               text_len, &text_f[text_offset], 
               -breakpoint_score[i]);
        printf("\n");
    }

    printf("Checking alignment scores\n");
    
    wavefront_aligner_attr_t attributes = wavefront_aligner_attr_default;
    attributes.distance_metric = gap_affine;
    attributes.affine_penalties.mismatch = 4;
    attributes.affine_penalties.gap_opening = 6;
    attributes.affine_penalties.gap_extension = 2;
    
    wavefront_aligner_t* wf_aligner = wavefront_aligner_new(&attributes);
    
    bool all_correct = true;
    
    for (int i = 0; i < num_alignments; i++) {
        int pattern_len = pattern_len_array[i];
        int text_len = text_len_array[i];
        int pattern_offset = pattern_offsets[i];
        int text_offset = text_offsets[i];
        
        const char* pattern = &pattern_f[pattern_offset];
        const char* text = &text_f[text_offset];
    
        wavefront_align(wf_aligner, pattern, pattern_len, text, text_len);
        int cpu_score = wf_aligner->cigar->score;
        int gpu_score = breakpoint_score[i];  
    
        printf("Alignment %d - CPU score: %d | GPU score: %d\n", i, cpu_score, gpu_score);
    
        if (cpu_score != gpu_score) {
            printf("ERROR on alignment %d: CPU %d != GPU %d\n", i, cpu_score, gpu_score);
            all_correct = false;
        }
    }

    free(pattern_f);
    free(text_f);
    free(pattern_r);
    free(text_r);
    free(breakpoint_score);
    free(offsets);
    free(pattern_len_array);
    free(text_len_array);
    free(pattern_offsets);
    free(text_offsets);

    wavefront_aligner_delete(wf_aligner); 
    
    fclose(fp);
    return 0;
}
