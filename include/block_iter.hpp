
#pragma once

#include <cassert>

// BlockIterator: kernels can use this to
//   - quickly go to some sequential index
//   - move to next location
struct BlockIterator {
    struct blkloop {
        int cnt;
        int sz_m;
        int sz_n;
    };

    int idx[16];  // index at each level of blocks
    const blkloop *bloops;
    int num_bloops;

    int M;
    int N;

    int m;
    int n;
    int seq;
    bool reach_end;

    BlockIterator() = default;

    void reset(const blkloop * _bloops, int _num_bloops, int _M, int _N) {
        assert(_num_bloops <= 16);
        bloops = _bloops;
        num_bloops = _num_bloops;
        M = _M;
        N = _N;
        // reset coordinates to sequence index
        for(int i = 0; i < num_bloops; i++)
            idx[i] = 0;
        seq = 0;
        m = 0;
        n = 0;
        reach_end = false;
    }
    // update coordinates
    bool next() {
        if (reach_end)
            return false;
        int carry_on = 1;
        for(int i = 0; i < num_bloops; i++) {
            const auto & bl = bloops[i];
            if (idx[i] == (bl.cnt - 1)) {
                // carry-on on block boundary, no contribution to m/n
                m -= idx[i] * bl.sz_m;
                n -= idx[i] * bl.sz_n;
                idx[i] = 0;
            } else {
                // carry-on on matrix boundary
                if (m + bl.sz_m >= M || n + bl.sz_n >= N) {
                    m -= idx[i] * bl.sz_m;
                    n -= idx[i] * bl.sz_n;
                    idx[i] = 0;
                } else {
                    idx[i]++;
                    m += bl.sz_m;
                    n += bl.sz_n;
                    carry_on = 0;
                    break;
                }
            }
        }
        seq++;
        if (carry_on) {
            // after reach_end
            //  - seq has the number of blocks
            //  - idx are all zeros
            reach_end = true;
            return false;
        }
        return true;
    }
};
