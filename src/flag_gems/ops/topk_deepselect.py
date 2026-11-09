# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""DeepSelect-style single-pass top-k.

Reference: https://github.com/deepseek-ai/DeepSelect
(docs/DeepSelect-deep-dive.md)

A row is scanned once.  The first WIN elements (the "init window") are loaded
into registers and radix-selected there, which yields exactly K candidates and
the initial threshold T without touching the candidate buffer.  The rest of the
row is visited in BLOCK-sized chunks in a pseudo-random order: elements whose
key is > T are appended to the candidate buffer, which is radix-selected back
down to exactly K entries (raising T) whenever it holds K + B2 candidates.
Every input element is read from DRAM once and the extra work is bounded by
O((1 + K/B2) * (K + BLOCK + B2) * log(N / BLOCK)) in expectation.

Short rows degenerate gracefully: if the row fits the init window there is no
scan loop at all.  When the window is small, or K is small relative to it, the
window is reduced with a bitonic top-k in registers, which selects and orders
the candidates in one step and, for rows that fit the window, writes the result
without going through the candidate buffer.

Values are mapped to order-preserving unsigned keys (NaN ranks as the largest,
matching torch.topk); ``largest=False`` complements the key.

Few, long rows are split into segments processed by separate CTAs, and a second
kernel merges the per-segment candidates (the counterpart of DeepSelect's
thread-block-cluster variant).

Only inputs this kernel does not cover (non-float dtypes, k > 4096) fall back to
the legacy FlagGems kernels.
"""

import logging
import math
import os
from typing import NamedTuple

import torch
import triton
import triton.language as tl
from flag_gems.ops.topk import topk as _topk_legacy
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils.triton_version_utils import HAS_TLE

if HAS_TLE:
    import triton.experimental.tle.language as tle
else:
    tle = None

logger = logging.getLogger(__name__)

_DS_MAX_K = 4096
_DS_DTYPES = (torch.float16, torch.float32, torch.bfloat16)

# Bits of the FLAGS constexpr.
_F_LARGEST = tl.constexpr(1)
_F_SORTED = tl.constexpr(2)
_F_WRITE_OUT = tl.constexpr(4)
_F_NARROW = tl.constexpr(8)
_F_WIN_SORT = tl.constexpr(16)
_F_HAS_SCAN = tl.constexpr(32)
_F_WIN_ARGMAX = tl.constexpr(64)
_F_TLE = tl.constexpr(128)
# Only split a row across CTAs when each row would get at least this many.
_DS_MIN_CTAS_PER_ROW = 4


@triton.jit
def _ds_to_key(x, LARGEST: tl.constexpr):
    if x.dtype.primitive_bitwidth == 32:
        bits = x.to(tl.uint32, bitcast=True)
        top: tl.constexpr = 0x80000000
        full: tl.constexpr = 0xFFFFFFFF
    else:
        bits = x.to(tl.uint16, bitcast=True).to(tl.uint32)
        top: tl.constexpr = 0x8000
        full: tl.constexpr = 0xFFFF
    flip = tl.where((bits & top) != 0, full, top).to(tl.uint32)
    key = bits ^ flip
    key = tl.where(x != x, full, key).to(tl.uint32)
    if not LARGEST:
        key = key ^ full
    return key


@triton.jit
def _ds_from_key(key, out_dtype: tl.constexpr, LARGEST: tl.constexpr):
    if out_dtype.primitive_bitwidth == 32:
        top: tl.constexpr = 0x80000000
        full: tl.constexpr = 0xFFFFFFFF
    else:
        top: tl.constexpr = 0x8000
        full: tl.constexpr = 0xFFFF
    if not LARGEST:
        key = key ^ full
    flip = tl.where((key & top) != 0, top, full).to(tl.uint32)
    bits = key ^ flip
    if out_dtype.primitive_bitwidth == 32:
        return bits.to(tl.float32, bitcast=True)
    else:
        return bits.to(tl.uint16).to(out_dtype, bitcast=True)


@triton.jit
def _ds_hist_alloc(NUM_BINS: tl.constexpr, USE_TLE: tl.constexpr):
    """Shared-memory histogram (TLE), or a dummy when TLE is unavailable."""
    if USE_TLE:
        return tle.gpu.alloc(
            [NUM_BINS],
            dtype=tl.int32,
            layout=None,
            scope=tle.gpu.smem,
            nv_mma_shared_layout=False,
        )
    else:
        return 0


@triton.jit
def _ds_hist_zero(hist, NUM_BINS: tl.constexpr, USE_TLE: tl.constexpr):
    if USE_TLE:
        tl.debug_barrier()  # previous readers are done with the counters
        tl.store(
            tle.gpu.local_ptr(hist, (tl.arange(0, NUM_BINS),)),
            tl.zeros([NUM_BINS], dtype=tl.int32),
        )
        tl.debug_barrier()


@triton.jit
def _ds_hist_add(
    acc, hist, digit, valid, NUM_BINS: tl.constexpr, USE_TLE: tl.constexpr
):
    """Count `digit` into the histogram.

    With TLE every element is one shared-memory atomic (as DeepSelect's
    `red.shared.add`); tl.histogram's ballot-per-bit lowering costs ~90
    instructions per element and per pass at 256 bins.
    """
    if USE_TLE:
        tl.atomic_add(
            tle.gpu.local_ptr(hist, (digit,)),
            tl.full(digit.shape, 1, tl.int32),
            mask=valid,
            sem="relaxed",
            scope="cta",
        )
        return acc
    else:
        return acc + tl.histogram(digit, NUM_BINS, mask=valid)


@triton.jit
def _ds_hist_read(acc, hist, NUM_BINS: tl.constexpr, USE_TLE: tl.constexpr):
    if USE_TLE:
        tl.debug_barrier()
        return tl.load(tle.gpu.local_ptr(hist, (tl.arange(0, NUM_BINS),)))
    else:
        return acc


@triton.jit
def _ds_pick_digit(hist, kk):
    """Given a digit histogram and the rank kk still to find, return the digit
    holding the kk-th largest element and the count of elements above it."""
    bins = tl.arange(0, hist.shape[0])
    # suffix[b] = number of candidates whose digit is >= b
    suffix = tl.cumsum(hist, axis=0, reverse=True)
    dsel = tl.max(tl.where(suffix >= kk, bins, 0), axis=0)
    n_gt = tl.sum(tl.where(bins > dsel, hist, 0), axis=0)
    return dsel, n_gt


@triton.jit
def _ds_radix_select_regs(
    key, valid, kk, hist_smem, KEY_BITS: tl.constexpr, USE_TLE: tl.constexpr
):
    """Radix-select the kk-th largest key of a register tensor.

    Returns (thr, n_eq): thr is that key, n_eq is how many elements equal to
    thr must be kept so that exactly kk are selected.
    """
    RADIX_BITS: tl.constexpr = 8
    NUM_BINS: tl.constexpr = 1 << RADIX_BITS
    prefix = tl.full((), 0, tl.uint32)
    pmask = tl.full((), 0, tl.uint32)
    for shift in tl.static_range(KEY_BITS - RADIX_BITS, -1, -RADIX_BITS):
        match = valid & ((key & pmask) == prefix)
        digit = ((key >> shift) & (NUM_BINS - 1)).to(tl.int32)
        _ds_hist_zero(hist_smem, NUM_BINS, USE_TLE)
        hist = tl.zeros([NUM_BINS], dtype=tl.int32)
        hist = _ds_hist_add(hist, hist_smem, digit, match, NUM_BINS, USE_TLE)
        hist = _ds_hist_read(hist, hist_smem, NUM_BINS, USE_TLE)
        dsel, n_gt = _ds_pick_digit(hist, kk)
        kk = kk - n_gt
        prefix = prefix | (dsel.to(tl.uint32) << shift)
        pmask = pmask | (tl.full((), NUM_BINS - 1, tl.uint32) << shift)
    return prefix, kk


@triton.jit
def _ds_pack(key, idx):
    """Candidate pair as one int64: key in the high word, index in the low."""
    return ((key.to(tl.uint64) << 32) | idx.to(tl.uint32).to(tl.uint64)).to(
        tl.int64, bitcast=True
    )


@triton.jit
def _ds_unpack(p):
    u = p.to(tl.uint64, bitcast=True)
    return (u >> 32).to(tl.uint32), (u & 0xFFFFFFFF).to(tl.uint32).to(tl.int32)


@triton.jit
def _ds_radix_select_buf(
    buf,
    cnt,
    kk,
    hist_smem,
    KEY_BITS: tl.constexpr,
    TILE: tl.constexpr,
    USE_TLE: tl.constexpr,
):
    """Same as _ds_radix_select_regs, for packed pairs living in buf[0:cnt]."""
    RADIX_BITS: tl.constexpr = 8
    NUM_BINS: tl.constexpr = 1 << RADIX_BITS
    offs = tl.arange(0, TILE)
    prefix = tl.full((), 0, tl.uint32)
    pmask = tl.full((), 0, tl.uint32)
    for shift in tl.static_range(KEY_BITS - RADIX_BITS, -1, -RADIX_BITS):
        _ds_hist_zero(hist_smem, NUM_BINS, USE_TLE)
        hist = tl.zeros([NUM_BINS], dtype=tl.int32)
        for t in range(0, cnt, TILE):
            o = t + offs
            m = o < cnt
            key, _ = _ds_unpack(tl.load(buf + o, mask=m, other=0))
            match = m & ((key & pmask) == prefix)
            digit = ((key >> shift) & (NUM_BINS - 1)).to(tl.int32)
            hist = _ds_hist_add(hist, hist_smem, digit, match, NUM_BINS, USE_TLE)
        hist = _ds_hist_read(hist, hist_smem, NUM_BINS, USE_TLE)
        dsel, n_gt = _ds_pick_digit(hist, kk)
        kk = kk - n_gt
        prefix = prefix | (dsel.to(tl.uint32) << shift)
        pmask = pmask | (tl.full((), NUM_BINS - 1, tl.uint32) << shift)
    return prefix, kk


@triton.jit
def _ds_select_positions(key, valid, thr, n_eq, eq_seen):
    """Selection mask and output slots for one tile of keys.

    Keeps every key > thr and the first n_eq keys == thr (counting eq_seen
    already passed in earlier tiles).  A single scan carries both counters:
    the low 16 bits count ">", the high 16 bits count "==".
    Returns (sel, slot_in_tile, n_sel_tile, n_eq_tile).
    """
    gt = valid & (key > thr)
    eq = valid & (key == thr)
    gt_i = gt.to(tl.int32)
    eq_i = eq.to(tl.int32)
    cs = tl.cumsum(gt_i + (eq_i << 16), axis=0)
    gt_before = (cs & 0xFFFF) - gt_i
    eq_before = eq_seen + (cs >> 16) - eq_i
    sel = gt | (eq & (eq_before < n_eq))
    base_eq = tl.minimum(eq_seen, n_eq)
    slot = gt_before + tl.minimum(eq_before, n_eq) - base_eq
    n_gt_tile = tl.sum(gt_i, axis=0)
    n_eq_tile = tl.sum(eq_i, axis=0)
    n_sel_tile = n_gt_tile + tl.minimum(eq_seen + n_eq_tile, n_eq) - base_eq
    return sel, slot, n_sel_tile, n_eq_tile


@triton.jit
def _ds_compact(src, dst, cnt, thr, n_eq, TILE: tl.constexpr):
    """Copy every pair whose key is > thr, plus the first n_eq pairs whose key
    is == thr, from src[0:cnt] to dst (which must not alias src)."""
    offs = tl.arange(0, TILE)
    out = tl.full((), 0, tl.int32)
    eq_seen = tl.full((), 0, tl.int32)
    for t in range(0, cnt, TILE):
        o = t + offs
        m = o < cnt
        p = tl.load(src + o, mask=m, other=0)
        key, _ = _ds_unpack(p)
        sel, slot, n_sel, n_eq_t = _ds_select_positions(key, m, thr, n_eq, eq_seen)
        tl.store(dst + out + slot, p, mask=sel)
        out += n_sel
        eq_seen += n_eq_t


@triton.jit
def _ds_sort_pairs(keys, idx, valid, NARROW: tl.constexpr):
    """Sort pairs by (key desc, index asc), padding (invalid lanes) last.

    Key and index are packed into one word so a single sort orders both: a
    32-bit word when a 16-bit key and a 16-bit index fit (NARROW), 64-bit
    otherwise.
    """
    if NARROW:
        packed = (keys.to(tl.uint32) << 16) | (0xFFFF - idx.to(tl.uint32))
        packed = tl.where(valid, packed, 0)
        packed = tl.sort(packed, dim=0, descending=True)
        keys = packed >> 16
        idx = (0xFFFF - (packed & 0xFFFF)).to(tl.int32)
    else:
        packed = (keys.to(tl.uint64) << 32) | (0xFFFFFFFF - idx.to(tl.uint32)).to(
            tl.uint64
        )
        packed = tl.where(valid, packed, 0)
        packed = tl.sort(packed, dim=0, descending=True)
        keys = (packed >> 32).to(tl.uint32)
        idx = (0xFFFFFFFF - (packed & 0xFFFFFFFF).to(tl.uint32)).to(tl.int32)
    return keys, idx


@triton.jit
def _ds_topk_pairs(keys, idx, valid, K_PAD: tl.constexpr, NARROW: tl.constexpr):
    """Bitonic top-K_PAD of a register tensor, sorted by (key desc, index asc).
    Invalid lanes rank last."""
    if NARROW:
        packed = (keys.to(tl.uint32) << 16) | (0xFFFF - idx.to(tl.uint32))
        packed = tl.where(valid, packed, 0)
        packed = tl.topk(packed, K_PAD)
        keys = packed >> 16
        idx = (0xFFFF - (packed & 0xFFFF)).to(tl.int32)
    else:
        packed = (keys.to(tl.uint64) << 32) | (0xFFFFFFFF - idx.to(tl.uint32)).to(
            tl.uint64
        )
        packed = tl.where(valid, packed, 0)
        packed = tl.topk(packed, K_PAD)
        keys = (packed >> 32).to(tl.uint32)
        idx = (0xFFFFFFFF - (packed & 0xFFFFFFFF).to(tl.uint32)).to(tl.int32)
    return keys, idx


@triton.jit
def _ds_argmax_pairs(
    keys, idx, valid, K: tl.constexpr, K_PAD: tl.constexpr, NARROW: tl.constexpr
):
    """Top-K by K rounds of max-reduction, for tiny K.

    Each round costs one block reduction, far less than the cross-thread
    exchanges of a bitonic network over the whole window.  The packed word
    holds the index, so every round removes exactly one element.
    """
    ok = tl.arange(0, K_PAD)
    if NARROW:
        packed = (keys.to(tl.uint32) << 16) | (0xFFFF - idx.to(tl.uint32))
        packed = tl.where(valid, packed, 0)
        top = tl.zeros([K_PAD], dtype=tl.uint32)
    else:
        packed = (keys.to(tl.uint64) << 32) | (0xFFFFFFFF - idx.to(tl.uint32)).to(
            tl.uint64
        )
        packed = tl.where(valid, packed, 0)
        top = tl.zeros([K_PAD], dtype=tl.uint64)
    for i in tl.static_range(K):
        m = tl.max(packed, axis=0)
        top = tl.where(ok == i, m, top)
        packed = tl.where(packed == m, 0, packed)
    if NARROW:
        keys = top >> 16
        idx = (0xFFFF - (top & 0xFFFF)).to(tl.int32)
    else:
        keys = (top >> 32).to(tl.uint32)
        idx = (0xFFFFFFFF - (top & 0xFFFFFFFF).to(tl.uint32)).to(tl.int32)
    return keys, idx


@triton.jit
def _ds_store_out(keys, idx, mk, Yv, Yi, y_off, LARGEST: tl.constexpr):
    vals = _ds_from_key(keys, Yv.dtype.element_ty, LARGEST)
    tl.store(Yv + y_off, vals, mask=mk)
    tl.store(Yi + y_off, idx.to(tl.int64), mask=mk)


@libentry()
@triton.jit
def topk_deepselect_kernel(
    X,
    Yv,
    Yi,
    BUF,
    n_slots,
    N,
    stride_xm,
    seg_elems,
    segs_per_row,
    perm_stride,
    K: tl.constexpr,
    K_PAD: tl.constexpr,
    WIN: tl.constexpr,
    BLOCK: tl.constexpr,
    B2: tl.constexpr,
    TILE: tl.constexpr,
    FLAGS: tl.constexpr,
):
    # Boolean switches are packed into one constexpr to keep launches cheap.
    LARGEST: tl.constexpr = (FLAGS & _F_LARGEST) != 0
    SORTED: tl.constexpr = (FLAGS & _F_SORTED) != 0
    WRITE_OUT: tl.constexpr = (FLAGS & _F_WRITE_OUT) != 0
    NARROW: tl.constexpr = (FLAGS & _F_NARROW) != 0
    WIN_SORT: tl.constexpr = (FLAGS & _F_WIN_SORT) != 0
    HAS_SCAN: tl.constexpr = (FLAGS & _F_HAS_SCAN) != 0
    WIN_ARGMAX: tl.constexpr = (FLAGS & _F_WIN_ARGMAX) != 0
    USE_TLE: tl.constexpr = (FLAGS & _F_TLE) != 0
    KEY_BITS: tl.constexpr = X.dtype.element_ty.primitive_bitwidth
    key_max: tl.constexpr = (1 << KEY_BITS) - 1
    hist_smem = _ds_hist_alloc(256, USE_TLE)
    # Candidate buffer: the window leaves at most K, the scan keeps < K + B2
    # and appends at most BLOCK more before compacting.
    CAP: tl.constexpr = K_PAD + B2 + BLOCK
    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)
    ok = tl.arange(0, K_PAD)
    mk = ok < K
    wo = tl.arange(0, WIN)
    offs = tl.arange(0, BLOCK)
    cbuf = BUF + pid.to(tl.int64) * (2 * CAP)
    # Per-slot partial results (segmented layout) follow all candidate areas,
    # then one threshold per slot.
    pbuf = BUF + num_progs.to(tl.int64) * (2 * CAP)
    tbuf = pbuf + n_slots.to(tl.int64) * K

    for slot in range(pid, n_slots, num_progs):
        row = slot // segs_per_row
        seg = slot - row * segs_per_row
        x_row = X + row.to(tl.int64) * stride_xm
        seg_lo = seg * seg_elems
        seg_hi = tl.minimum(seg_lo + seg_elems, N)
        y_off = row.to(tl.int64) * K + ok
        p_off = slot.to(tl.int64) * K + ok

        # ---- init window: K candidates and the first threshold, from registers
        wcols = seg_lo + wo
        wm = wcols < seg_hi
        kw = _ds_to_key(tl.load(x_row + wcols, mask=wm, other=0), LARGEST)
        cnt = tl.minimum(tl.minimum(seg_hi - seg_lo, WIN), K)
        cur = tl.full((), 0, tl.int32)
        if WIN_SORT:
            # Selection and the final ordering in one go.
            if WIN_ARGMAX:
                tkeys, tidx = _ds_argmax_pairs(kw, wcols, wm, K, K_PAD, NARROW)
            else:
                tkeys, tidx = _ds_topk_pairs(kw, wcols, wm, K_PAD, NARROW)
            if HAS_SCAN:
                tl.store(cbuf + ok, _ds_pack(tkeys, tidx), mask=ok < cnt)
                thr = tl.max(tl.where(ok == cnt - 1, tkeys, 0), axis=0)
            elif WRITE_OUT:
                _ds_store_out(tkeys, tidx, mk, Yv, Yi, y_off, LARGEST)
            else:
                tl.store(pbuf + p_off, _ds_pack(tkeys, tidx), mask=mk)
                # This segment's K-th key bounds the row's K-th key from below.
                seg_thr = tl.max(tl.where(ok == K - 1, tkeys, 0), axis=0)
                tl.store(tbuf + slot, seg_thr.to(tl.int64))
        else:
            thr, n_eq = _ds_radix_select_regs(kw, wm, cnt, hist_smem, KEY_BITS, USE_TLE)
            sel, wslot, _, _ = _ds_select_positions(kw, wm, thr, n_eq, 0)
            tl.store(cbuf + wslot, _ds_pack(kw, wcols), mask=sel)

        if HAS_SCAN or not WIN_SORT:
            if HAS_SCAN:
                # Until K candidates exist every element is still one.
                thr_lo = tl.where(cnt == K, tl.minimum(thr, key_max - 1) + 1, 0).to(
                    tl.uint32
                )
                # ---- scan the rest of the segment in permuted block order
                main_lo = seg_lo + WIN
                num_blocks = tl.cdiv(seg_elems - WIN, BLOCK)
                # Software pipelining: the next block's load is issued before
                # the current block is filtered, so DRAM latency overlaps the
                # compare/append work (DeepSelect prefetches with TMA).
                # A permutation of [0, num_blocks) visits each block of a
                # shorter (last) segment once; blocks past its end are fully
                # masked off.
                # Only the block number and the loaded data are carried across
                # iterations; columns and masks are recomputed (cheaper than
                # the registers they would pin).
                x_n = tl.load(
                    x_row + main_lo + offs, mask=main_lo + offs < seg_hi, other=0
                )
                blk = tl.full((), 0, tl.int32)
                for i in range(num_blocks):
                    x = x_n
                    blk_n = ((i + 1) * perm_stride) % num_blocks
                    base_n = main_lo + blk_n * BLOCK
                    x_n = tl.load(
                        x_row + base_n + offs,
                        mask=(base_n + offs < seg_hi) & (i + 1 < num_blocks),
                        other=0,
                    )
                    cols = main_lo + blk * BLOCK + offs
                    m_in = cols < seg_hi
                    blk = blk_n

                    key = _ds_to_key(x, LARGEST)
                    take = m_in & (key >= thr_lo)
                    take_i = take.to(tl.int32)
                    n_take = tl.sum(take_i, axis=0)
                    if n_take > 0:
                        pos = cnt + tl.cumsum(take_i, axis=0) - 1
                        tl.store(
                            cbuf + cur * CAP + pos,
                            _ds_pack(key, cols),
                            mask=take,
                        )
                        cnt += n_take
                        if cnt >= K + B2:
                            tl.debug_barrier()
                            thr, n_eq = _ds_radix_select_buf(
                                cbuf + cur * CAP,
                                cnt,
                                K,
                                hist_smem,
                                KEY_BITS,
                                TILE,
                                USE_TLE,
                            )
                            _ds_compact(
                                cbuf + cur * CAP,
                                cbuf + (1 - cur) * CAP,
                                cnt,
                                thr,
                                n_eq,
                                TILE,
                            )
                            tl.debug_barrier()
                            cur = 1 - cur
                            cnt = tl.full((), K, tl.int32)
                            thr_lo = tl.minimum(thr, key_max - 1) + 1

            tl.debug_barrier()
            if cnt > K:
                thr, n_eq = _ds_radix_select_buf(
                    cbuf + cur * CAP, cnt, K, hist_smem, KEY_BITS, TILE, USE_TLE
                )
                _ds_compact(
                    cbuf + cur * CAP, cbuf + (1 - cur) * CAP, cnt, thr, n_eq, TILE
                )
                tl.debug_barrier()
                cur = 1 - cur

            p = tl.load(cbuf + cur * CAP + ok, mask=mk, other=0)
            if WRITE_OUT:
                keys, idx = _ds_unpack(p)
                if SORTED:
                    keys, idx = _ds_sort_pairs(keys, idx, mk, NARROW)
                _ds_store_out(keys, idx, mk, Yv, Yi, y_off, LARGEST)
            else:
                # Exactly K survivors per slot (segments hold at least K
                # elements), stored back to back for the merge kernel, plus
                # the slot's K-th key, which bounds the row's from below.
                tl.store(pbuf + p_off, p, mask=mk)
                tl.store(tbuf + slot, thr.to(tl.int64))
            tl.debug_barrier()


@libentry()
@triton.jit
def topk_merge_kernel(
    Yv,
    Yi,
    PBUF,
    OBUF,
    K: tl.constexpr,
    K_PAD: tl.constexpr,
    SEGS: tl.constexpr,
    TILE: tl.constexpr,
    FLAGS: tl.constexpr,
):
    """Merge the SEGS * K per-segment candidates of a row into its top K.

    The per-segment thresholds prune almost nothing on i.i.d. data (every
    segment holds about K candidates at least as large as the largest of its
    K-th keys), so they are not worth a filtering pass: select directly on
    all SEGS * K candidates.
    """
    LARGEST: tl.constexpr = (FLAGS & _F_LARGEST) != 0
    SORTED: tl.constexpr = (FLAGS & _F_SORTED) != 0
    NARROW: tl.constexpr = (FLAGS & _F_NARROW) != 0
    USE_TLE: tl.constexpr = (FLAGS & _F_TLE) != 0
    KEY_BITS: tl.constexpr = Yv.dtype.element_ty.primitive_bitwidth
    CNT: tl.constexpr = SEGS * K
    hist_smem = _ds_hist_alloc(256, USE_TLE)
    row = tl.program_id(0)
    src = PBUF + row.to(tl.int64) * CNT
    dst = OBUF + row.to(tl.int64) * K_PAD

    thr, n_eq = _ds_radix_select_buf(src, CNT, K, hist_smem, KEY_BITS, TILE, USE_TLE)
    _ds_compact(src, dst, CNT, thr, n_eq, TILE)
    tl.debug_barrier()
    ok = tl.arange(0, K_PAD)
    mk = ok < K
    keys, idx = _ds_unpack(tl.load(dst + ok, mask=mk, other=0))
    if SORTED:
        keys, idx = _ds_sort_pairs(keys, idx, mk, NARROW)
    _ds_store_out(keys, idx, mk, Yv, Yi, row.to(tl.int64) * K + ok, LARGEST)


# ---------------------------------------------------------------------------
# Host side
# ---------------------------------------------------------------------------

_DS_SM_COUNT = {}
_DS_SCRATCH = {}
_DS_PLAN_CACHE = {}
_DS_ENV = {}


def _env_int(name, default):
    """Tuning overrides, read once per process (clear _DS_ENV to re-read)."""
    if name not in _DS_ENV:
        v = os.environ.get(name)
        _DS_ENV[name] = int(v) if v else None
    v = _DS_ENV[name]
    return default if v is None else v


def _ds_sm_count(device):
    idx = device.index if device.index is not None else torch.cuda.current_device()
    if idx not in _DS_SM_COUNT:
        _DS_SM_COUNT[idx] = torch.cuda.get_device_properties(idx).multi_processor_count
    return _DS_SM_COUNT[idx]


def _ds_scratch(device, numel):
    key = (device.type, device.index)
    buf = _DS_SCRATCH.get(key)
    if buf is None or buf.numel() < numel:
        buf = torch.empty(numel, dtype=torch.int64, device=device)
        _DS_SCRATCH[key] = buf
    return buf


def _ds_perm_stride(num_blocks):
    """A stride coprime with num_blocks, close to the golden ratio of it."""
    if num_blocks <= 2:
        return 1
    s = max(1, int(num_blocks * 0.6180339887))
    while math.gcd(s, num_blocks) != 1:
        s += 1
    return s


class _Plan(NamedTuple):
    k_pad: int
    narrow: bool
    win: int
    block: int
    b2: int
    tile: int
    num_warps: int
    grid0: int
    segs: int
    seg_elems: int
    perm_stride: int
    win_sort: bool
    has_scan: bool
    partial_base: int
    merge_base: int
    scratch: int
    merge_tile: int
    merge_warps: int
    maxnreg: int
    flags: int
    needs_buf: bool


def _ds_use_tle(device):
    """Shared-memory histograms via TLE when the compiler provides it."""
    return bool(
        HAS_TLE and device.type == "cuda" and _env_int("FLAGGEMS_TOPK_DS_TLE", 1)
    )


def _ds_warps(width):
    return max(1, min(8, triton.next_power_of_2(width) // 512))


def _ds_make_plan(n_rows, n_cols, k, dtype, device):
    k_pad = triton.next_power_of_2(k)
    key_bits = 32 if dtype == torch.float32 else 16
    # A 16-bit key plus a 16-bit index fit one 32-bit sort word.
    narrow = key_bits == 16 and n_cols <= 65536
    n_pad = triton.next_power_of_2(n_cols)
    win_max = max(_env_int("FLAGGEMS_TOPK_DS_WIN", 4096), k_pad)
    block = min(_env_int("FLAGGEMS_TOPK_DS_BLOCK", 4096), n_pad)
    # Compaction trigger (DeepSelect's RECONSTRUCT_THRESHOLD): larger means
    # fewer radix passes over the candidate buffer.
    b2 = _env_int("FLAGGEMS_TOPK_DS_B2", max(4096, k_pad))
    sm = _ds_sm_count(device)

    # Warps follow the widest register tensor: short rows get small CTAs and
    # many of them per SM.  When there are too few rows to fill the GPU that
    # way, give each row more warps instead.
    width = max(min(n_pad, win_max), k_pad)
    num_warps = _ds_warps(width)
    if width >= 1024 and n_rows < sm * max(1, 16 // num_warps):
        num_warps = max(num_warps, min(8, triton.next_power_of_2(width) // 256))
    num_warps = _env_int("FLAGGEMS_TOPK_DS_WARPS", num_warps)
    occupancy = _env_int("FLAGGEMS_TOPK_DS_OCC", max(1, 16 // num_warps))
    grid_target = sm * occupancy
    # Left alone, ptxas spends up to 255 registers and only one CTA fits per
    # SM; cap registers so the planned occupancy is actually reached (a few
    # spills cost far less than half the resident warps).
    maxnreg = _env_int(
        "FLAGGEMS_TOPK_DS_MAXNREG",
        min(255, 65536 // (num_warps * 32 * occupancy) // 8 * 8),
    )

    # Intra-row parallelism only pays off when rows leave most CTAs idle and
    # are long enough for the extra merge launch to be amortised.
    segs = grid_target // max(1, n_rows)
    # 32-bit keys cost twice the radix passes, so splitting pays off sooner.
    seg_min_n = _env_int(
        "FLAGGEMS_TOPK_DS_SEG_MIN_N", 8192 if key_bits == 32 else 16384
    )
    if segs < _DS_MIN_CTAS_PER_ROW or n_cols < seg_min_n:
        segs = 1
    # The merge reads segs * K candidates per row and every segment pays a
    # fixed window/compaction cost, so cap both: at most ~16K candidates per
    # row, and segments of at least 16K elements (4K for short rows).
    min_seg = max(4096, min(16384, n_cols // 8))
    # The merge kernel launches one CTA per row, so with a handful of rows it
    # is itself the bottleneck: halve the candidates it has to merge there.
    merge_cand = _env_int("FLAGGEMS_TOPK_DS_MERGE_CAND", 16384)
    if n_rows * 8 < grid_target:
        merge_cand = max(8192, 16 * k)
    segs = min(
        segs,
        _env_int("FLAGGEMS_TOPK_DS_SEGS", 64),
        max(1, n_cols // block),
        max(1, merge_cand // k),
        max(1, n_cols // min_seg),
    )
    # Every segment must yield exactly K candidates, so none may be shorter
    # than K (or than half a block); shrink the split until the last one is.
    seg_elems = n_cols
    while segs > 1:
        seg_elems = triton.cdiv(triton.cdiv(n_cols, segs), block) * block
        n_seg = triton.cdiv(n_cols, seg_elems)
        if n_cols - (n_seg - 1) * seg_elems >= max(k, block // 2):
            segs = n_seg
            break
        segs //= 2
    if segs <= 1:
        segs = 1
        seg_elems = n_cols

    win = min(triton.next_power_of_2(seg_elems), win_max)
    has_scan = seg_elems > win
    num_blocks = triton.cdiv(seg_elems - win, block) if has_scan else 0
    # How the init window is reduced to K candidates.  With shared-memory
    # histograms (TLE) radix select wins unless K is tiny (K rounds of
    # max-reduction) or the window is no larger than K_PAD (a plain sort).
    # Without TLE, radix histograms are expensive and a bitonic top-k wins
    # whenever the window is small or K is small relative to it.
    use_tle = _ds_use_tle(device)
    if use_tle:
        win_sort_rule = win <= k_pad or k <= _env_int("FLAGGEMS_TOPK_DS_ARGMAX_K", 8)
    else:
        win_sort_rule = win <= 2 * k_pad or k_pad <= 64
    win_sort = bool(_env_int("FLAGGEMS_TOPK_DS_WINSORT", int(win_sort_rule)))
    # For tiny K, K rounds of max-reduction beat any sorting network.
    win_argmax = (
        win_sort and k <= _env_int("FLAGGEMS_TOPK_DS_ARGMAX_K", 8) and win > k_pad
    )
    cap = k_pad + b2 + block
    tile = min(_env_int("FLAGGEMS_TOPK_DS_TILE", 2048), triton.next_power_of_2(cap))

    n_slots = n_rows * segs
    grid0 = min(n_slots, grid_target)
    # Scratch layout: per-CTA candidate areas | per-slot partial results |
    # per-slot thresholds | merge: final K.
    partial_base = grid0 * 2 * cap
    merge_base = partial_base + n_slots * k + n_slots
    scratch = merge_base + (n_rows * k_pad if segs > 1 else 0)
    # One wide tile beats several narrow ones: the radix passes loop over the
    # candidate buffer once per tile, and a single pass keeps it all in flight.
    merge_tile = _env_int(
        "FLAGGEMS_TOPK_DS_MTILE", min(8192, triton.next_power_of_2(segs * k))
    )
    merge_warps = _env_int("FLAGGEMS_TOPK_DS_MWARPS", 8)
    return _Plan(
        k_pad,
        narrow,
        win,
        block,
        b2,
        tile,
        num_warps,
        grid0,
        segs,
        seg_elems,
        _ds_perm_stride(num_blocks),
        win_sort,
        has_scan,
        partial_base,
        merge_base,
        scratch,
        merge_tile,
        merge_warps,
        maxnreg,
        (int(_F_WRITE_OUT) if segs == 1 else 0)
        | (int(_F_NARROW) if narrow else 0)
        | (int(_F_WIN_SORT) if win_sort else 0)
        | (int(_F_HAS_SCAN) if has_scan else 0)
        | (int(_F_WIN_ARGMAX) if win_argmax else 0)
        | (int(_F_TLE) if use_tle else 0),
        has_scan or not win_sort or segs > 1,
    )


def _ds_plan(n_rows, n_cols, k, dtype, device):
    key = (n_rows, n_cols, k, dtype, device)
    plan = _DS_PLAN_CACHE.get(key)
    if plan is None:
        plan = _ds_make_plan(n_rows, n_cols, k, dtype, device)
        _DS_PLAN_CACHE[key] = plan
    return plan


_DS_LAUNCH_CACHE = {}


def _ds_launch(kernel, key, grid, args, **opts):
    """Launch through libentry once, then reuse the compiled kernel directly.

    libentry already caches compiled kernels, but rebuilding its cache key from
    ~20 arguments costs more than the whole kernel for tiny inputs.  `key` must
    capture everything Triton specialises on (shape-derived integers, the row
    stride, pointer alignment, constexprs); misses fall back to libentry.
    """
    compiled = _DS_LAUNCH_CACHE.get(key)
    if compiled is not None:
        compiled[grid](*args)
        return
    ret = kernel[grid](*args, **opts)
    if isinstance(ret, tuple) and len(ret) == 2 and hasattr(ret[0], "run"):
        _DS_LAUNCH_CACHE[key] = ret[0]


def _topk_deepselect(x2d, n_rows, n_cols, stride, out_shape, k, largest, sorted):
    """x2d: any tensor whose rows (last dim, contiguous) start `stride` apart."""
    device = x2d.device
    p = _ds_plan(n_rows, n_cols, k, x2d.dtype, device)
    flags = (
        p.flags
        | (int(_F_LARGEST) if largest else 0)
        | (int(_F_SORTED) if sorted else 0)
    )

    y_vals = torch.empty(out_shape, device=device, dtype=x2d.dtype)
    y_idx = torch.empty(out_shape, device=device, dtype=torch.int64)
    buf = _ds_scratch(device, p.scratch) if p.needs_buf else y_idx
    # Outputs and scratch come from the caching allocator (always aligned);
    # only the input pointer's alignment varies.
    launch_key = (
        n_rows,
        n_cols,
        k,
        x2d.dtype,
        device,
        stride,
        flags,
        x2d.data_ptr() % 16 == 0,
    )

    with torch_device_fn.device(device):
        _ds_launch(
            topk_deepselect_kernel,
            launch_key,
            (p.grid0, 1, 1),
            (
                x2d,
                y_vals,
                y_idx,
                buf,
                n_rows * p.segs,
                n_cols,
                stride,
                p.seg_elems,
                p.segs,
                p.perm_stride,
                k,
                p.k_pad,
                p.win,
                p.block,
                p.b2,
                p.tile,
                flags,
            ),
            num_warps=p.num_warps,
            num_stages=1,
            maxnreg=p.maxnreg,
        )
        if p.segs > 1:
            _ds_launch(
                topk_merge_kernel,
                ("merge",) + launch_key,
                (n_rows, 1, 1),
                (
                    y_vals,
                    y_idx,
                    buf[p.partial_base :],
                    buf[p.merge_base :],
                    k,
                    p.k_pad,
                    p.segs,
                    p.merge_tile,
                    flags,
                ),
                num_warps=p.merge_warps,
                num_stages=1,
            )
    return y_vals, y_idx


def topk(x, k, dim=-1, largest=True, sorted=True):
    logger.debug("GEMS TOPK (DeepSelect)")

    if dim < 0:
        dim = dim + x.ndim
    assert dim == x.ndim - 1, "Currently only support topk in last dimension"
    topk_elem_cnt = x.shape[dim]
    assert 0 <= k <= topk_elem_cnt, "selected index k out of range"

    if k == 0:
        out_shape = list(x.shape[:-1]) + [0]
        return (
            torch.empty(out_shape, device=x.device, dtype=x.dtype),
            torch.empty(out_shape, device=x.device, dtype=torch.int64),
        )

    if x.dtype in _DS_DTYPES and k <= _DS_MAX_K and topk_elem_cnt < 2**31 - 1:
        out_shape = x.shape[:-1] + (k,)
        # The kernel only needs the row stride, so contiguous inputs of any
        # rank (and 2-D inputs with padded rows) are used as they are.
        if x.ndim == 2 and x.stride(1) == 1:
            return _topk_deepselect(
                x, x.shape[0], topk_elem_cnt, x.stride(0), out_shape, k, largest, sorted
            )
        if not x.is_contiguous():
            x = x.contiguous()
        n_rows = x.numel() // topk_elem_cnt
        return _topk_deepselect(
            x, n_rows, topk_elem_cnt, topk_elem_cnt, out_shape, k, largest, sorted
        )

    return _topk_legacy(x, k, dim, largest, sorted)


__all__ = ["topk"]
