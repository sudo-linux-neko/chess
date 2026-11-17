// Copyright (C) 2026 Vaibhav
// SPDX-License-Identifier: GPL-3.0-or-later
//
// HalfKAv2_hm feature set, byte-for-byte compatible with Stockfish 16.1
// (src/nnue/features/half_ka_v2_hm.{h,cpp}). This is the *real* layout:
//   64 king buckets x 11 piece-square planes x 32 oriented squares = 22528.

use sfbinpack::chess::color::Color;
use sfbinpack::chess::piecetype::PieceType;
use sfbinpack::chess::position::Position;

pub const NUM_BUCKETS: usize = 64;
pub const PS_NB: usize = 11 * 64; // 704
pub const INPUT_DIM: usize = NUM_BUCKETS * PS_NB / 2; // 22528
pub const MAX_ACTIVE: usize = 32;
/// Padding index used by the trainer. It is >= INPUT_DIM and maps to a
/// permanently-zero extra embedding row, so padded slots contribute nothing.
pub const PAD_INDEX: i16 = INPUT_DIM as i16;

const PS_KING: u16 = 10 * 64;

#[inline(always)]
fn piece_square_offset(persp: u8, pt: PieceType, color: Color) -> u16 {
    if pt == PieceType::King {
        return PS_KING;
    }
    let base = match pt {
        PieceType::Pawn => 0,
        PieceType::Knight => 2 * 64,
        PieceType::Bishop => 4 * 64,
        PieceType::Rook => 6 * 64,
        PieceType::Queen => 8 * 64,
        PieceType::King | PieceType::None => return PS_KING,
    };
    // Own pieces in the first slot, enemy pieces in the second slot of the pair.
    if color.ordinal() == persp {
        base
    } else {
        base + 64
    }
}

#[inline(always)]
fn king_bucket(ksq: u32, persp: u8) -> u16 {
    let rank = ksq >> 3;
    let file = ksq & 7;
    // File mirror: a b c d e f g h -> 0 1 2 3 3 2 1 0.
    let off = [0u32, 1, 2, 3, 3, 2, 1, 0][file as usize];
    let base = if persp == 0 { (7 - rank) * 4 } else { rank * 4 };
    ((base + off) as u16) * (PS_NB as u16)
}

#[inline(always)]
fn orient(ksq: u32, persp: u8) -> u32 {
    let file = ksq & 7;
    match persp {
        0 => {
            if file < 4 {
                7
            } else {
                0
            }
        }
        _ => {
            if file < 4 {
                63
            } else {
                56
            }
        }
    }
}

/// Fill `out[persp * MAX_ACTIVE ..]` with active feature indices for one
/// perspective and return the number of active features (<= 32).
#[inline(always)]
pub fn active_for_perspective(pos: &Position, persp: u8, out: &mut [u16; MAX_ACTIVE]) -> usize {
    let color = if persp == 0 { Color::White } else { Color::Black };
    let ksq = pos.king_sq(color).index();
    let bucket = king_bucket(ksq, persp);
    let ori = orient(ksq, persp);

    let mut n = 0usize;
    for sq in pos.occupied().iter() {
        let pc = pos.piece_at(sq);
        let (pt, c) = pc.parts();
        let idx = (sq.index() ^ ori) + piece_square_offset(persp, pt, c) as u32 + bucket as u32;
        debug_assert!((idx as usize) < INPUT_DIM);
        out[n] = idx as u16;
        n += 1;
    }
    n
}

/// Compute both perspectives. Output layout: `[white(32) | black(32)]`.
///
/// Unused slots are filled with `PAD_INDEX`, so the returned array is always
/// safe to feed straight into an embedding lookup with an extra zero row.
pub fn active_features(pos: &Position) -> ([u16; 2 * MAX_ACTIVE], u8, u8) {
    let mut buf = [PAD_INDEX as u16; 2 * MAX_ACTIVE];
    let mut w = [0u16; MAX_ACTIVE];
    let mut b = [0u16; MAX_ACTIVE];
    let nw = active_for_perspective(pos, 0, &mut w);
    let nb = active_for_perspective(pos, 1, &mut b);
    buf[..nw].copy_from_slice(&w[..nw]);
    buf[MAX_ACTIVE..MAX_ACTIVE + nb].copy_from_slice(&b[..nb]);
    (buf, nw as u8, nb as u8)
}

/// Stockfish 16.1 layer-stack / PSQT bucket: (piece_count - 1) / 4, clamped.
#[inline(always)]
pub fn layer_stack_bucket(piece_count: u32) -> u8 {
    (((piece_count.saturating_sub(1)) / 4) as u8).min(7)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dims_match_stockfish() {
        assert_eq!(INPUT_DIM, 22528);
        assert_eq!(MAX_ACTIVE, 32);
    }

    #[test]
    fn startpos_features_bounds() {
        let pos = Position::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1").unwrap();
        let (feat, nw, nb) = active_features(&pos);
        assert_eq!(nw, 32);
        assert_eq!(nb, 32);
        for &f in feat.iter() {
            assert!((f as usize) < INPUT_DIM);
        }
    }

    #[test]
    fn buckets_and_king_squares() {
        // White king on e1 (index 4), white perspective: file e -> no mirror,
        // rank 1 -> base 28, e-file off 3 -> bucket 31.
        assert_eq!(king_bucket(4, 0) / PS_NB as u16, 31);
        // Black king on e8 (index 60), black perspective: bucket 31.
        assert_eq!(king_bucket(60, 1) / PS_NB as u16, 31);
    }
}
