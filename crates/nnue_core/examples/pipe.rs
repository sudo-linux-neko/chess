use std::time::Instant;
fn main() {
    let path = std::env::args().nth(1).unwrap();
    let n: usize = std::env::args().nth(2).map(|s| s.parse().unwrap()).unwrap_or(200_000);
    let iters: usize = std::env::args().nth(3).map(|s| s.parse().unwrap()).unwrap_or(20);
    let mut p = nnue_core::dataset::BinpackPipeline::open(&path, 0, 0);
    let mut total = 0usize;
    let t = Instant::now();
    for _ in 0..iters {
        let b = p.next_batch(n).unwrap();
        total += b.len();
        if b.is_empty() { break; }
    }
    let dt = t.elapsed().as_secs_f64();
    eprintln!("{total} pos in {dt:.3}s = {:.2} M/s", total as f64 / dt / 1e6);
}
