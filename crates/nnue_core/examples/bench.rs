use std::fs::File;
use std::io::{Seek, SeekFrom};
use std::time::Instant;

fn main() {
    let path = std::env::args().nth(1).unwrap();
    let mut f = File::open(&path).unwrap();
    let mut buf: Vec<u8> = Vec::new();
    let t = Instant::now();
    let (mut chunks, mut bytes) = (0u64, 0u64);
    while sfbinpack::read_chunk_into(&mut f, &mut buf).unwrap() {
        chunks += 1;
        bytes += buf.len() as u64;
    }
    eprintln!("RAW  chunks={chunks} bytes={bytes} {:?}", t.elapsed());

    f.seek(SeekFrom::Start(0)).unwrap();
    let t = Instant::now();
    let mut ents = 0u64;
    while sfbinpack::read_chunk_into(&mut f, &mut buf).unwrap() {
        ents += sfbinpack::parse_chunk(&buf).len() as u64;
    }
    eprintln!("PARSE entries={ents} {:?} = {:.2} M/s", t.elapsed(), ents as f64 / t.elapsed().as_secs_f64() / 1e6);

    f.seek(SeekFrom::Start(0)).unwrap();
    let t = Instant::now();
    let mut ents = 0u64;
    while sfbinpack::read_chunk_into(&mut f, &mut buf).unwrap() {
        let entries = sfbinpack::parse_chunk(&buf);
        ents += entries.len() as u64;
        for e in &entries {
            let (_f, _a, _b) = nnue_core::features::active_features(&e.pos);
            let _c = nnue_core::features::layer_stack_bucket(e.pos.occupied().count());
        }
    }
    eprintln!("FULL  entries={ents} {:?} = {:.2} M/s", t.elapsed(), ents as f64 / t.elapsed().as_secs_f64() / 1e6);
}
