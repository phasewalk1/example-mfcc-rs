extern crate rustfft;
extern crate ndarray;
extern crate hound;
extern crate serde_json;

use mfcc::mfcc::Transform as MfccTransform;

static SEGMENT_SIZE: u16 = 1024;
static NORM_LEN: usize = 10;
static N_FILTERS: (usize, usize) = (20, 40);

// load a .wav file into sample rate and signals data
fn load_file(filename: &str) -> (u32, Vec<f32>) {
    let wavreader = hound::WavReader::open(filename).unwrap();
    let sample_rate = wavreader.spec().sample_rate;
    let data: Vec<f32> = wavreader.into_samples::<i16>()
        .map(|x| x.unwrap() as f32 / std::i16::MAX as f32)
        .collect();
    return (sample_rate as u32, data);
}

// segment the .wav data into 1024 sample segments
fn segment_data(data: Vec<f32>, seg_len: u16) -> Vec<Vec<i16>> {
    let mut segments = Vec::new();
    let mut segment = Vec::new();
    for i in 0..data.len() {
        segment.push(data[i] as i16);
        if segment.len() == seg_len as usize {
            segments.push(segment);
            segment = Vec::new();
        }
    }
    return segments;
}

// transform a segment into mfcc features
fn transform_segment(segment: &[i16], sr: u32, seg_len: u16) -> Vec<f64> {
    let mut state = MfccTransform::new(sr as usize, seg_len as usize)
        .normlength(NORM_LEN)
        .nfilters(N_FILTERS.0, N_FILTERS.1);

    let mut output = vec![0.0 as f64; 20*3];
    state.transform(&segment, &mut output);

    return output;
}

// dump the mfcc features to a json file
fn dump_batch(batch: Vec<Vec<f64>>) {
    let json_data = serde_json::json!({"mfcc_batches": batch});
    let file = std::fs::File::create("mfcc_batches.json").unwrap();
    let write = std::io::BufWriter::new(file);
    serde_json::to_writer_pretty(write, &json_data).unwrap();
}

fn main() {
    let args = std::env::args().collect::<Vec<_>>();
    let filname = &args[1];
    let (sample_rate, data) = load_file(filname);
    let segments = segment_data(data, SEGMENT_SIZE);
    let mut transformations = vec![vec![]];
    for s in segments {
        let transformed = transform_segment(&s, sample_rate, 1024);
        transformations.push(transformed);
    }
    println!("Transformed {} segments", transformations.len());
    dump_batch(transformations);
}
