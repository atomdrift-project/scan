//! Standalone mock hopper for benchmarking the litmus worker model over a
//! local dataset directory (e.g. `~/data/benchmark/realworld-small`).
//!
//! Run it, point a worker at the printed port, and the worker exercises its
//! real claim → prefetch → analyze → result-post path with no external hopper:
//!
//! ```text
//! scan-bench-hopper --dataset ~/data/benchmark/realworld-small --port 8090
//! ascan worker --url http://127.0.0.1:8090 --data-dir ~/data/benchmark/realworld-small \
//!     --workers 12 --max-jobs <file-count>
//! ```
//!
//! Lives in its own process so its (tiny) memory never lands in the worker's
//! RSS measurement.

use std::path::PathBuf;
use std::process::ExitCode;

use clap::Parser;
use scan::bench_hopper::{BenchHopper, Order};

#[derive(Parser)]
#[command(about = "Mock hopper serving a local dataset for worker benchmarks")]
struct Args {
    /// Dataset directory to serve (one job per file, recursive).
    #[arg(long)]
    dataset: PathBuf,

    /// Port to bind (0 = pick an ephemeral port and print it).
    #[arg(long, default_value = "8090")]
    port: u16,

    /// Append every posted result body (one JSON object per line) to this file,
    /// for parity diffing between worker runs.
    #[arg(long)]
    dump: Option<PathBuf>,

    /// Job handout order: fifo, shuffle[:seed], big-first, or small-first.
    /// big-first reproduces the small-job-starvation worst case.
    #[arg(long, default_value = "fifo")]
    order: Order,

    /// Write the latency/throughput summary JSON here (rewritten after every
    /// result, so it is current whenever the benchmark harness reads it).
    #[arg(long)]
    summary: Option<PathBuf>,
}

fn main() -> ExitCode {
    let args = Args::parse();
    if !args.dataset.is_dir() {
        eprintln!(
            "error: dataset is not a directory: {}",
            args.dataset.display()
        );
        return ExitCode::FAILURE;
    }

    let hopper = match BenchHopper::build(&args.dataset, args.dump, args.order, args.summary) {
        Ok(h) => h,
        Err(e) => {
            eprintln!(
                "error: failed to build hopper from {}: {e}",
                args.dataset.display()
            );
            return ExitCode::FAILURE;
        }
    };

    let handle = match hopper.serve(args.port) {
        Ok(h) => h,
        Err(e) => {
            eprintln!("error: failed to bind port {}: {e}", args.port);
            return ExitCode::FAILURE;
        }
    };

    println!(
        "bench-hopper: serving {} jobs on http://127.0.0.1:{} (Ctrl-C to stop)",
        handle.jobs_total, handle.port
    );
    println!("PORT={}", handle.port);
    println!("JOBS={}", handle.jobs_total);

    // Park forever; the accept thread does the work. A short poll keeps the
    // result tally visible for interactive runs without busy-spinning.
    loop {
        std::thread::sleep(std::time::Duration::from_secs(5));
        eprintln!(
            "bench-hopper: {}/{} results received",
            handle.results_received(),
            handle.jobs_total,
        );
    }
}
