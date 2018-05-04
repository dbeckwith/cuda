extern crate cuda;

use cuda::driver::{self, Device, Result};

fn main() {
    run().unwrap();
}

fn run() -> Result<()> {
    driver::initialize()?;

    let n = Device::count()?;
    println!("Total devices: {}", n);

    for i in 0..n {
        let device = Device(i as u16)?;

        println!("");
        println!("# Device {}", i);
        println!("NAME: {}", device.name()?.to_string_lossy().into_owned());
        println!("MAX_THREADS_PER_BLOCK: {}", device.max_threads_per_block()?);
        println!("TOTAL_MEMORY: {}", device.total_memory()?);
    }

    Ok(())
}
