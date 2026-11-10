// Prevents additional console window on Windows in release, DO NOT REMOVE!!
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|a| a == "--mcp-stdio") {
        optixel_studio_lib::run_mcp_stdio();
    } else if let Some(i) = args.iter().position(|a| a == "--import") {
        let Some(folder) = args.get(i + 1) else {
            eprintln!("usage: optixel-studio --import <folder>");
            std::process::exit(2);
        };
        match optixel_studio_lib::run_import(folder) {
            Ok(n) => eprintln!("\nimported {n} new photo(s)"),
            Err(e) => {
                eprintln!("\nimport failed: {e}");
                std::process::exit(1);
            }
        }
    } else {
        optixel_studio_lib::run();
    }
}
