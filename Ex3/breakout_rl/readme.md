curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs/ | sh

cargo build --release

./target/release/breakout_rl

set mode in code -> varibale MODE in main$