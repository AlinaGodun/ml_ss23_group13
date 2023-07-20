# 1. Install Rust compiler
On Linux/MacOS:

```curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs/ | sh```

After running this command you might need to restart the terminal, so that the updated $PATH variable is used (set during installation).

If you encounter any issues or use another operating systems refer to
https://www.rust-lang.org/tools/install

# 2. Build binary
```cd``` to the project, then:

```cargo build --release```
# 3. Execute binary
The program will first train the RL model. After the training is done and the user presses enter into the terminal, the game will be rendered in a window. The movements of the paddle are done by the RL model by using the policy that was found in training. To terminate the program press Ctrl+C in the terminal.

```./target/release/breakout_rl```

If the program crashes right at startup and you are using MacOS try commenting out the following function call and rebuild the binary. The lines are found in main.rs in line ~78.
```
request_new_screen_size(
        (GRID_SIZE_X * SCALING_FACTOR) as f32,
        (GRID_SIZE_Y * SCALING_FACTOR) as f32,
);
```

# Optional: Manual Play
You can switch the mode variable at the top of main.rs to Mode::ManualControl to control the game manually with the left and right arrow key.
After changing the variable you will need to rebuild the binary and execute it again.