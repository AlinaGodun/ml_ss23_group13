use macroquad::color::*;
use macroquad::input::*;
use macroquad::shapes::*;
use macroquad::window::*;
use std::collections::HashMap;
use std::collections::LinkedList;
use std::io::Write;
pub mod breakout_game;
use crate::breakout_game::*;
pub mod breakout_types;
use crate::breakout_types::*;
pub mod mc_control;
use crate::mc_control::*;
pub mod experiments_types;
use crate::experiments_types::*;
use std::io;
use std::time::SystemTime;


use csv::{Writer, WriterBuilder};
use std::error::Error;
use std::fs::File;

pub enum Mode {
    ManualControl,
    KI,
}

pub const MODE: Mode = Mode::KI; //ManualControl/KI
pub const FRAME_DELAY: u64 = 10;
fn render_scene(ball: &Ball, paddle: &Paddle, bricks: &LinkedList<Brick>) {
    clear_background(BLACK);

    draw_circle(
        ((2 * ball.position.x + BALL_SIZE) * SCALING_FACTOR / 2) as f32,
        ((2 * ball.position.y + BALL_SIZE) * SCALING_FACTOR / 2) as f32,
        ((BALL_SIZE * SCALING_FACTOR) / 2) as f32,
        YELLOW,
    );

    draw_rectangle(
        (paddle.position.x * SCALING_FACTOR) as f32,
        (paddle.position.y * SCALING_FACTOR) as f32,
        (PADDLE_LEN * SCALING_FACTOR) as f32,
        (1 * SCALING_FACTOR) as f32,
        WHITE,
    );

    let colors = [RED, ORANGE, YELLOW, GREEN, BLUE, PURPLE];

    for brick in bricks.iter() {
        let row_color = colors[brick.position.y as usize % colors.len()];
        draw_rectangle(
            (brick.position.x * SCALING_FACTOR) as f32,
            (brick.position.y * SCALING_FACTOR) as f32,
            (BRICK_LEN * SCALING_FACTOR) as f32,
            (1 * SCALING_FACTOR) as f32,
            row_color,
        );
    }
}

#[allow(dead_code)]
fn get_action() -> Action {
    let key: Option<KeyCode> = get_last_key_pressed();
    let keycode = match key {
        Some(keycode) => keycode,
        _ => return Action::StandStill,
    };
    match keycode {
        KeyCode::Left => Action::MoveLeft,
        KeyCode::Right => Action::MoveRight,
        _ => Action::StandStill,
    }
}

fn write_experiment_data_to_csv(experiments: &LinkedList<Experiment>, file_path: &str) -> Result<(), Box<dyn Error>> {
    let file = File::create(file_path)?;
    let mut csv_writer = WriterBuilder::new().from_writer(file);

    for e in experiments {
        csv_writer.serialize(e)?;
    }

    csv_writer.flush()?;
    Ok(())
}

// fn write_policy_data_to_csv(policy_infos: &LinkedList<PolicyInfo>, file_path: &str) -> Result<(), Box<dyn Error>> {
//     let file = File::create(file_path)?;
//     let mut csv_writer = WriterBuilder::new().from_writer(file);

//     for pi in policy_infos {
//         csv_writer.serialize(pi)?;
//     }

//     csv_writer.flush()?;
//     Ok(())
// }

#[macroquad::main("Breakout")]
async fn main() {
    let bricks_layout: String = String::from("RainbowStaircase");
    let epsilon: f32 = 0.05;
    let gamma: f32 = 0.98;
    let max_num_episodes: usize = 40000;
    let max_iter_per_episode: usize  = 1000;
    let mut policy_infos: LinkedList<Experiment> = LinkedList::new();

    for epsilon in [0.01, 0.03, 0.05, 0.1,  0.3] {
        assert!(GRID_SIZE_Y > BRICK_ROWS + 2); // enough place for balls + bricks
        assert!(GRID_SIZE_X % BRICK_LEN == 0); // complete row of bricks
        assert!(GRID_SIZE_X % 2 != 0); // center pixel available
        assert!(SCALING_FACTOR % 2 == 0); // guarantee positions are int

        request_new_screen_size(
            (GRID_SIZE_X * SCALING_FACTOR) as f32,
            (GRID_SIZE_Y * SCALING_FACTOR) as f32,
        );

        let now = SystemTime::now();
        let (policy, experiments) = match MODE {
            Mode::KI => {
                let (policy, experiments) = mc_control_loop(max_num_episodes, max_iter_per_episode, epsilon, gamma, &bricks_layout);
                match now.elapsed() {
                    Ok(elapsed) => {
                        println!("Policy found in {0}s!", elapsed.as_secs());
                        let mut policy_experiment_final = experiments.back().expect("Experiments cannot be empty.").clone();
                        policy_experiment_final.policy_found_in_s = elapsed.as_secs();
                        policy_infos.push_back(policy_experiment_final);
                    }
                    Err(e) => {
                        println!("Error: {e:?}");
                    }
                }
                // print!("Press enter to start visualization: ");
                // io::stdout().flush().unwrap();
                // let mut irrelevant_input = String::new();
                // io::stdin()
                //     .read_line(&mut irrelevant_input)
                //     .expect("failed to readline");
                    (policy, experiments)
            }
            Mode::ManualControl => (HashMap::new(), LinkedList::new()),
        };

        // let mut experiments: LinkedList<Experiment> = LinkedList::new();

        // for ball_start_x in -2..=2 {
            
        //     let (mut ball, mut paddle, mut bricks) =
        //         reset_game(GRID_SIZE_X, GRID_SIZE_Y, PADDLE_LEN, BRICK_LEN, BRICK_ROWS);
        //     let mut number_of_steps = 0;

        //     ball.velocity.x = ball_start_x;

        //     loop {
        //         let action = match MODE {
        //             Mode::ManualControl => get_action(),
        //             Mode::KI => {
        //                 let state = State::new(&ball, &paddle, &bricks);
        //                 take_action(&policy, &state, 0.0)
        //             }
        //         };

        //         let game_status = game_step(&mut paddle, &mut ball, &mut bricks, &action);

        //         if let GameStatus::ResetGame = game_status {
        //             (ball, paddle, bricks) =
        //                 reset_game(GRID_SIZE_X, GRID_SIZE_Y, PADDLE_LEN, BRICK_LEN, BRICK_ROWS);
        //         }

        //         render_scene(&ball, &paddle, &bricks);
        //         std::thread::sleep(std::time::Duration::from_millis(FRAME_DELAY));
        //         next_frame().await;
        //         number_of_steps += 1;

        //         if let GameStatus::GameWon = game_status {
        //             println!("You win");
        //             break;
        //         }
        //     }
        //     println!("Took {number_of_steps} steps!");

        //     // let bricks_layout: String = String::from("SlavicGrandmaTextil");
        //     // let reward: String = String::from("normal");
        //     // let current_experiment = Experiment::new(&bricks_layout, &ball_start_x, &number_of_steps, &gamma, &epsilon, &max_num_episodes, &100, &reward);

        //     // experiments.push_back(current_experiment);
        // }
        

        let file_name = format!("./data/{bricks_layout}_eps={epsilon}.csv");
        if let Err(e) = write_experiment_data_to_csv(&experiments, &file_name) {
            println!("{e}"); // "There is an error: Oops"
        }
    }

    if let Err(e) = write_experiment_data_to_csv(&policy_infos, &format!("./data/{bricks_layout}_eps_experiments.csv")) {
        println!("{e}"); // "There is an error: Oops"
    }
}
