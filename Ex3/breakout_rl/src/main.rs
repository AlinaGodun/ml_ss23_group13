use std::collections::HashMap;
use std::collections::LinkedList;
use std::io::Write;
use macroquad::color::*;
use macroquad::window::*;
use macroquad::shapes::*;
use macroquad::input::*;
pub mod breakout_game;
use crate::breakout_game::*;
pub mod breakout_types;
use crate::breakout_types::*;
pub mod mc_control;
use crate::mc_control::*;
use std::io;
use std::time::SystemTime;


pub enum Mode{
    ManualControl,
    KI,
}

pub const MODE:Mode = Mode::ManualControl; //ManualControl/KI
pub const FRAME_DELAY:u64 = 200;
fn render_scene(ball: &Ball, paddle: &Paddle, bricks: &LinkedList<Brick>){
    clear_background(BLACK);

    draw_circle(
        ((2*ball.position.x+BALL_SIZE)*SCALING_FACTOR/2) as f32, 
        ((2*ball.position.y+BALL_SIZE)*SCALING_FACTOR/2) as f32,
        ((BALL_SIZE * SCALING_FACTOR)/2) as f32, 
        YELLOW);

    draw_rectangle(
        (paddle.position.x*SCALING_FACTOR) as f32,
        (paddle.position.y*SCALING_FACTOR) as f32,
        (PADDLE_LEN*SCALING_FACTOR) as f32,
        (1*SCALING_FACTOR) as f32,
        WHITE);
    

    let colors = [RED, ORANGE, YELLOW, GREEN, BLUE, PURPLE];

    for brick in bricks.iter(){
        let row_color = colors[brick.position.y as usize % colors.len()];
        draw_rectangle(
            (brick.position.x*SCALING_FACTOR) as f32,
            (brick.position.y*SCALING_FACTOR) as f32,
            (BRICK_LEN*SCALING_FACTOR) as f32,
            (1*SCALING_FACTOR) as f32,
            row_color);
    }

}

#[allow(dead_code)]
fn get_action() -> Action{
    let key: Option<KeyCode> = get_last_key_pressed();
    let keycode = match key{
        Some(keycode) => keycode,
        _ => return Action::StandStill
    };
    match keycode {
        KeyCode::Left => Action::MoveLeft,
        KeyCode::Right => Action::MoveRight,
        _ => Action::StandStill
    }
}

#[macroquad::main("Breakout")]
async fn main() {
    assert!(GRID_SIZE_Y > BRICK_ROWS + 2);  // enough place for balls + bricks
    assert!(GRID_SIZE_X % BRICK_LEN  == 0); // complete row of bricks
    assert!(GRID_SIZE_X % 2 != 0);          // center pixel available
    assert!(SCALING_FACTOR % 2 == 0);       // guarantee positions are int 

    request_new_screen_size(
        (GRID_SIZE_X*SCALING_FACTOR) as f32, 
        (GRID_SIZE_Y*SCALING_FACTOR) as f32);


    let now = SystemTime::now();
    let policy = match MODE{
        Mode::KI => {
            let policy = mc_control_loop(20000, 1000, 0.05);
            match now.elapsed() {
                Ok(elapsed) => {
                    println!("Policy found in {0}s!", elapsed.as_secs());
                }
                Err(e) => {
                    println!("Error: {e:?}");
                }
            }
            print!("Press enter to start visualization: ");
            io::stdout().flush().unwrap();
            let mut irrelevant_input = String::new(); 
            io::stdin().read_line(&mut irrelevant_input).expect("failed to readline");
            policy
        },
        Mode::ManualControl => HashMap::new()
    };

    let (mut ball, mut paddle, mut bricks) = reset_game(GRID_SIZE_X, GRID_SIZE_Y, PADDLE_LEN, BRICK_LEN, BRICK_ROWS);
    let mut number_of_steps = 0;

    loop {
        let action = match MODE{ 
            Mode::ManualControl => get_action(), 
            Mode::KI => {
                let state = State::new(&ball, &paddle, &bricks);
                take_action(&policy, &state, 0.0)
            }
        };

        let game_status = game_step(&mut paddle, &mut ball, &mut bricks, &action);


        if let GameStatus::ResetGame = game_status{
            (ball, paddle, bricks) = reset_game(GRID_SIZE_X, GRID_SIZE_Y, PADDLE_LEN, BRICK_LEN, BRICK_ROWS);
        }

        render_scene(&ball, &paddle, &bricks);
        std::thread::sleep(std::time::Duration::from_millis(FRAME_DELAY));
        next_frame().await;
        number_of_steps += 1;

        if let GameStatus::GameWon = game_status{
            println!("You win");
            break;
        }
    }
    println!("Took {number_of_steps} steps!");
}

