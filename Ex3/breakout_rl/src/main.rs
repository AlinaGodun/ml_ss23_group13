use std::collections::LinkedList;
use macroquad::color::*;
use macroquad::window::*;
use macroquad::shapes::*;
use macroquad::input::*;
pub mod breakout_game;
use crate::breakout_game::*;
pub mod breakout_types;
use crate::breakout_types::*;

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
        
    for brick in bricks.iter(){
        draw_rectangle(
            (brick.position.x*SCALING_FACTOR) as f32,
            (brick.position.y*SCALING_FACTOR) as f32,
            (BRICK_LEN*SCALING_FACTOR) as f32,
            (1*SCALING_FACTOR) as f32,
            brick.color);
    }

}

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

    let (mut ball, mut paddle, mut bricks) = reset_game(GRID_SIZE_X, GRID_SIZE_Y, PADDLE_LEN, BRICK_LEN, BRICK_ROWS);
    loop {

        let action = get_action();

        let game_status = game_step(&mut paddle, &mut ball, &mut bricks, &action);

        if let GameStatus::GameWon = game_status{
            println!("You win");
            break;
        }

        if let GameStatus::ResetGame = game_status{
            (ball, paddle, bricks) = reset_game(GRID_SIZE_X, GRID_SIZE_Y, PADDLE_LEN, BRICK_LEN, BRICK_ROWS);
        }

        render_scene(&ball, &paddle, &bricks);
        std::thread::sleep(std::time::Duration::from_millis(100 as u64));
        next_frame().await;
       
    }
}


