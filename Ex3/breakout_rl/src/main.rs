use std::collections::LinkedList;

use macroquad::color::*;
use macroquad::window::*;
use macroquad::shapes::*;
use rand::Rng;


struct Velocity{
    x: i32,
    y: i32,
}

#[derive(Debug)]
struct Position{
    x: i32,
    y: i32
}

struct Ball{
    position: Position,
    velocity: Velocity
}

struct Paddle{
    position: Position,
    velocity: Velocity
}

#[derive(Debug)]
struct Brick{
    position: Position,
    color: Color
}

fn initialize_ball(grid_size_x:i32, grid_size_y:i32) -> Ball{
    let initial_position = Position{x:grid_size_x/2, y:grid_size_y-1};

    let mut rng = rand::thread_rng();
    let rand_vel_x = rng.gen_range(-2..=2);
    println!("{rand_vel_x}");

    let initial_velocity = Velocity{x:rand_vel_x, y:-1};
    Ball {
        position:initial_position, 
        velocity: initial_velocity
    }
}

fn initialize_paddle(grid_size_x:i32, grid_size_y:i32, paddle_size:i32) -> Paddle{
    let initial_position = Position{x:grid_size_x/2 - paddle_size/2, y:grid_size_y-1};

    let initial_velocity = Velocity{x:0, y:0};
    Paddle {
        position:initial_position, 
        velocity: initial_velocity}
}

fn initialize_bricks(grid_size_x:i32, brick_size:i32, brick_rows: i32) -> LinkedList<Brick>{
    let mut pos_y = 0;
    let colors = [RED, ORANGE, YELLOW, GREEN, BLUE, PURPLE];
    let mut bricks: LinkedList<Brick> = LinkedList::new();
    loop {    
        let mut pos_x = 0;
        if pos_y >= brick_rows {break;}
        let row_color = colors[pos_y as usize % colors.len()];
        loop{
            if pos_x >= grid_size_x {break;}
            let position =  Position{x: pos_x, y:pos_y};
            let brick = Brick{position, color: row_color};
            bricks.push_back(brick);

            pos_x += brick_size;
        }
        pos_y += 1;
    }
    return bricks;
}

// -----> x
// |
// v
// Y
fn reset_game(grid_size_x:i32, grid_size_y:i32, paddle_size:i32, brick_size:i32, brick_rows:i32) -> (Ball, Paddle, LinkedList<Brick>) {
    let ball = initialize_ball(grid_size_x, grid_size_y);
    let paddle = initialize_paddle(grid_size_x, grid_size_y, paddle_size);
    let bricks = initialize_bricks(grid_size_x, brick_size, brick_rows);
    (ball, paddle, bricks)
}


#[macroquad::main("Breakout")]
async fn main() {
    const GRID_SIZE_X:i32 = 21; 
    const GRID_SIZE_Y:i32 = 21;
    const BRICK_ROWS: i32 = 1;
    const BALL_SIZE:i32 = 1;
    const PADDLE_LEN:i32 = 5;
    const BRICK_LEN:i32 = 3;
    const SCALING_FACTOR:i32 = 20;


    //GRID_SIZE_X must be multiple of BRICK_LEN and odd
    assert!(GRID_SIZE_Y > BRICK_ROWS + 2);  // enough place for balls + bricks
    assert!(GRID_SIZE_X % BRICK_LEN  == 0); // complete row of bricks
    assert!(GRID_SIZE_X % 2 != 0);          // center pixel available
    assert!(SCALING_FACTOR % 2 == 0);       // guarantee positions are int 

    request_new_screen_size(
        (GRID_SIZE_X*SCALING_FACTOR) as f32, 
        (GRID_SIZE_Y*SCALING_FACTOR) as f32);

    let (mut ball, mut paddle, mut bricks) = reset_game(GRID_SIZE_X, GRID_SIZE_Y, PADDLE_LEN, BRICK_LEN, BRICK_ROWS);
    loop {
        ball.position.x += ball.velocity.x;
        ball.position.y += ball.velocity.y;

        // x < 0 || x >= x_gridsize-> x vel *(-1)
        // y < 0 -> y vel *(-1)
        // y >= y grod -> reset

        if ball.position.x <= 0 || ball.position.x >= GRID_SIZE_X-1 {
            ball.velocity.x *= -1
        };
        if ball.position.y <= 0 {
            ball.velocity.y *= -1
        };
        if ball.position.y > GRID_SIZE_Y {
            (ball, paddle, bricks) = reset_game(GRID_SIZE_X, GRID_SIZE_Y, PADDLE_LEN, BRICK_LEN, BRICK_ROWS);
        };

        ball.position.x = ball.position.x.clamp(0, GRID_SIZE_X - 1);
        clear_background(BLACK);

        draw_circle(
            ((2*ball.position.x+BALL_SIZE)*SCALING_FACTOR/2) as f32, 
            ((2*ball.position.y-BALL_SIZE)*SCALING_FACTOR/2) as f32,
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

        std::thread::sleep(std::time::Duration::from_millis(100 as u64));
        next_frame().await
    }
}