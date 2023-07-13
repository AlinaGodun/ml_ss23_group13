use std::collections::LinkedList;

use macroquad::prelude::*;


struct Velocity{
    x: f32,
    y: f32,
}

#[derive(Debug)]
struct Position{
    x: f32,
    y: f32
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

fn initialize_ball(grid_size_x:f32, grid_size_y:f32) -> Ball{
    let initial_position = Position{x:grid_size_x/2.0, y:grid_size_y - 1.0};

    let rand_vel_x = rand::gen_range(-3, 3);
    println!("{rand_vel_x}");

    let initial_velocity = Velocity{x:rand_vel_x as f32, y:-1.0};
    Ball {
        position:initial_position, 
        velocity: initial_velocity
    }
}

fn initialize_paddle(grid_size_x:f32, grid_size_y:f32, paddle_size:f32) -> Paddle{
    let initial_position = Position{x:grid_size_x/2.0 - paddle_size/2.0, y:grid_size_y-1.0};

    let initial_velocity = Velocity{x:0.0, y:0.0};
    Paddle {
        position:initial_position, 
        velocity: initial_velocity}
}

fn initialize_bricks(grid_size_x:f32, brick_size:f32, brick_rows:f32) -> LinkedList<Brick>{
    let mut pos_y = 0.0;
    let colors = [RED, ORANGE, YELLOW, GREEN, BLUE, PURPLE];
    let mut bricks: LinkedList<Brick> = LinkedList::new();
    loop {    
        let mut pos_x = 0.0;
        if pos_y >= brick_rows {break;}
        let row_color = colors[pos_y as usize % colors.len()];
        loop{
            if pos_x >= grid_size_x {break;}
            let position =  Position{x: pos_x, y:pos_y};
            let brick = Brick{position, color: row_color};
            bricks.push_back(brick);

            pos_x += brick_size;
        }
        pos_y += 1.0;
    }
    return bricks;
}

// -----> x
// |
// v
// Y
fn reset_game(grid_size_x:f32, grid_size_y:f32, paddle_size:f32, brick_size:f32, brick_rows:f32) -> (Ball, Paddle, LinkedList<Brick>) {
    let ball = initialize_ball(grid_size_x, grid_size_y);
    let paddle = initialize_paddle(grid_size_x, grid_size_y, paddle_size);
    let bricks = initialize_bricks(grid_size_x, brick_size, brick_rows);
    (ball, paddle, bricks)
}


#[macroquad::main("Breakout")]
async fn main() {
    const GRID_SIZE_X:f32 = 11.0;
    const GRID_SIZE_Y:f32 = 11.0;
    const BRICK_ROWS: f32 = 6.0;
    const BALL_SIZE:f32 = 1.0;
    const PADDLE_LEN:f32 = 5.0;
    const BRICK_LEN:f32 = 3.0;
    const SCALING_FACTOR:f32 = 12.0;

    assert!(GRID_SIZE_Y > BRICK_ROWS+2.0);

    request_new_screen_size(
        (GRID_SIZE_X*SCALING_FACTOR) as f32, 
        (GRID_SIZE_Y*SCALING_FACTOR) as f32);

    let (ball, paddle, bricks) = reset_game(GRID_SIZE_X, GRID_SIZE_Y, PADDLE_LEN, BRICK_LEN, BRICK_ROWS);
    rand::srand(macroquad::miniquad::date::now() as _);

    loop {
            


        clear_background(BLACK);

        draw_circle(
            ball.position.x*SCALING_FACTOR, 
            (ball.position.y-BALL_SIZE/2.0)*SCALING_FACTOR,
            (BALL_SIZE * SCALING_FACTOR)/2.0, 
            YELLOW);

        draw_rectangle(
            paddle.position.x*SCALING_FACTOR,
            (paddle.position.y)*SCALING_FACTOR,
            PADDLE_LEN*SCALING_FACTOR,
            1.0*SCALING_FACTOR,
            WHITE);
        for brick in bricks.iter(){
            draw_rectangle(
                brick.position.x*SCALING_FACTOR,
                (brick.position.y)*SCALING_FACTOR,
                BRICK_LEN*SCALING_FACTOR,
                1.0*SCALING_FACTOR,
                brick.color);
        }

        next_frame().await
    }
}