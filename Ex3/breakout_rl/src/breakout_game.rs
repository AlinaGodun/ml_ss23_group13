use std::collections::LinkedList;
use macroquad::color::*;
use rand::Rng;
use crate::breakout_types::*;

impl Paddle{
    fn new(grid_size_x:i32, grid_size_y:i32, paddle_size:i32) -> Self{
        let initial_position = Position{x:grid_size_x/2 - paddle_size/2, y:grid_size_y-1};
    
        let initial_velocity = Velocity{x:0, y:0};
        Self {
            position:initial_position, 
            velocity: initial_velocity}
    }
    
    fn update(&mut self, action: &Action){
        self.velocity.x = match action{
            Action::MoveLeft => (self.velocity.x-1).clamp(-2, 2),
            Action::MoveRight => (self.velocity.x+1).clamp(-2, 2),
            Action::StandStill => self.velocity.x
        };

        self.position.x += self.velocity.x;

        if self.position.x <= 0 || self.position.x >= GRID_SIZE_X - PADDLE_LEN{
            self.position.x = self.position.x.clamp(0, GRID_SIZE_X - PADDLE_LEN);
            self.velocity.x = 0;
        };
    }

}

impl Ball{
    fn new(grid_size_x:i32, grid_size_y:i32) -> Self{
        let initial_position = Position{x:grid_size_x/2, y:grid_size_y-1};
        let mut rng = rand::thread_rng();
        let rand_vel_x = rng.gen_range(-2..=2);
        println!("rand starting vel_x: {rand_vel_x}");

        let initial_velocity = Velocity{x:rand_vel_x, y:-1};
        Self {
            position:initial_position, 
            velocity: initial_velocity
        }
    }

    fn update(&mut self, paddle: &Paddle) -> bool{


        self.position.x += self.velocity.x;
        self.position.y += self.velocity.y;


        if self.position.x <= 0 || self.position.x >= GRID_SIZE_X-1 {
            self.velocity.x *= -1
        };
        if self.position.y <= 0 {
            self.velocity.y *= -1
        };
        if self.position.y > GRID_SIZE_Y {
            return true;
        };

        self.position.x = self.position.x.clamp(0, GRID_SIZE_X - 1);
        
        if self.position.y == GRID_SIZE_Y -2 && self.position.x >= paddle.position.x && self.position.x < paddle.position.x + PADDLE_LEN{
            self.velocity.x = self.position.x - paddle.position.x - 2;
            self.velocity.y = -1;
        };

        return false;

    }
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

fn check_for_brick_collision(bricks: &LinkedList<Brick>, ball: &Ball) -> Option<usize>{
    let mut brick_remove_idx:Option<usize> = Option::None;
    for (i, brick) in bricks.iter().enumerate(){
        if ball.position.y == brick.position.y + 1 && ball.position.x >= brick.position.x && ball.position.x < brick.position.x + BRICK_LEN{
            brick_remove_idx = Option::Some(i);
            break;
        } 
    }
    return brick_remove_idx
}

fn remove_brick(bricks: &mut LinkedList<Brick>, colliding_brick_idx: usize){
    let mut split_list = bricks.split_off(colliding_brick_idx);
    split_list.pop_front();
    bricks.append(&mut split_list);
}
// -----> x
// |
// v
// Y
pub fn reset_game(grid_size_x:i32, grid_size_y:i32, paddle_size:i32, brick_size:i32, brick_rows:i32) -> (Ball, Paddle, LinkedList<Brick>) {
    let ball = Ball::new(grid_size_x, grid_size_y);
    let paddle = Paddle::new(grid_size_x, grid_size_y, paddle_size);
    let bricks = initialize_bricks(grid_size_x, brick_size, brick_rows);
    (ball, paddle, bricks)
}




pub fn game_step(paddle: &mut Paddle, ball: &mut Ball, bricks: &mut LinkedList<Brick>, action: &Action) -> GameStatus{
        
    paddle.update(action);

    let ball_is_out_of_bounds = ball.update(&paddle);
    if ball_is_out_of_bounds {
        return GameStatus::ResetGame
    }
    
    if let Some(colliding_brick_idx) = check_for_brick_collision(&bricks, &ball){
        ball.velocity.y = 1;
        remove_brick(bricks, colliding_brick_idx);
    }

    if bricks.len() == 0{
        return GameStatus::GameWon;
    } 
    return GameStatus::Continue;
}