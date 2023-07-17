use std::collections::LinkedList;
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

fn initialize_bricks_pride_rectangle(grid_size_x:i32, brick_size:i32, brick_rows: i32) -> LinkedList<Brick>{
    let mut pos_y = 0;
    let mut bricks: LinkedList<Brick> = LinkedList::new();

    loop {    
        let mut pos_x = 0;
        if pos_y >= brick_rows {break;}
        loop{
            if pos_x >= grid_size_x {break;}
            let position =  Position{x: pos_x, y:pos_y};
            let brick = Brick{position};
            bricks.push_back(brick);

            pos_x += brick_size;
        }
        pos_y += 1;
    }
    return bricks;
}

fn initialize_bricks_rainbow_staircase(grid_size_x:i32, brick_size:i32, brick_rows: i32) -> LinkedList<Brick>{
    let mut bricks: LinkedList<Brick> = LinkedList::new();
    let gap_size: i32 = 2;

    for pos_y in 0..brick_rows {
        for pos_x in (pos_y..grid_size_x).step_by((brick_size + gap_size) as usize) {
            let position =  Position{x: pos_x, y:pos_y};
            let brick = Brick{position};
            bricks.push_back(brick);
        }
    }

    return bricks;
}

fn initialize_bricks_slavic_grandmother_lace() -> LinkedList<Brick>{
    let mut bricks: LinkedList<Brick> = LinkedList::new();

    let brick_nums: [i32;7] = [2, 2, 3, 4, 3, 2, 2];
    let y_positions: [i32;7] = [1, 2, 1, 0, 1, 2, 1];
    let step: usize = 2;

    for (i, (brick_num, y_pos)) in brick_nums.iter().zip(y_positions.iter()).enumerate() {
        for j in (0..brick_num*(step as i32)).step_by(step) {
            let position: Position = Position { x: (i as i32) * (step as i32), y: y_pos + j};
            let brick: Brick = Brick { position: position };
            bricks.push_back(brick);
        }
    }
    
    return bricks;
}

fn initialize_bricks(grid_size_x:i32, brick_size:i32, brick_rows: i32, bricks_layout: BricksLayout) -> LinkedList<Brick>{
    match bricks_layout {
        BricksLayout::PrideRectangle => initialize_bricks_pride_rectangle(grid_size_x, brick_size, brick_rows),
        BricksLayout::SlavicGrandmaTextil => initialize_bricks_slavic_grandmother_lace(),
        BricksLayout::RainbowStaircase => initialize_bricks_rainbow_staircase(grid_size_x, brick_size, brick_rows)
    }
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
    let bricks = initialize_bricks(grid_size_x, brick_size, brick_rows, BRICKS_LAYOUT);
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