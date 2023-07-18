use std::collections::LinkedList;
use rand::Rng;
use crate::breakout_types::*;


#[derive(Clone)]
#[derive(Debug)]
pub enum CollisionObject{
    BORDER,
    PADDLE,
    NONE,
    BRICK(usize)
}

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
        // let rand_vel_x = rng.gen_range(-2..=2);
        let rand_vel_x = -1;

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

fn print_collision_grid(collision_grid: &[Vec<CollisionObject>]) {
    for row in collision_grid.iter() {
        for cell in row.iter() {
            let cell_symbol = match cell {
                CollisionObject::NONE => " ",
                CollisionObject::BRICK(_) => "B",
                CollisionObject::PADDLE => "P",
                CollisionObject::BORDER => "#",
            };
            print!("{}", cell_symbol);
        }
        println!();
    }
}

fn create_collision_grid (
    bricks: &LinkedList<Brick>, 
    paddle: &Paddle, 
    grid_size_x:i32, 
    grid_size_y:i32, 
    brick_length:i32, 
    paddle_length:i32
) -> Vec<Vec<CollisionObject>> {
    let grid_size_x = grid_size_x as usize;
    let grid_size_y = grid_size_y as usize;
    let brick_length = brick_length as usize;
    let mut collision_grid:Vec<Vec<CollisionObject>> = vec![vec![CollisionObject::BORDER; grid_size_x + 4]; grid_size_y + 2];

    println!("{grid_size_x}, {grid_size_y}, {0}, {1}", collision_grid.len(), collision_grid[0].len());
    for row in &mut collision_grid[1..grid_size_y+1] {
        for cell in &mut row[2..grid_size_x+2].iter_mut(){
            *cell = CollisionObject::NONE;
        }
    }
    
    for (i, brick) in bricks.iter().enumerate() {
        let x = brick.position.x as usize;
        let y = brick.position.y as usize;
        for cell in &mut collision_grid[y+ 1][x + 2..x + 2 + brick_length] {
            *cell = CollisionObject::BRICK(i);
        }
    }

    let x: usize = paddle.position.x as usize;
    let y = paddle.position.y as usize;
    for cell in &mut collision_grid[y + 1][x + 2..x + 2 + paddle_length as usize] {
        *cell = CollisionObject::PADDLE;
    }
    // print_collision_grid(&collision_grid);
    // println!("{collision_grid:?}");
    collision_grid
}

// fn check_collision_env(x: i32, y: i32, ball: &Ball, obj_id: i32) {
    
// }

fn check_for_brick_collision(bricks: &LinkedList<Brick>, ball: &Ball) -> (Option<i32>, Option<i32>, Option<i32>){

    let mut collision_env:[[Option<i32>; 3]; 5] = [[Option::None; 3];5];

    let mut x: i32 ;
    let mut y;
    //1,2,3,4,5 array // save index 
    for (i, brick) in bricks.iter().enumerate(){
        for j in 0..BRICK_LEN{
            x = brick.position.x+j - ball.position.x;
            y = brick.position.y - ball.position.y;
            
            let i = i as i32;

            if x == 0 && y*ball.velocity.y == 1 {
                collision_env[1][0] = Option::Some(i);
                collision_env[1][1] = Option::Some(x);
                collision_env[1][2] = Option::Some(y);
            } 
            if ball.velocity.x.abs() > 0{
                if x*ball.velocity.x/ball.velocity.x.abs() == 1 && y == 0 {
                    collision_env[0][0] = Option::Some(i);
                    collision_env[0][1] = Option::Some(x);
                    collision_env[0][2] = Option::Some(y);
                };
                if x*ball.velocity.x/ball.velocity.x.abs() == 1 && y*ball.velocity.y == 1 {
                    collision_env[2][0] = Option::Some(i);
                    collision_env[2][1] = Option::Some(x);
                    collision_env[2][2] = Option::Some(y);
                };
                if x*ball.velocity.x/ball.velocity.x.abs() == 2 && y == 0 {
                    collision_env[3][0] = Option::Some(i);
                    collision_env[3][1] = Option::Some(x);
                    collision_env[3][2] = Option::Some(y);
                };
                if x*ball.velocity.x/ball.velocity.x.abs() == 2 && y*ball.velocity.y == 1 {
                    collision_env[4][0] = Option::Some(i);
                    collision_env[4][1] = Option::Some(x);
                    collision_env[4][2] = Option::Some(y);
                };
            }

        }
    }

    if ball.velocity.x == 0 {
        return (collision_env[1][0], collision_env[1][1], collision_env[1][2])
    }

    if ball.velocity.x == 1 {
        for collision in &collision_env[0..3]{
            if let Some(_) = collision[0] {
                return (collision[0], collision[1], collision[2])
            }
        }        
    }

    if ball.velocity.x == 2 {
        for collision in collision_env{
            if let Some(_) = collision[0] {
                return (collision[0], collision[1], collision[2])
            }
        }        
    }

    return (Option::None,Option::None,Option::None)
}

fn remove_brick(bricks: &mut LinkedList<Brick>, colliding_brick_idx: usize) -> Brick{
    let mut split_list = bricks.split_off(colliding_brick_idx);
    let brick = split_list.pop_front().expect("COLLISION BRICK DOES NOT EXIST");
    bricks.append(&mut split_list);
    return brick
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
    println!("x_v:{: <10}, y_v:{: <10}", ball.velocity.x, ball.velocity.y);

    loop {
        create_collision_grid(bricks, paddle, GRID_SIZE_X, GRID_SIZE_Y, BRICK_LEN, PADDLE_LEN); //todo param not const
        let (colliding_brick_idx, x, y) = check_for_brick_collision(&bricks, &ball);
        if let None = colliding_brick_idx {println!("+"); break;};
        let removed_brick = remove_brick(bricks, colliding_brick_idx.unwrap() as usize);
        //let x: i32 = removed_brick.position.x-ball.position.x;
        // let y: i32 = removed_brick.position.y-ball.position.y;
        // let x: i32 = removed_brick.position.x-ball.position.x;
        let x = x.unwrap();
        let y = y.unwrap();
        println!("x:{: <10}, y:{: <10}, x_v:{: <10}, y_v:{: <10}", x, y, ball.velocity.x, ball.velocity.y);

        // if x == 2 {ball.position.x += ball.velocity.x/ball.velocity.x.abs()}
        if y == 0 {ball.velocity.x *= -1}
        if y != 0 {ball.velocity.y *=-1}
    } 

    if bricks.len() == 0{
        return GameStatus::GameWon;
    } 
    return GameStatus::Continue;
}