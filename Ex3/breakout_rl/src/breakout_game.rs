use crate::breakout_types::*;
use macroquad::prelude::clamp;
use rand::Rng;
use std::collections::LinkedList;

// -----> x
// |
// v
// Y

impl Paddle {
    fn new(grid_size_x: i32, grid_size_y: i32, paddle_size: i32) -> Self {
        // returns new Paddle object that gets initialized to the bottom center of the grid and with a velocity of 0.
        let initial_position = Position {
            x: grid_size_x / 2 - paddle_size / 2,
            y: grid_size_y - 1,
        };

        let initial_velocity = Velocity { x: 0, y: 0 };
        Self {
            position: initial_position,
            velocity: initial_velocity,
        }
    }

    fn update(&mut self, action: &Action) {
        // update paddle velocity based on selected action, 
        self.velocity.x = match action {
            Action::MoveLeft => (self.velocity.x - 1).clamp(-2, 2),
            Action::MoveRight => (self.velocity.x + 1).clamp(-2, 2),
            Action::StandStill => self.velocity.x,
        };

        // update position based on current velocity
        self.position.x += self.velocity.x;

        // make sure that paddle does not go out of bounds
        if self.position.x <= 0 || self.position.x >= GRID_SIZE_X - PADDLE_LEN {
            self.position.x = self.position.x.clamp(0, GRID_SIZE_X - PADDLE_LEN);
            self.velocity.x = 0;
        };
    }
}


impl Ball {
    fn new(grid_size_x: i32, grid_size_y: i32) -> Self {
        // create new ball with an initial position in the bottom center of the grid (one above paddle)
        let initial_position = Position {
            x: grid_size_x / 2,
            y: grid_size_y - 1,
        };
        let mut rng = rand::thread_rng();
        // get random start velocity in x
        let rand_vel_x = rng.gen_range(-2..=2);

        let initial_velocity = Velocity {
            x: rand_vel_x,
            y: -1,
        };
        Self {
            position: initial_position,
            velocity: initial_velocity,
        }
    }

    fn update(&mut self) {
        // update ball position based on velocity
        self.position.x += self.velocity.x;
        self.position.y += self.velocity.y;
    }
}

fn initialize_bricks_pride_rectangle(
    grid_size_x: i32,
    brick_size: i32,
    brick_rows: i32,
) -> LinkedList<Brick> {
    // creates rows of bricks with contigous bricks:
    // ==================
    // ==================
    // ================== 
    let mut pos_y = 0;
    let mut bricks: LinkedList<Brick> = LinkedList::new();

    loop {
        let mut pos_x = 0;
        if pos_y >= brick_rows {
            break;
        }
        loop {
            if pos_x >= grid_size_x {
                break;
            }
            let position = Position { x: pos_x, y: pos_y };
            let brick = Brick { position };
            bricks.push_back(brick);

            pos_x += brick_size;
        }
        pos_y += 1;
    }
    return bricks;
}

fn initialize_bricks_rainbow_staircase(
    grid_size_x: i32,
    brick_size: i32,
    brick_rows: i32,
) -> LinkedList<Brick> {
    // creates a staircase of bricks:
    // ===    ===
    //  ===    ===
    //   ===    ===
    //    ===    ===
    let mut bricks: LinkedList<Brick> = LinkedList::new();
    let gap_size: i32 = 2;

    for pos_y in 0..brick_rows {
        for pos_x in (pos_y..grid_size_x - brick_size + 1).step_by((brick_size + gap_size) as usize) {
            let position = Position { x: pos_x, y: pos_y };
            let brick = Brick { position };
            bricks.push_back(brick);
        }
    }
    return bricks;
}

fn initialize_bricks_slavic_grandmother_lace() -> LinkedList<Brick> {
    // creates the following pattern:
    //       ===
    // === === === === 
    //   === === ===
    // === === === ===
    //   === === ===
    //     === ===
    //       ===
    let mut bricks: LinkedList<Brick> = LinkedList::new();

    // number of bricks when counting vertically (column wise)
    let brick_nums: [i32; 7] = [2, 2, 3, 4, 3, 2, 2]; 
    // y position of first/upper brick (column wise)
    let y_positions: [i32; 7] = [1, 2, 1, 0, 1, 2, 1];
    let step: usize = 2;

    for (i, (brick_num, y_pos)) in brick_nums.iter().zip(y_positions.iter()).enumerate() {
        for j in (0..brick_num * (step as i32)).step_by(step) {
            let position: Position = Position {
                x: (i as i32 ) * (step as i32) + (GRID_SIZE_X - 15)/2,
                y: y_pos + j,
            };
            let brick: Brick = Brick { position: position };
            bricks.push_back(brick);
        }
    }
    return bricks;
}

fn initialize_bricks(
    grid_size_x: i32,
    brick_size: i32,
    brick_rows: i32,
    bricks_layout: BricksLayout,
) -> LinkedList<Brick> {
    // initialize list of bricks based on chosen brick layout
    match bricks_layout {
        BricksLayout::PrideRectangle => {
            initialize_bricks_pride_rectangle(grid_size_x, brick_size, brick_rows)
        }
        BricksLayout::SlavicGrandmaTextil => {
            initialize_bricks_slavic_grandmother_lace()
        },
        BricksLayout::RainbowStaircase => {
            initialize_bricks_rainbow_staircase(grid_size_x, brick_size, brick_rows)
        }
    }
}

#[allow(dead_code)]
fn print_collision_grid(collision_grid: &[Vec<CollisionObject>]) {
    // print collision grid (for debugging purposes)
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

fn create_collision_grid(
    bricks: &LinkedList<Brick>,
    paddle: &Paddle,
    grid_size_x: i32,
    grid_size_y: i32,
    brick_length: i32,
    paddle_length: i32,
) -> Vec<Vec<CollisionObject>> {
    // Create collision/occupancy grid:
    // Each pixel in the grid is either occupied by paddle, brick, border or not occupied at all
    let grid_size_x = grid_size_x as usize;
    let grid_size_y = grid_size_y as usize;
    let brick_length = brick_length as usize;

    // initialize whole grid as CollisionObject::BORDER
    let mut collision_grid: Vec<Vec<CollisionObject>> =
        vec![vec![CollisionObject::BORDER; grid_size_x+4]; grid_size_y + 2] ;

    // Set non-border pixels as CollisionObject::None
    for row in &mut collision_grid[1..grid_size_y + 1] {
        for cell in &mut row[2..grid_size_x + 2].iter_mut() {
            *cell = CollisionObject::NONE;
        }
    }

    // iterate over all bricks and set the occupied pixels to CollisionObject::Brick
    for (i, brick) in bricks.iter().enumerate() {
        let x = brick.position.x as usize;
        let y = brick.position.y as usize;
        for cell in &mut collision_grid[y + 1][x + 2..x + 2 + brick_length] {
            *cell = CollisionObject::BRICK(i);
        }
    }

    // set the pixels occupied by the paddle to CollisionObject::Paddle
    let x: usize = paddle.position.x as usize;
    let y = paddle.position.y as usize;
    for cell in &mut collision_grid[y + 1][x + 2..x + 2 + paddle_length as usize] {
        *cell = CollisionObject::PADDLE;
    }
    collision_grid
}


fn check_for_collision(ball: &mut Ball, paddle: &Paddle, collision_grid: &[Vec<CollisionObject>]) -> CollisionStatus{
    // check, based on the current ball position and velocity, whether any collision can be found in the collision grid
    // Possible collisions x_vel = 1; y_vel = -1 (B = Ball, number = priority for collision checks, i.e. Ball colides with object 1 before object 2)
    // 2 3
    // B 1
    // Possible collisions x_vel = 2; y_vel = -1
    // 2 3 5
    // B 1 4
    // Possible collisions x_vel = 0; y_vel = -1 
    // 2
    // B

    let y_val = ball.velocity.y/ball.velocity.y.abs();

    // if ball below bottom border -> reset game
    if ball.position.y >= GRID_SIZE_Y-1{
        return CollisionStatus::ResetGame
    }

    // Case 1 (case number refers to the numbers in the comments above): collision one to the side 
    // collision can only happen if x velocity > 0
    if ball.velocity.x.abs() > 0{
        let x_val = ball.velocity.x/ball.velocity.x.abs();
        match collision_grid[(ball.position.y+1) as usize][(ball.position.x+2+x_val) as usize]{
            CollisionObject::BRICK(idx) => {
                ball.velocity.x *= -1;
                // if collision with brick -> return index of colliding brick
                return CollisionStatus::Collision(Some(idx))
            },
            CollisionObject::PADDLE => {
                ball.velocity.x *= -1;
                return CollisionStatus::Collision(None)
            },
            CollisionObject::NONE => {},
            _ => {
                ball.velocity.x *= -1; 
                return CollisionStatus::Collision(None)
            }
        };
    }

    // Case 2: collision one above
    match collision_grid[(ball.position.y+1+y_val) as usize][(ball.position.x+2) as usize]{
        CollisionObject::BRICK(idx) => {
            ball.velocity.y *= -1;
            return CollisionStatus::Collision(Some(idx))
        },
        CollisionObject::PADDLE => {
            ball.velocity.x = ball.position.x-paddle.position.x-2;
            ball.velocity.y *= -1;
            return CollisionStatus::Collision(None)
        },
        CollisionObject::NONE => {},
        _ => {
            ball.velocity.y *= -1; 
            return CollisionStatus::Collision(None)
        }
    };


    // if vel_x = 0 -> ball can only collide in case 2 -> early return
    if ball.velocity.x.abs() == 0 {return CollisionStatus::NoCollision;}

    let x_val = ball.velocity.x/ball.velocity.x.abs();

    // Case 3: collision one to the side and above
    match collision_grid[(ball.position.y+1+y_val) as usize][(ball.position.x+2+x_val) as usize]{
        CollisionObject::BRICK(idx) => {
            ball.velocity.y *= -1;
            return CollisionStatus::Collision(Some(idx))
        },
        CollisionObject::PADDLE => {
            ball.velocity.x = ball.position.x-paddle.position.x-2;
            ball.velocity.y *= -1;
            return CollisionStatus::Collision(None)
        },
        CollisionObject::NONE => {},
        _ => {
            ball.velocity.y *= -1; 
            return CollisionStatus::Collision(None)
        }
    };

    // if ball velocity < 2 => can only colide in cases 1,2,3 -> early return
    if ball.velocity.x.abs() != 2 {return CollisionStatus::NoCollision;}

    // Case 4: collision two to the side
    match collision_grid[(ball.position.y+1) as usize][(ball.position.x+2+x_val*2) as usize]{
        CollisionObject::BRICK(idx) => {
            ball.velocity.x *= -1;
            return CollisionStatus::Collision(Some(idx))
        },
        CollisionObject::PADDLE => {
            ball.velocity.x *= -1;
            return CollisionStatus::Collision(None)
        },
        CollisionObject::NONE => {},
        _ => {
            ball.velocity.x *= -1; 
            return CollisionStatus::Collision(None)
        }
    };

    // Case 5: collision two to the side and above
    match collision_grid[(ball.position.y+1+y_val) as usize][(ball.position.x+2+x_val*2) as usize]{
        CollisionObject::BRICK(idx) => {
            ball.velocity.x *= -1;
            return CollisionStatus::Collision(Some(idx))
        },
        CollisionObject::PADDLE => {
            ball.velocity.x = ball.position.x-paddle.position.x-2;
            ball.velocity.y *= -1;
            return CollisionStatus::Collision(None)
        },
        CollisionObject::NONE => {},
        _ => {
            ball.velocity.x *= -1; 
            return CollisionStatus::Collision(None)
        }
    };
    return CollisionStatus::NoCollision
}

fn remove_brick(bricks: &mut LinkedList<Brick>, colliding_brick_idx: usize) -> Brick {
    // Remove brick with colliding_brick_idx from linked list
    // as the API does not have a .remove function, we have to split the list into two parts and then pop of the
    // first element from the second list 
    // (this element had the colliding_brick_idx in the original list, i.e. is the element that we wanted to delete)
    let mut split_list = bricks.split_off(colliding_brick_idx);
    let brick = split_list
        .pop_front()
        .expect("COLLISION BRICK DOES NOT EXIST");
    bricks.append(&mut split_list);
    return brick;
}

pub fn reset_game(
    grid_size_x: i32,
    grid_size_y: i32,
    paddle_size: i32,
    brick_size: i32,
    brick_rows: i32,
) -> (Ball, Paddle, LinkedList<Brick>) {
    // Reset game state by creating a new ball, a new paddle and new bricks 
    let ball = Ball::new(grid_size_x, grid_size_y);
    let paddle = Paddle::new(grid_size_x, grid_size_y, paddle_size);
    let bricks = initialize_bricks(grid_size_x, brick_size, brick_rows, BRICKS_LAYOUT);
    (ball, paddle, bricks)
}

pub fn game_step(
    paddle: &mut Paddle,
    ball: &mut Ball,
    bricks: &mut LinkedList<Brick>,
    action: &Action,
) -> GameStatus {
    // perform one game step (updating positions and velocities, handling collisions)

    paddle.update(action);
    ball.update();

    loop {
        // create collisiong grid based on current positions and check for collisions afterwards
        let collision_grid = create_collision_grid(
            bricks,
            paddle,
            GRID_SIZE_X,
            GRID_SIZE_Y,
            BRICK_LEN,
            PADDLE_LEN,
        );
        let collision_status = check_for_collision(ball, paddle, &collision_grid);

        // make sure that ball velocity stays in the valid range after collision handling
        ball.velocity.x = clamp(ball.velocity.x, -2, 2);
        // handle edge case
        if ball.position.y + ball.velocity.y < 0{
            ball.velocity.y *= -1;
        }
        
        // if no collision was found -> game step finished
        // if collision found -> updated velocities could lead to a new collision -> repeatedly check until all collisions resolved
        let potential_brick_idx = match collision_status {
            CollisionStatus::ResetGame => return GameStatus::ResetGame,
            CollisionStatus::NoCollision => break,
            CollisionStatus::Collision(potential_idx) => potential_idx 
        };
        
        // if coliding with brick -> remove corresponding brick
        if let Some(idx) = potential_brick_idx {
            remove_brick(bricks, idx);
        };
    }

    // no bricks left -> game won
    if bricks.len() == 0 {
        return GameStatus::GameWon;
    }
    
    // if not won and not reset -> just continue game normally
    return GameStatus::Continue;
}
