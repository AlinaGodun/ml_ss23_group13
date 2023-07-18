use crate::breakout_types::*;
use macroquad::prelude::clamp;
use rand::Rng;
use std::collections::LinkedList;

#[derive(Clone, Debug)]
pub enum CollisionObject {
    BORDER,
    PADDLE,
    NONE,
    BRICK(usize),
}

impl Paddle {
    fn new(grid_size_x: i32, grid_size_y: i32, paddle_size: i32) -> Self {
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
        self.velocity.x = match action {
            Action::MoveLeft => (self.velocity.x - 1).clamp(-2, 2),
            Action::MoveRight => (self.velocity.x + 1).clamp(-2, 2),
            Action::StandStill => self.velocity.x,
        };

        self.position.x += self.velocity.x;

        if self.position.x <= 0 || self.position.x >= GRID_SIZE_X - PADDLE_LEN {
            self.position.x = self.position.x.clamp(0, GRID_SIZE_X - PADDLE_LEN);
            self.velocity.x = 0;
        };
    }
}

impl Ball {
    fn new(grid_size_x: i32, grid_size_y: i32) -> Self {
        let initial_position = Position {
            x: grid_size_x / 2,
            y: grid_size_y - 1,
        };
        let mut rng = rand::thread_rng();
        let rand_vel_x = rng.gen_range(-2..=2);
        // let rand_vel_x = 0;

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
        self.position.x += self.velocity.x;
        self.position.y += self.velocity.y;
    }
}

fn initialize_bricks_pride_rectangle(
    grid_size_x: i32,
    brick_size: i32,
    brick_rows: i32,
) -> LinkedList<Brick> {
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
    let mut bricks: LinkedList<Brick> = LinkedList::new();
    let gap_size: i32 = 2;

    for pos_y in 0..brick_rows {
        for pos_x in (pos_y..grid_size_x).step_by((brick_size + gap_size) as usize) {
            let position = Position { x: pos_x, y: pos_y };
            let brick = Brick { position };
            bricks.push_back(brick);
        }
    }

    return bricks;
}

fn initialize_bricks_slavic_grandmother_lace() -> LinkedList<Brick> {
    let mut bricks: LinkedList<Brick> = LinkedList::new();

    let brick_nums: [i32; 7] = [2, 2, 3, 4, 3, 2, 2];
    let y_positions: [i32; 7] = [1, 2, 1, 0, 1, 2, 1];
    let step: usize = 2;

    for (i, (brick_num, y_pos)) in brick_nums.iter().zip(y_positions.iter()).enumerate() {
        for j in (0..brick_num * (step as i32)).step_by(step) {
            let position: Position = Position {
                x: (i as i32) * (step as i32),
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
    match bricks_layout {
        BricksLayout::PrideRectangle => {
            initialize_bricks_pride_rectangle(grid_size_x, brick_size, brick_rows)
        }
        BricksLayout::SlavicGrandmaTextil => initialize_bricks_slavic_grandmother_lace(),
        BricksLayout::RainbowStaircase => {
            initialize_bricks_rainbow_staircase(grid_size_x, brick_size, brick_rows)
        }
    }
}

#[allow(dead_code)]
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

fn create_collision_grid(
    bricks: &LinkedList<Brick>,
    paddle: &Paddle,
    grid_size_x: i32,
    grid_size_y: i32,
    brick_length: i32,
    paddle_length: i32,
) -> Vec<Vec<CollisionObject>> {
    let grid_size_x = grid_size_x as usize;
    let grid_size_y = grid_size_y as usize;
    let brick_length = brick_length as usize;
    let mut collision_grid: Vec<Vec<CollisionObject>> =
        vec![vec![CollisionObject::BORDER; grid_size_x+4]; grid_size_y + 2] ;

    // println!(
    //     "{grid_size_x}, {grid_size_y}, {0}, {1}",
    //     collision_grid.len(),
    //     collision_grid[0].len()
    // );
    for row in &mut collision_grid[1..grid_size_y + 1] {
        for cell in &mut row[2..grid_size_x + 2].iter_mut() {
            *cell = CollisionObject::NONE;
        }
    }

    for (i, brick) in bricks.iter().enumerate() {
        let x = brick.position.x as usize;
        let y = brick.position.y as usize;
        for cell in &mut collision_grid[y + 1][x + 2..x + 2 + brick_length] {
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

enum CollisionStatus{
    ResetGame,
    NoCollision,
    Collision(Option<usize>),
}

fn check_for_collision(ball: &mut Ball, paddle: &Paddle, collision_grid: &[Vec<CollisionObject>]) -> CollisionStatus{
    let y_val = ball.velocity.y/ball.velocity.y.abs();

    if ball.position.y >= GRID_SIZE_Y-1{
        return CollisionStatus::ResetGame
    }

    // if ball.position.x <= 0{
    //     ball.velocity.x *= -1;
    //     return (Option::None, reset)
    // }

    // collision one to the side
    if ball.velocity.x.abs() > 0{
        let x_val = ball.velocity.x/ball.velocity.x.abs();
        match collision_grid[(ball.position.y+1) as usize][(ball.position.x+2+x_val) as usize]{
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
    }

    // collision one above
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


    if ball.velocity.x.abs() == 0 {return CollisionStatus::NoCollision;}

    let x_val = ball.velocity.x/ball.velocity.x.abs();

    // collision one to the side and above
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
    if ball.velocity.x.abs() != 2 {return CollisionStatus::NoCollision;}

    // if x == 2 {ball.position.x += ball.velocity.x/ball.velocity.x.abs()}

    // collision two to the side
    match collision_grid[(ball.position.y+1) as usize][(ball.position.x+2+x_val*2) as usize]{
        CollisionObject::BRICK(idx) => {
            ball.velocity.x *= -1;
            // ball.position.x += x_val;
            return CollisionStatus::Collision(Some(idx))
        },
        CollisionObject::PADDLE => {
            ball.velocity.x *= -1;
            return CollisionStatus::Collision(None)
        },
        CollisionObject::NONE => {},
        _ => {
            ball.velocity.x *= -1; 
            // ball.position.x += x_val;
            return CollisionStatus::Collision(None)
        }
    };

    // collision two to the side and above
    match collision_grid[(ball.position.y+1+y_val) as usize][(ball.position.x+2+x_val*2) as usize]{
        CollisionObject::BRICK(idx) => {
            ball.velocity.y *= -1;
            // ball.position.x += x_val;
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
            // ball.position.x += x_val;
            return CollisionStatus::Collision(None)
        }
    };
    return CollisionStatus::NoCollision
}

fn remove_brick(bricks: &mut LinkedList<Brick>, colliding_brick_idx: usize) -> Brick {
    let mut split_list = bricks.split_off(colliding_brick_idx);
    let brick = split_list
        .pop_front()
        .expect("COLLISION BRICK DOES NOT EXIST");
    bricks.append(&mut split_list);
    return brick;
}
// -----> x
// |
// v
// Y
pub fn reset_game(
    grid_size_x: i32,
    grid_size_y: i32,
    paddle_size: i32,
    brick_size: i32,
    brick_rows: i32,
) -> (Ball, Paddle, LinkedList<Brick>) {
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
    paddle.update(action);

    ball.update();
    loop {
        let collision_grid = create_collision_grid(
            bricks,
            paddle,
            GRID_SIZE_X,
            GRID_SIZE_Y,
            BRICK_LEN,
            PADDLE_LEN,
        ); //todo param not const
        let collision_status = check_for_collision(ball, paddle, &collision_grid);
        ball.velocity.x = clamp(ball.velocity.x, -2, 2);
        if ball.position.y + ball.velocity.y < 0{
            ball.velocity.y *= -1;
        }
        let potential_brick_idx = match collision_status {
            CollisionStatus::ResetGame => return GameStatus::ResetGame,
            CollisionStatus::NoCollision => break,
            CollisionStatus::Collision(potential_idx) => potential_idx 
        };

        if let Some(idx) = potential_brick_idx {
            remove_brick(bricks, idx);
        };
        // }
    }

    if bricks.len() == 0 {
        return GameStatus::GameWon;
    }
    return GameStatus::Continue;
}
