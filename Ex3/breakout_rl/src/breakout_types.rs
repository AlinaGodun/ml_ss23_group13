pub const GRID_SIZE_X: i32 = 21;
pub const GRID_SIZE_Y: i32 = 10;
pub const BRICK_ROWS: i32 = 4;
pub const BALL_SIZE: i32 = 1;
pub const PADDLE_LEN: i32 = 5;
pub const BRICK_LEN: i32 = 3;
// Scaling factor for visualization i.e. a 10x10 grid gets scaled to 700x700 pixels if the scaling factor is 70
pub const SCALING_FACTOR: i32 = 70;
pub const BRICKS_LAYOUT: BricksLayout = BricksLayout::RainbowStaircase;

#[derive(Debug, Clone, Eq, Hash, PartialEq)]
pub struct Velocity {
    pub x: i32,
    pub y: i32,
}

#[derive(Debug, Clone, Eq, Hash, PartialEq)]
pub struct Position {
    pub x: i32,
    pub y: i32,
}

#[derive(Debug, Clone, Eq, Hash, PartialEq)]
pub struct Ball {
    pub position: Position,
    pub velocity: Velocity,
}

#[derive(Debug, Clone, Eq, Hash, PartialEq)]
pub struct Paddle {
    pub position: Position,
    pub velocity: Velocity,
}

#[derive(Debug, Clone, Eq, Hash, PartialEq)]
pub struct Brick {
    pub position: Position,
}

#[derive(Clone, Eq, Hash, PartialEq)]
pub enum Action {
    IncreaseVelocityLeft,
    IncreaseVelocityRight,
    DontChangeVelocity,
}

pub enum GameStatus {
    Continue,
    ResetGame,
    GameWon,
}

pub enum BricksLayout {
    PrideRectangle,
    SlavicGrandmaTextil,
    RainbowStaircase,
}


#[derive(Debug)]
pub enum CollisionStatus{
    ResetGame,
    NoCollision,
    Collision(Option<usize>),
}

#[derive(Clone, Debug)]
pub enum CollisionObject {
    BORDER,
    PADDLE,
    NONE,
    BRICK(usize),
}