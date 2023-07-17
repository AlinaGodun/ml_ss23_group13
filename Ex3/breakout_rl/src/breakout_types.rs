pub const GRID_SIZE_X:i32 = 15; 
pub const GRID_SIZE_Y:i32 = 10;
pub const BRICK_ROWS: i32 = 3;
pub const BALL_SIZE:i32 = 1;
pub const PADDLE_LEN:i32 = 5;
pub const BRICK_LEN:i32 = 3;
pub const SCALING_FACTOR:i32 = 100;
pub const BRICKS_LAYOUT:BricksLayout = BricksLayout::PrideRectangle;

#[derive(Debug)]
#[derive(Clone)]
#[derive(Eq, Hash, PartialEq)]
pub struct Velocity{
    pub x: i32,
    pub y: i32,
}

#[derive(Debug)]
#[derive(Clone)]
#[derive(Eq, Hash, PartialEq)]
pub struct Position{
    pub x: i32,
    pub y: i32
}

#[derive(Debug)]
#[derive(Clone)]
#[derive(Eq, Hash, PartialEq)]
pub struct Ball{
    pub position: Position,
    pub velocity: Velocity
}

#[derive(Debug)]
#[derive(Clone)]
#[derive(Eq, Hash, PartialEq)]
pub struct Paddle{
    pub position: Position,
    pub velocity: Velocity
}

#[derive(Debug)]
#[derive(Clone)]
#[derive(Eq, Hash, PartialEq)]
pub struct Brick{
    pub position: Position
}

#[derive(Clone)]
#[derive(Eq, Hash, PartialEq)]
pub enum Action{
    MoveLeft,
    MoveRight,
    StandStill
}

pub enum GameStatus{
    Continue,
    ResetGame,
    GameWon
}

pub enum BricksLayout{
    PrideRectangle,
    SlavicGrandmaTextil,
    RainbowStaircase
}
