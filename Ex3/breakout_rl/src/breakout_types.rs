use std::collections::LinkedList;
use macroquad::color::*;


pub const GRID_SIZE_X:i32 = 21; 
pub const GRID_SIZE_Y:i32 = 21;
pub const BRICK_ROWS: i32 = 1;
pub const BALL_SIZE:i32 = 1;
pub const PADDLE_LEN:i32 = 5;
pub const BRICK_LEN:i32 = 3;
pub const SCALING_FACTOR:i32 = 20;

#[derive(Debug)]
pub struct Velocity{
    pub x: i32,
    pub y: i32,
}

#[derive(Debug)]
pub struct Position{
    pub x: i32,
    pub y: i32
}

#[derive(Debug)]
pub struct Ball{
    pub position: Position,
    pub velocity: Velocity
}

#[derive(Debug)]
pub struct Paddle{
    pub position: Position,
    pub velocity: Velocity
}

#[derive(Debug)]
pub struct Brick{
    pub position: Position,
    pub color: Color
}
pub struct State{
    pub ball: Ball,
    pub paddle: Paddle,
    pub bricks: LinkedList<Brick>
}

pub enum Action{
    MoveLeft,
    MoveRight,
    StandStill
}

pub struct StateAction{
    pub state: State,
    pub action: Action
}

pub enum GameStatus{
    Continue,
    ResetGame,
    GameWon
}
