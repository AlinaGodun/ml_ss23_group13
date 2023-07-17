use std::hash::Hash;
use std::{collections::HashMap, collections::LinkedList};
use crate::breakout_types::*;
use crate::breakout_game::*;
use rand::{
    distributions::{Distribution, Standard},
    Rng,
};

#[derive(Clone)]
#[derive(Eq, Hash, PartialEq)]
pub struct State{
    ball: Ball,
    paddle: Paddle,
    bricks: LinkedList<Brick>
}

impl State{
    pub fn new(ball: &Ball, paddle: &Paddle, bricks: &LinkedList<Brick>) -> Self{
        Self{
            ball: ball.clone(),
            paddle: paddle.clone(),
            bricks: bricks.clone()
        }
    }
}

#[derive(Clone)]
#[derive(Eq, Hash, PartialEq)]
struct StateAction{
    state: State,
    action: Action
}

impl StateAction{
    fn new(state: &State, action: &Action) -> Self{
        Self{
            state: state.clone(),
            action: action.clone()
        }
    }
}

impl Distribution<Action> for Standard {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Action {
        match rng.gen_range(0..=2) {
            0 => Action::MoveLeft,
            1 => Action::StandStill,
            _ => Action::MoveRight,
        }
    }
}

pub fn take_action(policy: &HashMap<State, Action>, state: &State, epsilon: f32) -> Action{
    let preferred_action = policy.get(state).expect("Some value should have been inserted so this should never fail");
    if epsilon <= 0.0 {
        return preferred_action.clone()
    }

    let mut rng = rand::thread_rng();
    let rand_val = rng.gen_range(0.0..=1.0);
    
    let taken_action;
    if rand_val > epsilon {
        // (prob 1-E)
        taken_action = preferred_action.clone();
    } else {
        // prob(E)
        // prob(E/3) for each value 
        taken_action = rand::random();
    }
    // in total: prob(1-E+E/3) for prefered state
    // prob(E/3) for the two not prefered states 
    taken_action
}

fn generate_episode(policy: &mut HashMap<State, Action>, max_iter: usize, epsilon:f32) -> (LinkedList<State>, LinkedList<Action>, LinkedList<f32>){
    let (mut ball, mut paddle, mut bricks) = reset_game(GRID_SIZE_X, GRID_SIZE_Y, PADDLE_LEN, BRICK_LEN, BRICK_ROWS);
    let mut taken_states: LinkedList<State> = LinkedList::new();
    let mut taken_actions: LinkedList<Action> = LinkedList::new();
    let mut rewards: LinkedList<f32> = LinkedList::new();
    // Episode: max timesteps (e.g. 1000) or game won
    let mut i = 0;
    while i < max_iter {
        let state = State::new(&ball, &paddle, &bricks);
        
        taken_states.push_front(state.clone());
            // randomly initialize if state has not been visited so far
        if !(policy.contains_key(&state)){
            let action: Action = rand::random();
            policy.insert(state.clone(), action);
        }

        let taken_action = take_action(policy, &state, epsilon);

        taken_actions.push_front(taken_action.clone());
        let game_status = game_step(&mut paddle, &mut ball, &mut bricks, &taken_action);
        
        if let GameStatus::GameWon = game_status {
            rewards.push_front(1000.0);
            println!("Game won!!!!!!!!!!");
            break;
        }
        if let GameStatus::ResetGame = game_status {
            rewards.push_front(-500.0);
            (ball, paddle, bricks) = reset_game(GRID_SIZE_X, GRID_SIZE_Y, PADDLE_LEN, BRICK_LEN, BRICK_ROWS);
        } else {
            rewards.push_front(-1.0);
        }
        i += 1;
    }
    println!("Timesteps: {i}");
    (taken_states, taken_actions, rewards)
}

pub fn mc_control_loop(num_episodes: usize, max_iter_per_episode: usize, epsilon: f32) -> HashMap<State, Action>{
    let mut policy:HashMap<State, Action> = HashMap::new();
    let mut state_action_values:HashMap<StateAction, f32> = HashMap::new();
    let mut visited_state_counter : HashMap<StateAction, usize> = HashMap:: new();
    let mut i = 0;

    while i < num_episodes {
        let mut g: f32 = 0.0;
        let mut state_visited_in_episode : HashMap<StateAction, bool> = HashMap:: new();
        let epsilon = epsilon * (-(i as f32)*1.0/num_episodes as f32).exp();
        println!("{epsilon}");

        let (states, actions, rewards) = generate_episode(&mut policy, max_iter_per_episode, epsilon);
        let state = states.front().expect("Some state should be filled");
        let brick_len = state.bricks.len();

        for (state, (action, rewards)) in states.iter().zip(actions.iter().zip(rewards.iter())) {
            g = 0.98*g + rewards;
            let state_action = StateAction::new(&state, &action);

            if !(state_visited_in_episode.get(&state_action).unwrap_or(&false)){
                let state_visited_count = visited_state_counter.entry(state_action.clone()).or_insert(0);
                *state_visited_count += 1;
                let q = state_action_values.entry(state_action.clone()).or_insert(0.0);
                *q += (g - *q)/(*state_visited_count as f32);
                update_policy(&mut policy, &state_action_values, &state);
                state_visited_in_episode.insert(state_action, true);
            }
        }
        i += 1;
        println!("Episode#: {i}, BricksLeft: {brick_len} g: {g}");
        println!("");
    }
    policy
}

fn update_policy(policy: &mut HashMap<State, Action>, state_action_values: &HashMap<StateAction, f32>, state: &State) -> (){
    let preferred_action = get_preferred_action(state_action_values, state);
    policy.insert(state.clone(), preferred_action);
}

fn get_preferred_action(state_action_values: &HashMap<StateAction, f32>, state: &State) -> Action{
    let state_action_left = StateAction::new(state, &Action::MoveLeft);
    let state_action_right = StateAction::new(state, &Action::MoveRight);
    let state_action_still = StateAction::new(state, &Action::StandStill);
    
    let value_left = *state_action_values.get(&state_action_left).unwrap_or(&f32::NEG_INFINITY);
    let value_right = *state_action_values.get(&state_action_right).unwrap_or(&f32::NEG_INFINITY);
    let value_still = *state_action_values.get(&state_action_still).unwrap_or(&f32::NEG_INFINITY);
    
    if value_left >= value_right && value_left >= value_still{
        Action::MoveLeft
    } else if value_right >= value_left && value_right >= value_still {
        Action::MoveRight
    } else {
        Action::StandStill
    }
}