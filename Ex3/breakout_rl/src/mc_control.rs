use crate::breakout_game::*;
use crate::breakout_types::*;
use rand::{
    distributions::{Distribution, Standard},
    Rng,
};
use std::hash::Hash;
use std::{collections::HashMap, collections::LinkedList};

#[derive(Clone, Eq, Hash, PartialEq)]
pub struct State {
    ball: Ball,
    paddle: Paddle,
    bricks: LinkedList<Brick>,
}

impl State {
    pub fn new(ball: &Ball, paddle: &Paddle, bricks: &LinkedList<Brick>) -> Self {
        Self {
            ball: ball.clone(),
            paddle: paddle.clone(),
            bricks: bricks.clone(),
        }
    }
}

#[derive(Clone, Eq, Hash, PartialEq)]
struct StateAction {
    state: State,
    action: Action,
}

impl StateAction {
    fn new(state: &State, action: &Action) -> Self {
        Self {
            state: state.clone(),
            action: action.clone(),
        }
    }
}

impl Distribution<Action> for Standard {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Action {
        // function to randomly decide on action
        match rng.gen_range(0..=2) {
            0 => Action::IncreaseVelocityLeft,
            1 => Action::DontChangeVelocity,
            _ => Action::IncreaseVelocityRight,
        }
    }
}

pub fn take_action(policy: &HashMap<State, Action>, state: &State, epsilon: f32) -> Action {
    // get preferred/optimal action if one exists in policy
    let preferred_action = if policy.contains_key(state) { 
        policy.get(state).expect("This shouldn't be possible!").clone() 
    } else { 
        rand::random() 
    };
    
    // if epsilon = 0 -> no exploration -> just take preferred action
    if epsilon <= 0.0 {
        return preferred_action.clone();
    }

    let mut rng = rand::thread_rng();
    // draw number from uniform distribution between 0.0 and 1.0
    let rand_val = rng.gen_range(0.0..=1.0);

    let taken_action;
    if rand_val > epsilon {
        // (prob 1-E) of this case happening
        taken_action = preferred_action.clone();
    } else {
        // prob(E) of this case happening
        // prob(E/3) for each value as we randomly decide on one of the three actions
        taken_action = rand::random();
    }
    // in total: prob(1-E+E/3) for prefered state
    // prob(E/3) for the two not prefered states
    taken_action
}

fn generate_episode(
    policy: &mut HashMap<State, Action>,
    max_iter: usize,
    epsilon: f32,
) -> (LinkedList<State>, LinkedList<Action>, LinkedList<f32>) {
    // start game
    let (mut ball, mut paddle, mut bricks) =
        reset_game(GRID_SIZE_X, GRID_SIZE_Y, PADDLE_LEN, BRICK_LEN, BRICK_ROWS);

    let mut taken_states: LinkedList<State> = LinkedList::new();
    let mut taken_actions: LinkedList<Action> = LinkedList::new();
    let mut rewards: LinkedList<f32> = LinkedList::new();

    // Episode: max timesteps (e.g. 1000) or game won
    let mut i = 0;
    while i < max_iter {
        let state = State::new(&ball, &paddle, &bricks);

        // we always push the taken states, taken actions and rewards to the front 
        // -> if we iterate in forward direction, we replay the game in reversed direction
        taken_states.push_front(state.clone());
        // randomly generate action if state has not been visited yet
        if !(policy.contains_key(&state)) {
            let action: Action = rand::random();
            policy.insert(state.clone(), action);
        }

        let taken_action = take_action(policy, &state, epsilon);

        taken_actions.push_front(taken_action.clone());

        // advance game by one step
        let game_status = game_step(&mut paddle, &mut ball, &mut bricks, &taken_action);

        if let GameStatus::GameWon = game_status {
            // reward model if game is won
            rewards.push_front(1000.0);
            break;
        }
        if let GameStatus::ResetGame = game_status {
            // penalize model if game is reset -> model quickly learns to prefer actions that don't reset games
            rewards.push_front(-500.0);
            (ball, paddle, bricks) =
                reset_game(GRID_SIZE_X, GRID_SIZE_Y, PADDLE_LEN, BRICK_LEN, BRICK_ROWS);
        } else {
            // penalize each time step
            rewards.push_front(-1.0);
        }
        i += 1;
    }
    // return all visited states, taken actions and rewards fore each state
    (taken_states, taken_actions, rewards)
}

pub fn mc_control_loop(
    num_episodes: usize,
    max_iter_per_episode: usize,
    epsilon: f32,
) -> HashMap<State, Action> {
    let mut policy: HashMap<State, Action> = HashMap::new();
    let mut state_action_values: HashMap<StateAction, f32> = HashMap::new();
    let mut visited_state_counter: HashMap<StateAction, usize> = HashMap::new();
    let mut i = 0;

    // train model for num_episodes
    while i < num_episodes {
        let mut g: f32 = 0.0;
        let mut state_visited_in_episode: HashMap<StateAction, bool> = HashMap::new();
        // slightly decrease epsilon as game progresses
        let epsilon = epsilon * (-(i as f32) * 1.0 / num_episodes as f32).exp();

        // generate episode
        let (states, actions, rewards) =
            generate_episode(&mut policy, max_iter_per_episode, epsilon);

        // iterate over states, actions and rewards in reverse direction
        for (state, (action, rewards)) in states.iter().zip(actions.iter().zip(rewards.iter())) {
            // discount factor to weigh recent rewards more heavily
            g = 0.98 * g + rewards;
            let state_action = StateAction::new(&state, &action);
            
            // if first visit to state action in this episode -> update state action values and update policy for this state
            if !(state_visited_in_episode
                .get(&state_action)
                .unwrap_or(&false))
            {
                let state_visited_count = visited_state_counter
                    .entry(state_action.clone())
                    .or_insert(0);
                *state_visited_count += 1;
                let q = state_action_values
                    .entry(state_action.clone())
                    .or_insert(0.0);
                // state action values get increased incrementally (more efficient than saving all rewards and calculating the average each time)
                *q += (g - *q) / (*state_visited_count as f32);
                update_policy(&mut policy, &state_action_values, &state);
                state_visited_in_episode.insert(state_action, true);
            }
        }
        i += 1;
        if i % 100 == 0 {
            // print progress 'bar'
            println!("Episode#: {i}");
        }
    }
    policy
}

fn update_policy(
    policy: &mut HashMap<State, Action>,
    state_action_values: &HashMap<StateAction, f32>,
    state: &State,
) -> () {
    // update policy for passed state based on the action with the highest state-action-value
    let preferred_action = get_preferred_action(state_action_values, state);
    policy.insert(state.clone(), preferred_action);
}

fn get_preferred_action(state_action_values: &HashMap<StateAction, f32>, state: &State) -> Action {
    let state_action_left = StateAction::new(state, &Action::IncreaseVelocityLeft);
    let state_action_right = StateAction::new(state, &Action::IncreaseVelocityRight);
    let state_action_still = StateAction::new(state, &Action::DontChangeVelocity);

    // lookup state-action-values of all actions for the passed state
    let value_left = *state_action_values
        .get(&state_action_left)
        .unwrap_or(&f32::NEG_INFINITY);
    let value_right = *state_action_values
        .get(&state_action_right)
        .unwrap_or(&f32::NEG_INFINITY);
    let value_still = *state_action_values
        .get(&state_action_still)
        .unwrap_or(&f32::NEG_INFINITY);

    // depending on which value is biggest -> return the corresponding action as the preferred action
    if value_left >= value_right && value_left >= value_still {
        Action::IncreaseVelocityLeft
    } else if value_right >= value_left && value_right >= value_still {
        Action::IncreaseVelocityRight
    } else {
        Action::DontChangeVelocity
    }
}
