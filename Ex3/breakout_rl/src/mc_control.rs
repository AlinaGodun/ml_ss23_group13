use std::{collections::HashMap, hash::Hash};
use crate::breakout_types::*;



fn generate_episode(policy: &HashMap<State, Action>){
    // Episode: max timesteps (e.g. 1000) or game won
    loop {
        //until end of episode
        game.step()
    }

}

fn mc_control_loop(){
    let policy:HashMap<State, Action> = HashMap::new();
    let state_action_values:HashMap<StateAction, f32> = HashMap::new();
    let mut G = 0.0;
    
    loop {

        G = 0.0;
    }

}

